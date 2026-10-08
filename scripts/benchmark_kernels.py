"""Experimental CUDA float32 kernel feasibility benchmark; not a production backend.

Fused skinning forward retains the existing PyTorch backward. Rotation fusion is
forward-only and is tested only for inference. This script never changes the package
default and makes no vmap/compile/AMP compatibility claim for its experimental kernels.
"""

import argparse
import hashlib
import json
import statistics
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
from benchmark_fitting_anatomy import FittingCase

import manotorch.manolayer as mano_module
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, _apply_transforms, _SkinApply
from manotorch.utils.geometry import axis_angle_to_matrix

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def skin_kernel(T, P, Output, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        vertex, row = i // 3, i % 3
        t = vertex * 12 + row * 4
        p = vertex * 3
        x = tl.load(P + p, mask, 0)
        y = tl.load(P + p + 1, mask, 0)
        z = tl.load(P + p + 2, mask, 0)
        a = tl.load(T + t, mask, 0)
        b = tl.load(T + t + 1, mask, 0)
        c = tl.load(T + t + 2, mask, 0)
        d = tl.load(T + t + 3, mask, 0)
        value = tl.fma(a, x, d)
        value = tl.fma(b, y, value)
        value = tl.fma(c, z, value)
        tl.store(Output + i, value, mask)

    @triton.jit
    def rotation_kernel(P, Output, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < N
        x = tl.load(P + i * 3, mask, 0)
        y = tl.load(P + i * 3 + 1, mask, 0)
        z = tl.load(P + i * 3 + 2, mask, 0)
        s = x * x + y * y + z * z
        a_t = tl.sqrt(tl.maximum(s, 1e-4))
        a = tl.where(s < 1e-4, 1 - s * (1 / 6 - s / 120), tl.sin(a_t) / a_t)
        q = s * 0.25
        b_t = tl.sqrt(tl.maximum(q, 1e-4))
        half = tl.where(q < 1e-4, 1 - q * (1 / 6 - q / 120), tl.sin(b_t) / b_t)
        b = 0.5 * half * half
        d = 1 - s * b
        tl.store(Output + i * 9, b * x * x + d, mask)
        tl.store(Output + i * 9 + 1, b * x * y - a * z, mask)
        tl.store(Output + i * 9 + 2, b * x * z + a * y, mask)
        tl.store(Output + i * 9 + 3, b * y * x + a * z, mask)
        tl.store(Output + i * 9 + 4, b * y * y + d, mask)
        tl.store(Output + i * 9 + 5, b * y * z - a * x, mask)
        tl.store(Output + i * 9 + 6, b * z * x - a * y, mask)
        tl.store(Output + i * 9 + 7, b * z * y + a * x, mask)
        tl.store(Output + i * 9 + 8, b * z * z + d, mask)


class ExperimentalSkin(_SkinApply):
    @staticmethod
    def forward(T, points):
        if not (T.is_cuda and T.dtype == torch.float32 and points.dtype == torch.float32
                and T.is_contiguous() and points.is_contiguous()):
            return _apply_transforms(T, points)
        out = torch.empty_like(points)
        skin_kernel[(triton.cdiv(points.numel(), 128),)](T, points, out, points.numel(), 128)
        return out


def rotation_forward(points):
    if not (points.is_cuda and points.dtype == torch.float32 and points.is_contiguous()):
        raise ValueError("rotation prototype requires contiguous CUDA float32")
    out = torch.empty((*points.shape[:-1], 3, 3), device=points.device, dtype=points.dtype)
    n = points.numel() // 3
    rotation_kernel[(triton.cdiv(n, 128),)](points, out, n, 128, enable_fp_fusion=False)
    return out


@contextmanager
def backend(skin=False, rotation=False):
    old_skin, old_rotation = mano_module._SkinApply, mano_module.axis_angle_to_matrix
    try:
        if skin:
            mano_module._SkinApply = ExperimentalSkin
        if rotation:
            mano_module.axis_angle_to_matrix = rotation_forward
        yield
    finally:
        mano_module._SkinApply, mano_module.axis_angle_to_matrix = old_skin, old_rotation


def measure(fn, repeats):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def interleave(methods, groups, repeats):
    result = {name: [] for name in methods}
    names = list(methods)
    for fn in methods.values():
        for _ in range(5):
            fn()
    for group in range(groups):
        for name in names[group % len(names):] + names[:group % len(names)]:
            result[name].append(measure(methods[name], repeats))
    return {"groups_ms": result, "median_ms": {n: statistics.median(v) for n, v in result.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano-assets-root", type=Path, default=Path("assets/mano"))
    parser.add_argument("--targets", type=Path, default=Path("data/benchmarks/mano_targets.npz"))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 128, 1024])
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/kernel_feasibility.json"))
    args = parser.parse_args()
    if triton is None or not torch.cuda.is_available():
        parser.error("this optional experiment requires Triton and a CUDA GPU")
    if min(*args.batches, args.groups, args.repeats, args.steps) <= 0:
        parser.error("batches, groups, repeats and steps must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    arrays = np.load(args.targets, allow_pickle=False)
    result = {"torch": torch.__version__, "triton": triton.__version__,
              "gpu": torch.cuda.get_device_name(), "groups": args.groups, "steps": args.steps,
              "repeats": args.repeats, "threads": 4,
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (args.targets, Path(__file__), Path("scripts/benchmark_fitting_anatomy.py"),
                          Path("manotorch/manolayer.py"), Path("manotorch/utils/geometry.py"),
                          Path("manotorch/axislayer.py"), Path("manotorch/anatomy_loss.py"))},
              "scope": "experimental CUDA float32; native backward for skin; rotation inference only",
              "micro": [], "full": [], "fitting": []}
    # Exercise both Taylor boundaries, zero and larger angles outside the timed blocks.
    probes = torch.tensor([0, 1e-8, .009, .01, .011, .019, .02, .021, 1, 3.14159265], device="cuda")
    probes = torch.stack((probes, torch.zeros_like(probes), torch.zeros_like(probes)), -1)
    torch.testing.assert_close(rotation_forward(probes), axis_angle_to_matrix(probes), atol=3e-7, rtol=3e-6)
    axis = AxisLayerFK().cuda()
    for batch in args.batches:
        pose = (torch.randn(batch, 48, device="cuda") * 0.3).requires_grad_()
        betas = torch.randn(batch, 10, device="cuda", requires_grad=True)
        rotation = pose.detach().reshape(batch, 16, 3)
        T = torch.randn(batch, 778, 3, 4, device="cuda")
        points = torch.randn(batch, 778, 3, device="cuda")
        torch.testing.assert_close(ExperimentalSkin.apply(T, points), _SkinApply.apply(T, points), atol=0, rtol=0)
        torch.testing.assert_close(rotation_forward(rotation), axis_angle_to_matrix(rotation), atol=3e-7, rtol=3e-6)
        compiled_rotation = torch.compile(axis_angle_to_matrix, fullgraph=True)
        compiled_skin = torch.compile(_apply_transforms, fullgraph=True)
        torch.testing.assert_close(compiled_rotation(rotation), axis_angle_to_matrix(rotation), atol=3e-7, rtol=3e-6)
        torch.testing.assert_close(compiled_skin(T, points), _apply_transforms(T, points), atol=1e-6, rtol=1e-6)
        for label, methods in (
            ("rotation_forward", {"eager": partial(axis_angle_to_matrix, rotation),
                                  "compile": partial(compiled_rotation, rotation),
                                  "triton": partial(rotation_forward, rotation)}),
            ("skin_forward", {"eager": partial(_SkinApply.apply, T, points),
                              "compile": partial(compiled_skin, T, points),
                              "triton": partial(ExperimentalSkin.apply, T, points)}),
        ):
            result["micro"].append({"batch": batch, "operation": label,
                                    **interleave(methods, args.groups, args.repeats)})
        layer = ManoLayer(mano_assets_root=str(args.mano_assets_root)).cuda()
        reference = layer(pose, betas).verts
        grad_ref = torch.autograd.grad(reference.square().mean(), (pose, betas))
        with backend(skin=True):
            output = layer(pose, betas).verts
            grads = torch.autograd.grad(output.square().mean(), (pose, betas))
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        for a, b in zip(grads, grad_ref, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

        def forward(use_skin=False, use_rotation=False, backward=False, layer=layer, pose=pose, betas=betas):
            with backend(skin=use_skin, rotation=use_rotation):
                if use_rotation:
                    with torch.no_grad():
                        return layer(pose, betas).verts
                out = layer(pose, betas).verts
                if backward:
                    torch.autograd.grad(out.square().mean(), (pose, betas))
                return out

        def inference(layer=layer, pose=pose, betas=betas):
            with torch.no_grad():
                return layer(pose, betas).verts

        with backend(rotation=True), torch.no_grad():
            delta = (layer(pose, betas).verts - reference).abs().max().item() * 1000
        if delta > 1e-3:
            raise RuntimeError(f"rotation prototype geometry error {delta} mm")
        result["full"].append({"batch": batch, "rotation_vertex_max_mm": delta,
            "forward": interleave({"current": forward, "skin_triton": partial(forward, use_skin=True)},
                                   args.groups, args.repeats),
            "forward_backward": interleave({"current": partial(forward, backward=True),
                                            "skin_triton": partial(forward, use_skin=True, backward=True)},
                                           args.groups, args.repeats),
            "inference": interleave({"current": inference, "rotation_triton": partial(forward, use_rotation=True)},
                                     args.groups, args.repeats)})
        for weight in (0.0, 1e-4):
            cases = {name: FittingCase((ManoLayer, AxisLayerFK, AnatomyConstraintLossEE), args, arrays,
                                      batch, torch.device("cuda"), axis) for name in ("current", "skin_triton")}
            timings = {name: [] for name in cases}
            metrics = {}
            for name, case in cases.items():
                with backend(skin=name == "skin_triton"):
                    for _ in range(10):
                        case.step(weight)
            for group in range(args.groups):
                names = list(cases) if group % 2 == 0 else list(reversed(cases))
                for name in names:
                    case = cases[name]
                    case.reset()
                    with backend(skin=name == "skin_triton"):
                        timings[name].append(measure(partial(case.step, weight), args.steps) * args.steps)
                        metrics[name] = case.metrics(weight)
            for key in metrics["current"]:
                if abs(metrics["current"][key] - metrics["skin_triton"][key]) > 1e-7:
                    raise RuntimeError(f"fitting divergence: {key}")
            result["fitting"].append({"batch": batch, "weight": weight, "groups_total_ms": timings,
                "median_total_ms": {n: statistics.median(v) for n, v in timings.items()}, "metrics": metrics})
        print(f"Finished batch {batch}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
