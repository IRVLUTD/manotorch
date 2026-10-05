"""Interleaved eager MANO comparison; see doc/benchmark.md.

Upstream and manopth get a model-deserialization shim only; their numerical forward/backward source is unchanged.
Construction, loading, accuracy checks, warmup and compilation are excluded from steady-state timings.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from manotorch.manolayer import ManoLayer
from manotorch.utils.geometry import axis_angle_to_matrix
from manotorch.utils.mano_io import load_mano_model

MANO16 = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]


def load_package(alias, folder):
    spec = importlib.util.spec_from_file_location(alias, folder / "__init__.py", submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)


def legacy_loader():
    """Supply arrays at construction, avoiding chumpy installation; no changes to any timed operation."""
    def ready(path):
        data = load_mano_model(path)
        data.setdefault("betas", np.zeros(data["shapedirs"].shape[-1]))
        for key in ("betas", "shapedirs", "posedirs", "v_template", "weights"):
            data[key] = types.SimpleNamespace(r=data[key])
        matrix = data["J_regressor"]
        data["J_regressor"] = types.SimpleNamespace(toarray=lambda: matrix)
        return data

    module = types.ModuleType("mano.webuser.smpl_handpca_wrapper_HAND_only")
    module.ready_arguments = ready
    sys.modules[module.__name__] = module


def revision(folder):
    return subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("data/benchmarks/manotorch_upstream"))
    parser.add_argument("--manopth", type=Path, default=Path("data/benchmarks/manopth"))
    parser.add_argument("--smplx", type=Path, default=Path("data/benchmarks/smplx"))
    parser.add_argument("--baseline", type=Path, default=Path("data/benchmarks/baseline_source"))
    parser.add_argument("--before-geometry", type=Path, help="Optional saved geometry before eager optimization")
    parser.add_argument("--mano-assets-root", type=Path, default=Path("assets/mano"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 128, 1024])
    parser.add_argument("--sides", nargs="+", choices=["right", "left"], default=["right", "left"])
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/eager_layers.json"))
    args = parser.parse_args()
    if min(*args.batches, args.groups, args.repeats) <= 0:
        parser.error("Batches, groups and repeats must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(8)
    torch.manual_seed(20261005)
    legacy_loader()
    load_package("bench_upstream", args.upstream / "manotorch")
    load_package("manopth", args.manopth / "manopth")
    load_package("bench_smplx", args.smplx / "smplx")
    load_package("bench_baseline", args.baseline / "manotorch")
    upstream = importlib.import_module("bench_upstream.manolayer").ManoLayer
    manopth = importlib.import_module("manopth.manolayer").ManoLayer
    smplx = sys.modules["bench_smplx"]
    baseline = importlib.import_module("bench_baseline.manolayer").ManoLayer
    before = None
    if args.before_geometry:
        folder = Path(__file__).resolve().parents[1] / "manotorch"
        load_package("bench_before", folder)
        spec = importlib.util.spec_from_file_location("bench_before.utils.geometry", args.before_geometry)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        before = importlib.import_module("bench_before.manolayer").ManoLayer
    source = Path(__file__).resolve().parents[1] / "manotorch"
    result = {"environment": {
        "torch": torch.__version__, "numpy": np.__version__, "threads": 8,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "revisions": {name: revision(path) for name, path in
                      (("upstream", args.upstream), ("manopth", args.manopth), ("smplx", args.smplx))},
        "baseline": "c936b59", "groups": args.groups, "repeats": args.repeats,
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.rglob("*.py"))},
        "before_geometry_sha256": (hashlib.sha256(args.before_geometry.read_bytes()).hexdigest()
                                   if args.before_geometry else None),
        "pose": "full 48-axis-angle, flat mean, float32; MANOLayer includes AA conversion",
        "loss": "mean squared vertices (metres); pose and betas gradients",
        "timing": "warm synchronized wall-clock; each group's implementation order rotates",
    }, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for side in args.sides:
        kw = {"side": side, "mano_assets_root": str(args.mano_assets_root), "flat_hand_mean": True}
        model_file = args.mano_assets_root / "models" / f"MANO_{side.upper()}.pkl"
        struct = smplx.utils.Struct(**load_mano_model(model_file))
        for batch in args.batches:
            pose = (torch.randn(batch, 48, device=device) * 0.3).requires_grad_()
            betas = (torch.randn(batch, 10, device=device) * 0.5).requires_grad_()
            models = {"upstream": upstream(**kw).to(device), "claude_c936b59": baseline(**kw).to(device),
                      "current": ManoLayer(**kw).to(device)}
            if before:
                models["before_eager"] = before(**kw).to(device)
            models["manopth"] = manopth(side=side, mano_root=str(model_file.parent), use_pca=False,
                                       flat_hand_mean=True, center_idx=None).to(device)
            models["smplx.MANO"] = smplx.MANO(str(model_file), data_struct=struct, is_rhand=side == "right",
                                              use_pca=False, flat_hand_mean=True).to(device)
            models["smplx.MANOLayer"] = smplx.MANOLayer(str(model_file), data_struct=struct,
                                                        is_rhand=side == "right").to(device)

            def predict(name, models=models, pose=pose, betas=betas):
                if name == "manopth":
                    v, j = models[name](pose, betas)
                    return v / 1000, j[:, MANO16] / 1000
                if name == "smplx.MANO":
                    out = models[name](betas=betas, global_orient=pose[:, :3], hand_pose=pose[:, 3:])
                    return out.vertices, out.joints[:, :16]
                if name == "smplx.MANOLayer":
                    r = axis_angle_to_matrix(pose.reshape(-1, 16, 3))
                    out = models[name](betas=betas, global_orient=r[:, :1], hand_pose=r[:, 1:])
                    return out.vertices, out.joints[:, :16]
                out = models[name](pose, betas)
                return out.verts, out.joints[:, MANO16]

            vref, jref = predict("current")
            ref_grad = torch.autograd.grad(vref.square().mean(), (pose, betas))
            errors = {}
            for name in models:
                v, j = predict(name)
                grad = torch.autograd.grad(v.square().mean(), (pose, betas))
                errors[name] = {"vertex_mm": (v-vref).abs().max().item()*1000,
                                "joint_mm": (j-jref).abs().max().item()*1000,
                                "gradient_abs": max((a-b).abs().max().item()
                                                    for a, b in zip(grad, ref_grad, strict=True))}
                torch.testing.assert_close(v, vref, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(j, jref, atol=2e-6, rtol=1e-5)
                for a, b in zip(grad, ref_grad, strict=True):
                    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-4)

            def step(name, backward, pose=pose, betas=betas, predict=predict):
                pose.grad = betas.grad = None
                v, _ = predict(name)
                if backward:
                    v.square().mean().backward()

            for name in models:
                for _ in range(10):
                    step(name, True)
            for backward in (False, True):
                samples = {name: [] for name in models}
                names = list(models)
                for group in range(args.groups):
                    shift = group % len(names)
                    for name in names[shift:] + names[:shift]:
                        synchronize(device)
                        start = time.perf_counter()
                        for _ in range(args.repeats):
                            step(name, backward)
                        synchronize(device)
                        samples[name].append((time.perf_counter()-start)*1000/args.repeats)
                row = {"side": side, "batch": batch, "backward": backward, "errors": errors,
                       "samples_ms": samples, "median_ms": {name: statistics.median(v) for name, v in samples.items()}}
                result["cases"].append(row)
                args.output.write_text(json.dumps(result, indent=2)+"\n")
                print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)


if __name__ == "__main__":
    main()
