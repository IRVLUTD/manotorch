"""Reproducible MANO fitting and runtime benchmark; see doc/benchmark.md.

Targets are frozen on the first run. Use --implementation-root to benchmark an archived revision.
Licensed datasets and generated targets/results stay under the git-ignored data directory.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mark_step(enabled=True):
    if not enabled:
        return
    fn = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()


def measure(fn, device, repeats, compiled=False):
    """Synchronized wall time includes launch overhead, excludes data loading and compilation."""
    for _ in range(5):
        mark_step(compiled and device.type == "cuda")
        fn()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        initial = torch.cuda.memory_allocated(device)
    else:
        initial = 0
    start = time.perf_counter()
    for _ in range(repeats):
        mark_step(compiled and device.type == "cuda")
        fn()
    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000 / repeats
    memory = (torch.cuda.max_memory_allocated(device) - initial) / 2**20 if device.type == "cuda" else None
    return {"ms": elapsed, "peak_extra_mib": memory}


def prepare_targets(args, layer_class):
    cache = args.targets
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if args.source != "mano":
        raise ValueError("Real-data targets must be prepared with scripts/prepare_fitting_data.py first")
    rng = np.random.default_rng(args.seed)
    arrays, manifest = {}, {"seed": args.seed, "source": "MANO training articulation", "files": []}
    for side, suffix in [("right", "R"), ("left", "L")]:
        path = args.dataset / "mano_poses_v1_0" / f"handsOnly_REGISTRATIONS_r_lm___POSES___{suffix}.npy"
        poses = np.load(path, allow_pickle=False)
        ids = rng.permutation(len(poses))[: args.samples]
        p = np.concatenate([rng.normal(0, 0.3, (len(ids), 3)), poses[ids]], axis=1).astype(np.float32)
        b = rng.normal(0, 0.5, (len(ids), 10)).astype(np.float32)
        t = rng.normal(0, 0.1, (len(ids), 3)).astype(np.float32)
        layer = layer_class(side=side, mano_assets_root=str(args.mano_assets_root))
        with torch.no_grad():
            out = layer(torch.from_numpy(p), torch.from_numpy(b), torch.from_numpy(t))
        arrays.update({f"{side}_{k}": v for k, v in {
            "pose": p, "betas": b, "transl": t, "joints": out.joints.numpy(), "verts": out.verts.numpy(),
            "initial_pose": p + rng.normal(0, 0.05, p.shape).astype(np.float32),
            "initial_betas": b + rng.normal(0, 0.1, b.shape).astype(np.float32),
            "initial_transl": t + rng.normal(0, 0.005, t.shape).astype(np.float32),
        }.items()})
        manifest["files"].append({"side": side, "path": str(path), "sha256": digest(path), "indices": ids.tolist()})
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    cache.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return arrays


def run_case(args, layer_class, arrays, device, side, batch, task, mode):
    def get(key):
        a = arrays[f"{side}_{key}"]
        indices = np.arange(batch) % len(a)
        return torch.as_tensor(a[indices], device=device).clone()

    layer = layer_class(side=side, mano_assets_root=str(args.mano_assets_root)).to(device)
    joints_only = task == "joints_only"
    field = "verts" if task == "vertices" else "joints"
    target = get(field)
    # Real observations use the 16 MANO joints: datasets can sample different fingertip vertices.
    ids = torch.tensor([0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], device=device)

    def predict(p, b, t):
        result = getattr(layer(p, b, t, joints_only=joints_only), field)
        return result.index_select(1, ids) if args.source != "mano" else result

    if args.source != "mano":
        target = target.index_select(1, ids)
    p, b, t = [get(k).requires_grad_() for k in ("initial_pose", "initial_betas", "initial_transl")]
    compile_seconds = 0.0
    if mode == "compile":
        torch._dynamo.reset()
        predict = torch.compile(predict, mode="reduce-overhead", fullgraph=True)
        synchronize(device)
        start = time.perf_counter()
        predict(p, b, t).square().mean().backward()
        synchronize(device)
        compile_seconds = time.perf_counter() - start
    for x in (p, b, t):
        x.grad = None

    def backward():
        for x in (p, b, t):
            x.grad = None
        (predict(p, b, t) - target).square().mean().backward()

    # Keep grad mode enabled for forward: fitting retains an autograd graph.
    forward = measure(lambda: predict(p, b, t), device, args.repeats, compiled=mode == "compile")
    forward_backward = measure(backward, device, args.repeats, compiled=mode == "compile")
    optimizer = torch.optim.Adam([
        {"params": [p], "lr": 0.01}, {"params": [b], "lr": 0.02}, {"params": [t], "lr": 0.001},
    ], foreach=False)
    initial_error = ((predict(p, b, t).detach() - target).square().sum(-1).mean().sqrt() * 1000).item()
    # Record the metric before optimizer updates: CUDA graph outputs can share optimizer parameter storage.
    errors = []
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        mark_step(mode == "compile" and device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        residual = predict(p, b, t) - target
        errors.append(residual.detach().square().sum(-1).mean().sqrt())
        residual.square().mean().backward()
        optimizer.step()
    synchronize(device)
    fit_ms = (time.perf_counter() - start) * 1000
    fit_memory = (torch.cuda.max_memory_allocated(device) - initial_memory) / 2**20 if device.type == "cuda" else None
    curve = (torch.stack(errors) * 1000).cpu().tolist()
    # Final prediction uses the same grad-mode graph as fitting, avoiding a second compile variant.
    mark_step(mode == "compile" and device.type == "cuda")
    final_error = ((predict(p, b, t).detach() - target).square().sum(-1).mean().sqrt() * 1000).item()
    curve.append(final_error)  # Index equals completed updates, including the final optimizer step.
    threshold = next((i for i, e in enumerate(curve) if e <= args.threshold_mm), None)
    return {
        "side": side, "batch": batch, "task": task, "mode": mode, "compile_seconds": compile_seconds,
        "forward": forward, "forward_backward": forward_backward, "fit_step_ms": fit_ms / args.steps,
        "fit_total_ms": fit_ms, "fit_peak_extra_mib": fit_memory,
        "initial_rmse_mm": initial_error, "final_rmse_mm": final_error, "curve_rmse_mm": curve,
        "threshold_mm": args.threshold_mm, "steps_to_threshold": threshold,
        "estimated_ms_to_threshold": threshold * fit_ms / args.steps if threshold is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=Path("data/MANO_Poses"))
    parser.add_argument("--mano-assets-root", type=Path, default=Path("assets/mano"))
    parser.add_argument("--targets", type=Path, default=Path("data/benchmarks/mano_targets.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/results.json"))
    parser.add_argument("--source", choices=["mano", "real"], default="mano")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 1024])
    parser.add_argument("--tasks", nargs="+", choices=["joints", "joints_only", "vertices"],
                        default=["joints", "joints_only", "vertices"])
    parser.add_argument("--modes", nargs="+", choices=["eager", "compile"], default=["eager", "compile"])
    parser.add_argument("--sides", nargs="+", choices=["left", "right"], default=["right", "left"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threshold-mm", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.samples, *args.batches) < 1:
        parser.error("steps, repeats, samples and batches must be positive")
    if args.source == "real" and "vertices" in args.tasks:
        parser.error("real observations only supply joints; select --tasks joints joints_only")
    sys.path.insert(0, str(args.implementation_root.resolve()))
    from manotorch.manolayer import ManoLayer

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    arrays = prepare_targets(args, ManoLayer)
    result = {"environment": {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_root": str(args.implementation_root.resolve()), "targets_sha256": digest(args.targets),
        "steps": args.steps, "repeats": args.repeats, "source": args.source,
        "schema_version": 2, "curve_convention": "index = completed updates; includes final prediction",
        "timing": "synchronized wall clock, warm steady state, fitting optimizer eager",
        "implementation_sha256": hashlib.sha256(b"".join(
            p.relative_to(args.implementation_root).as_posix().encode() + p.read_bytes()
            for p in sorted((args.implementation_root / "manotorch").rglob("*.py"))
        )).hexdigest(),
        "benchmark_sha256": digest(Path(__file__)),
    }, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for side in args.sides:
        for batch in args.batches:
            for task in args.tasks:
                for mode in args.modes:
                    case = run_case(args, ManoLayer, arrays, device, side, batch, task, mode)
                    result["cases"].append(case)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps({k: case[k] for k in (
                        "side", "batch", "task", "mode", "fit_step_ms", "final_rmse_mm",
                    )}), flush=True)


if __name__ == "__main__":
    main()
