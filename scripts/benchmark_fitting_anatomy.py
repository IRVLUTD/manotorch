"""Interleaved eager MANO fitting with and without the anatomy prior; see doc/benchmark.md.

Each implementation uses its own ManoLayer, AxisLayerFK and AnatomyConstraintLossEE. Reference basis buffers
and limit configurations are aligned at construction. Frozen targets and raw results stay under ignored data.
"""

import argparse
import hashlib
import importlib
import json
import math
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from benchmark_layers import MANO16, legacy_loader, load_package, revision, synchronize

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def native_imports(alias):
    """Resolve legacy absolute imports to their own package, then restore the current package."""
    saved = {k: v for k, v in sys.modules.items() if k == "manotorch" or k.startswith("manotorch.")}
    try:
        for suffix in ("", ".utils", ".utils.geometry", ".manolayer"):
            sys.modules["manotorch" + suffix] = importlib.import_module(alias + suffix)
        yield
    finally:
        for name in list(sys.modules):
            if name == "manotorch" or name.startswith("manotorch."):
                del sys.modules[name]
        sys.modules.update(saved)


def source_hash(folder):
    return hashlib.sha256(b"".join(
        p.relative_to(folder).as_posix().encode() + p.read_bytes() for p in sorted(folder.rglob("*.py"))
    )).hexdigest()


def load_classes(args):
    legacy_loader()
    classes = {"current": (ManoLayer, AxisLayerFK, AnatomyConstraintLossEE)}
    for name, alias, folder in (("claude_c936b59", "fit_claude", args.baseline),
                                ("upstream", "fit_upstream", args.upstream)):
        load_package(alias, folder / "manotorch")
        with native_imports(alias):
            classes[name] = (importlib.import_module(alias + ".manolayer").ManoLayer,
                             importlib.import_module(alias + ".axislayer").AxisLayerFK,
                             importlib.import_module(alias + ".anatomy_loss").AnatomyConstraintLossEE)
    return classes


class FittingCase:
    def __init__(self, classes, args, arrays, batch, device, reference):
        mano_cls, axis_cls, loss_cls = classes
        self.mano = mano_cls(side="right", use_pca=False, flat_hand_mean=True,
                             mano_assets_root=str(args.mano_assets_root)).to(device)
        self.axis = axis_cls(side="right", mano_assets_root=str(args.mano_assets_root))
        self.axis.load_state_dict(reference.state_dict())
        if hasattr(self.axis, "_Ra_par_tmplchd"):
            # Archived code has no post-load hook; align its derived cache at construction as well.
            basis = self.axis.TMPL_R_p_a
            parent = torch.tensor(self.axis.transf_parent_mapping)
            self.axis._Ra_par_tmplchd = basis.index_select(1, parent).transpose(2, 3) @ basis
        self.axis = self.axis.to(device)
        self.prior = loss_cls(reduction="mean").to(device)
        self.prior.setup()
        self.ids = torch.tensor(MANO16, device=device)

        def get(key):
            data = arrays["right_" + key]
            return torch.as_tensor(data[np.arange(batch) % len(data)], device=device).clone()

        self.target = get("joints").index_select(1, self.ids)
        self.initial = [get(k) for k in ("initial_pose", "initial_betas", "initial_transl")]
        self.params = [x.clone().requires_grad_() for x in self.initial]
        self.optimizer = torch.optim.Adam([
            {"params": [self.params[0]], "lr": 0.01}, {"params": [self.params[1]], "lr": 0.02},
            {"params": [self.params[2]], "lr": 0.001},
        ], foreach=False)

    def objective(self, weight):
        p, b, t = self.params
        output = self.mano(p, b)
        residual = output.joints.index_select(1, self.ids) + t[:, None] - self.target
        data_loss = residual.square().mean()
        prior = self.prior(self.axis(output.transforms_abs)[2]) if weight else data_loss.new_zeros(())
        return data_loss + weight * prior, residual, prior

    def step(self, weight):
        self.optimizer.zero_grad(set_to_none=True)
        self.objective(weight)[0].backward()
        self.optimizer.step()

    def reset(self):
        with torch.no_grad():
            for p, initial in zip(self.params, self.initial, strict=True):
                p.copy_(initial)
            # Retain Adam's allocated state after warmup, but start every measured fit from step zero.
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        self.optimizer.zero_grad(set_to_none=True)

    def metrics(self, weight):
        with torch.no_grad():
            objective, residual, prior = self.objective(weight)
            values = torch.stack((residual.square().sum(-1).mean().sqrt() * 1000, prior, objective))
            if not torch.isfinite(values).all():
                raise RuntimeError("Non-finite fitting result")
            rmse, anatomy, total = values.cpu().tolist()
            return {"rmse_mm": rmse, "anatomy_rad": anatomy, "objective": total}


def validate(cases, weight):
    reference = None
    errors = {}
    for name, case in cases.items():
        objective, residual, prior = case.objective(weight)
        gradient = torch.autograd.grad(objective, case.params)
        current = (objective.detach(), residual.detach(), prior.detach(), tuple(g.detach() for g in gradient))
        if reference is None:
            reference = current
        obj, points, penalty, grads = reference
        torch.testing.assert_close(current[0], obj, atol=2e-8, rtol=1e-4)
        torch.testing.assert_close(current[1], points, atol=2e-6, rtol=1e-4)
        torch.testing.assert_close(current[2], penalty, atol=2e-5, rtol=1e-4)
        for g, expected in zip(gradient, grads, strict=True):
            torch.testing.assert_close(g, expected, atol=2e-7, rtol=5e-4)
        errors[name] = {"point_max_mm": (current[1]-points).abs().max().item()*1000,
                        "anatomy_abs_rad": (current[2]-penalty).abs().item(),
                        "gradient_max_abs": max((g-e).abs().max().item()
                                                for g, e in zip(gradient, grads, strict=True))}
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("data/benchmarks/baseline_source"))
    parser.add_argument("--upstream", type=Path, default=Path("data/benchmarks/manotorch_upstream"))
    parser.add_argument("--mano-assets-root", type=Path, default=Path("assets/mano"))
    parser.add_argument("--targets", type=Path, default=Path("data/benchmarks/mano_targets.npz"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 128, 1024])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--anatomy-weight", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/fitting_anatomy.json"))
    args = parser.parse_args()
    if min(*args.batches, args.steps, args.warmup, args.groups) <= 0:
        parser.error("Batches, steps, warmup and groups must be positive")
    if not math.isfinite(args.anatomy_weight) or args.anatomy_weight <= 0:
        parser.error("anatomy-weight must be positive and finite")
    if not args.targets.exists():
        parser.error("Prepare frozen MANO targets with the archived benchmark_fitting.py run first")
    torch.set_num_threads(8)
    torch.manual_seed(20261005)
    device = torch.device(args.device)
    classes = load_classes(args)
    with np.load(args.targets, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    reference = AxisLayerFK(side="right", mano_assets_root=str(args.mano_assets_root))
    source = Path(__file__).resolve().parents[1]
    result = {"environment": {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "threads": 8,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "upstream_commit": revision(args.upstream), "baseline_commit": "c936b59",
        "implementation_sha256": {name: source_hash(folder / "manotorch") for name, folder in
                                  (("current", source), ("claude_c936b59", args.baseline), ("upstream", args.upstream))},
        "benchmark_sha256": digest(Path(__file__)), "adapter_sha256": digest(Path(__file__).with_name("benchmark_layers.py")),
        "targets_sha256": digest(args.targets), "steps": args.steps, "groups": args.groups, "warmup": args.warmup,
        "objective": "mean squared 16 common joints in metres + anatomy_weight * mean anatomy penalty in radians",
        "anatomy_weight": args.anatomy_weight, "mean": "flat", "pose": "full 48-axis-angle, float32, right hand",
        "optimizer": "Adam foreach=False; pose lr=0.01, shape lr=0.02, translation lr=0.001",
        "basis": "current right-hand reference buffers copied to each native AxisLayerFK at construction",
        "basis_sha256": hashlib.sha256(reference.TMPL_R_p_a.numpy().tobytes()).hexdigest(),
        "timing": "eager only; synchronized wall time of full fitting loop, no per-step metric collection; "
                  "model/data loading, validation, warmup and Adam state reset excluded; order rotates per group",
    }, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        cases = {name: FittingCase(clss, args, arrays, batch, device, reference) for name, clss in classes.items()}
        for weight in (0.0, args.anatomy_weight):
            errors = validate(cases, weight)
            for case in cases.values():
                for _ in range(args.warmup):
                    case.step(weight)
                case.reset()
                # Initialize loss device/dtype caches before entering the timed block.
                case.metrics(weight)
            samples = {name: [] for name in cases}
            final = {name: [] for name in cases}
            initial = {name: case.metrics(weight) for name, case in cases.items()}
            names = list(cases)
            for group in range(args.groups):
                shift = group % len(names)
                for name in names[shift:] + names[:shift]:
                    case = cases[name]
                    case.reset()
                    synchronize(device)
                    start = time.perf_counter()
                    for _ in range(args.steps):
                        case.step(weight)
                    synchronize(device)
                    elapsed = (time.perf_counter() - start) * 1000
                    samples[name].append(elapsed)
                    final[name].append(case.metrics(weight))
                print(json.dumps({"batch": batch, "weight": weight, "group": group,
                                  "fit_ms": {name: values[-1] for name, values in samples.items()}}), flush=True)
            for case in cases.values():
                case.reset()
            row = {"batch": batch, "anatomy_weight": weight, "steps": args.steps, "validation": errors,
                   "samples_fit_ms": samples, "median_fit_ms": {k: statistics.median(v) for k, v in samples.items()},
                   "median_step_ms": {k: statistics.median(v)/args.steps for k, v in samples.items()},
                   "initial": initial, "final": final}
            result["cases"].append(row)
            args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
