"""Compare eager anatomy-loss forward+backward with the archived baseline."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from manotorch.anatomy_loss import AnatomyConstraintLossEE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("data/benchmarks/baseline_source/manotorch/anatomy_loss.py"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/anatomy.json"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("baseline_anatomy", args.baseline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(8)
    rows = []
    for batch in (1, 128, 1024):
        angles = (torch.randn(batch, 16, 3, device=args.device) * 0.4).requires_grad_()
        for name, cls in (("baseline", module.AnatomyConstraintLossEE), ("current", AnatomyConstraintLossEE)):
            loss = cls()
            loss.setup()

            def step(angles=angles, loss=loss):
                angles.grad = None
                loss(angles).backward()

            for _ in range(10):
                step()
            samples = []
            for _ in range(5):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(30):
                    step()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000 / 30)
            row = {"implementation": name, "batch": batch, "median_forward_backward_ms": statistics.median(samples),
                   "samples_ms": samples, "device": args.device}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
