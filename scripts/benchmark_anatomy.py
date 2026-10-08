"""Compare eager anatomy-loss forward+backward with the archived baseline, in rotating order."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from manotorch.anatomy_loss import AnatomyConstraintLossEE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('data/benchmarks/baseline_source/manotorch/anatomy_loss.py'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=Path('data/benchmarks/anatomy.json'))
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 128, 1024])
    parser.add_argument('--groups', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    if min(*args.batches, args.groups, args.repeats) < 1:
        parser.error('batches, groups and repeats must be positive')
    spec = importlib.util.spec_from_file_location('baseline_anatomy', args.baseline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows = []
    for batch in args.batches:
        angles = (torch.randn(batch, 16, 3, device=device) * .4).requires_grad_()
        methods = {}
        reference = None
        for name, cls in (('baseline', module.AnatomyConstraintLossEE), ('current', AnatomyConstraintLossEE)):
            loss = cls().to(device)
            loss.setup()
            value = loss(angles)
            gradient = torch.autograd.grad(value, angles)[0]
            if reference is not None:
                torch.testing.assert_close(value, reference[0], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(gradient, reference[1], atol=1e-6, rtol=1e-5)
            reference = (value.detach(), gradient.detach())

            def step(angles=angles, loss=loss):
                angles.grad = None
                loss(angles).backward()

            methods[name] = step
            for _ in range(10):
                step()
        samples = {name: [] for name in methods}
        names = list(methods)
        for group in range(args.groups):
            for name in names[group % len(names):] + names[:group % len(names)]:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                for _ in range(args.repeats):
                    methods[name]()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000 / args.repeats)
        for name, values in samples.items():
            row = {'implementation': name, 'batch': batch, 'median_forward_backward_ms': statistics.median(values),
                   'samples_ms': values, 'device': str(device), 'seed': args.seed, 'torch': torch.__version__,
                   'ordering': 'rotating interleaved', 'repeats': args.repeats, 'groups': args.groups,
                   'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
