"""Compare repeated MANO mesh subdivision with the archived uncached implementation.

Cold preparation is timed separately. Warm timings include midpoint interpolation and
a fresh faces output, but not MANO forward. No model files or raw results are distributed.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from manotorch.manolayer import ManoLayer
from manotorch.upsamplelayer import UpSampleLayer


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device, repeats):
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("data/benchmarks/baseline_source"))
    parser.add_argument("--mano-assets-root", default="assets/mano")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/upsample_cache.json"))
    args = parser.parse_args()
    if min(*args.batches, args.groups, args.repeats) < 1:
        parser.error("batches, groups and repeats must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    path = args.baseline / "manotorch/upsamplelayer.py"
    spec = importlib.util.spec_from_file_location("uncached_upsample", path)
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    mano = ManoLayer(mano_assets_root=args.mano_assets_root)
    vertices = mano(torch.zeros(1, 48)).verts.detach()
    faces = mano.th_faces
    result = {"torch": torch.__version__, "threads": 4, "faces": faces.shape[0],
              "vertices": vertices.shape[1], "topology": "shared expanded MANO right open-wrist faces",
              "baseline": "c936b59", "groups": args.groups, "repeats": args.repeats,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (path, Path("manotorch/upsamplelayer.py"), Path(__file__))}, "cases": []}
    for dev in args.devices:
        device = torch.device(dev)
        for batch in args.batches:
            v = vertices.to(device).expand(batch, -1, -1).contiguous()
            fs = faces.to(device).unsqueeze(0).expand(batch, -1, -1)
            old, cached = archived.UpSampleLayer(), UpSampleLayer()
            cold = measure(partial(cached.prepare, fs, v.shape[1]), device, 1)
            methods = {"uncached": partial(old, v, fs), "cached_faces": partial(cached, v, fs),
                       "prepared": partial(cached, v)}
            expected, expected_faces = methods["uncached"]()
            for method in methods.values():
                out, new_faces = method()
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                assert torch.equal(new_faces, expected_faces)
            timings = {name: [] for name in methods}
            names = list(methods)
            for group in range(args.groups):
                for name in names[group % len(names):] + names[:group % len(names)]:
                    repeats = args.repeats if name == "uncached" else args.repeats * 10
                    timings[name].append(measure(methods[name], device, repeats))
            row = {"device": dev, "batch": batch, "prepare_ms": cold, "group_ms": timings,
                   "median_ms": {n: statistics.median(v) for n, v in timings.items()}}
            result["cases"].append(row)
            print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
