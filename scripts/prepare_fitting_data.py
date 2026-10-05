"""Select ARCTIC/HO-Cap fitting samples from the UHAS exports without changing source data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/home/jikaiwang/Projects/UTD/UHAS/UHAS_HumanData/data/hand_poses"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("arctic", "hocap"):
        source = sorted((args.root / dataset).glob("*.npz"))[0]
        rng = np.random.default_rng(args.seed)
        result, selection = {}, []
        with np.load(source, allow_pickle=False) as z:
            metadata = json.loads(z["meta"].item())
            if metadata["units"] != "metres" or metadata["rotation"] != "axis-angle":
                raise ValueError("Expected metre units and axis-angle rotations")
            for index, side in enumerate(metadata["hands"]):
                valid = np.flatnonzero(z["valid"][:, index])
                ids = rng.permutation(valid)[:args.samples]
                if len(ids) == 0:
                    continue
                pose = np.concatenate([z["global_orient"][ids, index], z["hand_pose"][ids, index]], axis=1)
                betas, transl = z["betas"][ids, index], z["transl"][ids, index]
                values = {
                    "pose": pose, "betas": betas, "transl": transl, "joints": z["joints"][ids, index],
                    "initial_pose": pose + rng.normal(0, 0.05, pose.shape).astype(np.float32),
                    "initial_betas": betas + rng.normal(0, 0.1, betas.shape).astype(np.float32),
                    "initial_transl": transl + rng.normal(0, 0.005, transl.shape).astype(np.float32),
                }
                if not all(np.isfinite(v).all() for v in values.values()):
                    raise ValueError("Non-finite selected sample")
                result.update({side + "_" + key: value for key, value in values.items()})
                selection.append({"side": side, "array_indices": ids.tolist(),
                                  "frame_indices": z["frame_index"][ids].tolist()})
        output = args.output_dir / f"{dataset}_targets.npz"
        np.savez(output, **result)
        manifest = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "seed": args.seed, "selection": selection, "metadata": metadata,
                    "fitting_joints": "16 MANO joints; exclude all five tips", "targets": "unchanged exported joints"}
        output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(output, [(x["side"], len(x["frame_indices"])) for x in selection])


if __name__ == "__main__":
    main()
