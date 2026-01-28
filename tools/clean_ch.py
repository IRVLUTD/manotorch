from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from pathlib import Path
import pickle
import numpy as np
import argparse


def clean_fn(fn):
    import scipy.sparse
    import scipy.sparse.csc

    # Patch deprecated namespace
    scipy.sparse.csc.csc_matrix = scipy.sparse.csc_matrix
    with open(fn, "rb") as body_file:
        body_data = pickle.load(body_file, encoding="latin1")

    output_dict = {}
    for key, data in body_data.items():
        if "chumpy" in str(type(data)):
            output_dict[key] = np.asarray(data)
        else:
            output_dict[key] = data

    new_file_path = fn.parent / (fn.stem + "_new.pkl")

    with new_file_path.open("wb") as f:
        pickle.dump(output_dict, f)


def test_load_fn(fn):
    new_file_path = fn.parent / (fn.stem + "_new.pkl")
    with new_file_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    print(f"Loaded {new_file_path} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean MANO model files to remove chumpy objects."
    )
    parser.add_argument(
        "--mano_assets_folder",
        type=str,
        default="assets/mano/models",
        help="Path to the folder containing MANO model files.",
    )
    args = parser.parse_args()

    mano_files = [f"{args.mano_assets_folder}/MANO_{side}.pkl" for side in ["RIGHT", "LEFT"]]
    print(f"Found MANO files: {mano_files}")

    mano_files = [Path(f) for f in mano_files]
    for mano_file in mano_files:
        clean_fn(mano_file)
        test_load_fn(mano_file)
    print("All done.")
