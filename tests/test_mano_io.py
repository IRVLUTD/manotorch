import os
import pickle

import numpy as np
import pytest
from conftest import MANO_ASSETS_ROOT, requires_mano

from manotorch.utils.mano_io import find_mano_model, load_mano_model, load_mano_pickle

EXPECTED_SHAPES = {
    "v_template": (778, 3),
    "f": (1538, 3),
    "shapedirs": (778, 3, 10),
    "posedirs": (778, 3, 135),
    "J_regressor": (16, 778),
    "weights": (778, 16),
    "kintree_table": (2, 16),
    "hands_components": (45, 45),
    "hands_mean": (45,),
}


class _Malicious:
    def __reduce__(self):
        return (os.system, ("echo should-not-run",))


def test_refuses_unexpected_classes(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"x": _Malicious()}))
    with pytest.raises(pickle.UnpicklingError):
        load_mano_pickle(str(path))


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_mano_model(str(tmp_path), "right")


@requires_mano
@pytest.mark.parametrize("side", ["RIGHT", "LEFT"])
def test_model_entries(side):
    model = load_mano_model(find_mano_model(MANO_ASSETS_ROOT, side))
    for key, shape in EXPECTED_SHAPES.items():
        assert isinstance(model[key], np.ndarray) and model[key].shape == shape, key
    assert model["bs_type"] == "lrotmin" and model["bs_style"] == "lbs"


@requires_mano
@pytest.mark.parametrize("side", ["RIGHT", "LEFT"])
def test_npz_matches_pickle(side):
    stem = os.path.join(MANO_ASSETS_ROOT, "models", f"MANO_{side}")
    if not (os.path.isfile(stem + ".npz") and os.path.isfile(stem + ".pkl")):
        pytest.skip("needs both the .npz and the .pkl model")
    npz, pkl = load_mano_model(stem + ".npz"), load_mano_model(stem + ".pkl")
    assert sorted(npz) == sorted(pkl)
    for key, value in pkl.items():
        if isinstance(value, str):
            assert npz[key] == value
        else:
            assert npz[key].dtype == value.dtype and np.array_equal(npz[key], value), key
            assert npz[key].flags.f_contiguous == value.flags.f_contiguous, key
