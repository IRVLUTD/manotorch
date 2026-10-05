"""Load MANO model files with numpy only.

Two formats are supported:
- `MANO_{LEFT,RIGHT}.npz`: every entry stored as a dense array, loaded without pickle.
  Create it from the official pickle with `tools/mano_pkl_to_npz.py`.
- `MANO_{LEFT,RIGHT}.pkl`: the official pickles (or their chumpy-free `*_new.pkl` variant).

The official pickles reference chumpy (`shapedirs`) and scipy (`J_regressor`) objects. They are read with a
restricted unpickler that never imports those packages: their pickled state is captured by placeholder classes
and converted to dense numpy arrays, and any class a MANO pickle does not need is refused, so loading a file
cannot execute code.
"""

import importlib
import os
import pickle

import numpy as np


class _PickledState:
    """Placeholder for a chumpy or scipy object that keeps its pickled state."""

    _module = ""

    def __setstate__(self, state):
        self.__dict__.update(state)


def _numpy_reconstruct():
    # `numpy.core` is a deprecated alias of `numpy._core` since numpy 2
    module = "numpy._core.multiarray" if hasattr(np, "_core") else "numpy.core.multiarray"
    return importlib.import_module(module)._reconstruct


class _MANOUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in (("chumpy.ch", "Ch"), ("chumpy.reordering", "Select")):
            return type(name, (_PickledState,), {"_module": module})
        if module.startswith("scipy.sparse") and name == "csc_matrix":
            return type(name, (_PickledState,), {"_module": module})
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return _numpy_reconstruct()
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("builtins", "__builtin__") and name == "set":
            return set
        if module == "_codecs" and name == "encode":
            return _legacy_bytes
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: not part of a MANO pickle")


def _legacy_bytes(value, encoding):
    """Protocol-2 numpy pickles encode byte arrays this way; do not invoke arbitrary registered codecs."""
    if not isinstance(value, str) or encoding not in ("latin1", "latin-1"):
        raise pickle.UnpicklingError("only latin1 byte encoding is allowed in MANO pickles")
    return value.encode("latin1")


def _to_numpy(obj):
    if isinstance(obj, dict):
        return {key: _to_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_numpy(value) for value in obj]
    if not isinstance(obj, _PickledState):
        return obj
    state = obj.__dict__
    if "x" in state:  # chumpy.ch.Ch
        return np.asarray(state["x"])
    if "a" in state and "idxs" in state:  # chumpy.reordering.Select
        return _to_numpy(state["a"]).ravel()[np.asarray(state["idxs"]).ravel()].reshape(state["preferred_shape"])
    if type(obj).__name__ == "csc_matrix":  # scipy.sparse.csc_matrix
        shape = state["_shape"] if "_shape" in state else state["shape"]
        # Fortran order, as scipy's toarray() gives for CSC, so float32 matmuls accumulate exactly as before
        dense = np.zeros(shape, dtype=state["data"].dtype, order="F")
        cols = np.repeat(np.arange(shape[1]), np.diff(state["indptr"]))
        dense[state["indices"], cols] = state["data"]
        return dense
    raise TypeError(f"Unsupported object in MANO pickle: {obj._module}.{type(obj).__name__}")


def load_mano_pickle(path: str):
    """Load a MANO pickle (`MANO_{LEFT,RIGHT}.pkl` or its chumpy-free `*_new.pkl` variant).

    Python 2 pickles of numpy data that come with MANO, such as the training poses, load too.

    Returns:
        the pickled object, with every chumpy and scipy array converted to a dense `np.ndarray`.
    """
    with open(path, "rb") as f:
        data = _MANOUnpickler(f, encoding="latin1").load()
    return _to_numpy(data)


def load_mano_model(path: str) -> dict:
    """Load a MANO model file, either `.npz` or `.pkl`.

    Returns:
        dict: the model entries as dense `np.ndarray`s; string entries (`bs_style`, `bs_type`) as `str`.
    """
    path = os.fspath(path)
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return {key: arr.item() if arr.dtype.kind == "U" else arr for key, arr in arrays.items()}
    data = load_mano_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path} holds a {type(data).__name__}, not a MANO model dict")
    return data


def find_mano_model(mano_assets_root: str, side: str) -> str:
    """Find a model under root/models or a flat root, preferring NPZ in each folder.

    The canonical folder takes precedence; *_new.pkl and *_np.pkl support legacy converted model collections.
    """
    if side not in ("left", "right", "LEFT", "RIGHT"):
        raise ValueError(f"Unknown hand side {side!r}; expected left or right")
    for folder in (os.path.join(mano_assets_root, "models"), os.fspath(mano_assets_root)):
        stem = os.path.join(folder, f"MANO_{side.upper()}")
        for ext in (".npz", ".pkl", "_new.pkl", "_np.pkl"):
            if os.path.isfile(stem + ext):
                return stem + ext
    raise FileNotFoundError(f"Can not find MANO assets under {mano_assets_root}, please follow steps in README.md")
