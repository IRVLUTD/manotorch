"""Load MANO model pickles with numpy only.

The official MANO pickles reference chumpy (`shapedirs`) and scipy (`J_regressor`) objects.
Instead of importing those packages, their pickled state is captured by placeholder classes
and converted to dense numpy arrays.
"""

import pickle

import numpy as np


class _PickledState:
    """Placeholder for a chumpy or scipy object that keeps its pickled state."""

    _module = ""

    def __setstate__(self, state):
        self.__dict__.update(state)


class _MANOUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("chumpy", "scipy"):
            return type(name, (_PickledState,), {"_module": module})
        return super().find_class(module, name)


def _to_numpy(obj):
    if not isinstance(obj, _PickledState):
        return obj
    state = obj.__dict__
    if "x" in state:  # chumpy.ch.Ch
        return np.asarray(state["x"])
    if "a" in state and "idxs" in state:  # chumpy.reordering.Select
        return _to_numpy(state["a"]).ravel()[np.asarray(state["idxs"]).ravel()].reshape(state["preferred_shape"])
    if type(obj).__name__ == "csc_matrix":  # scipy.sparse.csc_matrix
        shape = state["_shape"] if "_shape" in state else state["shape"]
        dense = np.zeros(shape, dtype=state["data"].dtype)
        cols = np.repeat(np.arange(shape[1]), np.diff(state["indptr"]))
        dense[state["indices"], cols] = state["data"]
        return dense
    raise TypeError(f"Unsupported object in MANO pickle: {obj._module}.{type(obj).__name__}")


def load_mano_pickle(path: str) -> dict:
    """Load a MANO model file (`MANO_{LEFT,RIGHT}.pkl` or its chumpy-free `*_new.pkl` variant).

    Returns:
        dict: the model entries, with every array as a dense `np.ndarray`.
    """
    with open(path, "rb") as f:
        data = _MANOUnpickler(f, encoding="latin1").load()
    return {key: _to_numpy(value) for key, value in data.items()}
