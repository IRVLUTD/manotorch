#!/usr/bin/env python3
"""Convert a released MANO model pickle (``MANO_{LEFT,RIGHT}.pkl``) to an ``.npz`` that loads without pickle.

    python tools/mano_pkl_to_npz.py --input_file weights/mano/MANO_RIGHT.pkl
    python tools/mano_pkl_to_npz.py --input_file weights/mano/MANO_LEFT.pkl --output_file out/MANO_LEFT.npz

Without ``--output_file`` the result is written next to the pickle as ``<pickle name>.npz``.

**No chumpy needed.** The released pickles store ``shapedirs`` as a chumpy expression: a
``chumpy.reordering.Select`` over a ``chumpy.ch.Ch``. They are read with a restricted unpickler:
- ``Ch`` evaluates to its array ``x``;
- ``Select`` to ``a.ravel()[idxs].reshape(preferred_shape)``, chumpy's own ``compute_r``;
- numpy, scipy sparse matrices and ``set`` load as usual;
- every other class is refused, so the pickle cannot execute code.

Every key is kept. A scipy sparse matrix (``J_regressor``) is stored dense, in the layout scipy's
``toarray()`` gives it (Fortran order for CSC), and a string as a numpy unicode scalar. The file loads
with ``np.load(path, allow_pickle=False)``: no pickle, no chumpy, no scipy. The values are the
pickle's own, and the layout keeps float32 matmuls on ``J_regressor`` bit-identical to the pickle path.
Checked 2026-10-01: for both sides, every key equals the ``MANO_{SIDE}_np.pkl`` that chumpy itself had
evaluated (values and dtypes).

The ``.npz`` is uncompressed: it loads ~14x faster than an unpickled MANO dict (2.2 vs 31 ms per side,
warm cache), while compression would cost ~18 ms of zlib per load to save about 10 % of the size.

The result is written to a temporary file, read back and compared with the unpickled arrays key by key
(values and dtypes); it replaces ``--output_file`` only if every key matches.
"""

from __future__ import annotations

import argparse
import builtins
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


class _ChumpyNode:
    """Stand-in for a pickled chumpy object: keeps its state, evaluated by :func:`_evaluate`."""

    kind = ""

    def __setstate__(self, state):
        self.state = state


def _chumpy_class(kind: str) -> type:
    return type(kind, (_ChumpyNode,), {"kind": kind})


#: Classes a MANO pickle may name, by (module, name) as written in the file. Anything else is refused.
_SPARSE = {"csc_matrix", "csr_matrix", "coo_matrix"}
_CHUMPY = {("chumpy.ch", "Ch"): "Ch", ("chumpy.reordering", "Select"): "Select"}


class _MANOUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _CHUMPY:
            return _chumpy_class(_CHUMPY[module, name])
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
            return super().find_class("numpy._core.multiarray", name)  # numpy.core: deprecated alias
        if module == "numpy" and name in ("ndarray", "dtype"):
            return super().find_class(module, name)
        if module.startswith("scipy.sparse") and name in _SPARSE:
            # The released pickles name scipy.sparse.csc, deprecated and removed in SciPy 2.0.
            return getattr(sparse, name)
        if module in ("builtins", "__builtin__") and name == "set":
            return builtins.set
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: not part of a MANO model pickle")


def _evaluate(value: Any) -> Any:
    """A chumpy stand-in as the array chumpy would compute; anything else unchanged."""
    if not isinstance(value, _ChumpyNode):
        return value
    st = value.state
    if value.kind == "Ch":
        return np.asarray(st["x"])
    if value.kind == "Select":  # chumpy.reordering.Select.compute_r
        out = np.asarray(_evaluate(st["a"])).ravel()[st["idxs"]].copy()
        return out.reshape(st["preferred_shape"]) if "preferred_shape" in st else out
    raise pickle.UnpicklingError(f"unsupported chumpy class {value.kind}")


def to_array(key: str, value: Any) -> np.ndarray:
    """One MANO value as a numeric or unicode array that needs no pickle to load."""
    value = _evaluate(value)
    if sparse.issparse(value):
        # scipy's own dense form, layout included (Fortran order for CSC): float32 matmuls on it then
        # accumulate exactly as they did on the pickle's matrix.
        return value.toarray()
    if isinstance(value, bytes):
        value = value.decode("latin1")
    arr = np.asarray(value)
    if arr.dtype.kind not in "biufcU":
        raise TypeError(f"{key!r}: {type(value).__name__} of dtype {arr.dtype} cannot be stored without pickle")
    return arr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_file", "--input-file", type=Path, required=True, help="MANO_{LEFT,RIGHT}.pkl")
    ap.add_argument(
        "--output_file",
        "--output-file",
        type=Path,
        default=None,
        help="output .npz; default: <input_file without .pkl>.npz beside it",
    )
    args = ap.parse_args(argv)

    src: Path = args.input_file
    out: Path = args.output_file if args.output_file is not None else src.with_suffix(".npz")
    if out.suffix != ".npz":
        print(f"Error: --output_file must end in .npz, got {out}", file=sys.stderr)
        return 1
    if not src.is_file():
        print(f"Error: no such pickle: {src}", file=sys.stderr)
        return 1

    try:
        with open(src, "rb") as f:
            data = _MANOUnpickler(f, encoding="latin1").load()
        if not isinstance(data, dict):
            raise TypeError(f"{src} holds a {type(data).__name__}, not a MANO dict")
        arrays = {str(k): to_array(str(k), v) for k, v in data.items()}
    except (pickle.UnpicklingError, TypeError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=out.parent, prefix=f".{out.stem}.")
    os.close(fd)
    try:
        np.savez(tmp, allow_pickle=False, **arrays)
        with np.load(tmp, allow_pickle=False) as z:
            if sorted(z.files) != sorted(arrays):
                raise ValueError(f"keys differ after the round trip: {sorted(set(z.files) ^ set(arrays))}")
            for k, a in arrays.items():
                b = z[k]
                if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                    raise ValueError(f"{k!r} differs after the round trip")
        os.replace(tmp, out)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(out, 0o666 & ~umask)  # mkstemp's 0600 would hide the weights from the group
    except Exception as e:
        os.unlink(tmp)
        print(f"Error: {e}; {out} not written", file=sys.stderr)
        return 1

    print(f"{src} -> {out}  ({src.stat().st_size / 2**20:.2f} -> {out.stat().st_size / 2**20:.2f} MiB)")
    for k in sorted(arrays):
        a, v = arrays[k], data[k]
        note = ""
        if sparse.issparse(v):
            note = "  (scipy sparse, stored dense)"
        elif isinstance(v, _ChumpyNode):
            note = f"  (chumpy {v.kind}, evaluated)"
        print(f"  {k:18s} {str(a.dtype):9s} {str(a.shape):16s}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
