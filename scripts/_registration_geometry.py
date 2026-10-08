"""NumPy-only geometry and hashing for portable cached MANO QC rendering."""

import hashlib
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numpy_rotations(pose):
    """Rodrigues rotations with analytic limits at zero, in float64."""
    a = np.asarray(pose, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(a, axis=-1)
    skew = np.zeros((len(a), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -a[:, 2], a[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = a[:, 2], -a[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -a[:, 1], a[:, 0]
    return np.eye(3) + np.sinc(theta / np.pi)[:, None, None] * skew + (
        .5 * np.sinc(theta / (2 * np.pi)) ** 2)[:, None, None] * (skew @ skew)
