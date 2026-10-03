import os

import pytest

from manotorch.utils.mano_io import find_mano_model

MANO_ASSETS_ROOT = os.environ.get("MANO_ASSETS_ROOT", "assets/mano")


def _has_mano_models():
    try:
        for side in ("right", "left"):
            find_mano_model(MANO_ASSETS_ROOT, side)
    except FileNotFoundError:
        return False
    return True


requires_mano = pytest.mark.skipif(
    not _has_mano_models(),
    reason=f"MANO models not found under {MANO_ASSETS_ROOT}/models (set MANO_ASSETS_ROOT)",
)


@pytest.fixture(scope="session")
def mano_root():
    return MANO_ASSETS_ROOT
