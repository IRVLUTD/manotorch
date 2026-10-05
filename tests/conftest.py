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


def pytest_addoption(parser):
    parser.addoption("--require-mano", action="store_true", help="Fail if either licensed MANO model is missing")


def pytest_sessionstart(session):
    if session.config.getoption("--require-mano") and not _has_mano_models():
        raise pytest.UsageError(f"Both MANO models are required under {MANO_ASSETS_ROOT}")


def pytest_report_header(config):
    status = "available" if _has_mano_models() else "MISSING: model-dependent tests will be skipped"
    return f"Licensed MANO models: {status}; root={MANO_ASSETS_ROOT} (use --require-mano for release validation)"


@pytest.fixture(scope="session")
def mano_root():
    return MANO_ASSETS_ROOT
