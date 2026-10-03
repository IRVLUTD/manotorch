# Changelog

This repository is a modified version of [lixiny/manotorch](https://github.com/lixiny/manotorch), forked by [IRVLUTD](https://github.com/IRVLUTD) from upstream commit `2f6a701` (2025-07-16).
As required by Section 5(a) of the [GNU GPL v3](LICENSE), this file lists the modifications made in this fork and their dates.
The full history is available in the git log.

## [Unreleased]

### 2026-10-03

- Rewrote `README.md` for the fork: fork notice, updated installation steps, and a License section that separates GPL-3.0 code, MANO-licensed files and third-party code. The previous README is archived as `README.old.md`.
- Added this `CHANGELOG.md`.
- `manotorch/utils/geometry.py`: added the PyTorch3D copyright and BSD license notice that this file was adapted from.
- Packaging moved entirely to `pyproject.toml` (PEP 621, setuptools backend); `setup.py` was removed. The original authors are credited alongside the fork maintainer. Core dependencies are `torch`, `numpy` and `deprecation`; visualization packages moved to the `vis` extra and development tools to the `dev` dependency group. Only the `manotorch` package is installed.
- Added `manotorch/utils/mano_io.py`, which reads the official MANO pickles with numpy only. `ManoLayer` no longer depends on chumpy, scipy, OpenCV or the bundled `mano/webuser` code, and no longer needs the `MANO_*_new.pkl` conversion. Outputs are bit-identical to the previous loader.
- Replaced the conda environment files (`environment.yaml`, `environment.py310.cu121.yaml`) with a uv setup: `uv sync` creates a Python 3.12 + PyTorch 2.11.0 (CUDA 12.6) environment from the committed `uv.lock`. PyTorch comes from the CUDA 12.6 index on Linux and Windows; the version pin applies to the development environment only.

## [0.0.3]

### 2026-02-02

- `mano/webuser/smpl_handpca_wrapper_HAND_only.py`: fixed the remaining chumpy dependency when loading the chumpy-free model files.

### 2026-01-28

- Added `tools/clean_ch.py`, which converts the MANO model files into chumpy-free pickles (`MANO_*_new.pkl`); `ManoLayer` loads them when present.
- `manotorch/manolayer.py`: replaced the `MANOOutput` namedtuple with a `@dataclass`, and unified the fingertip vertex selection (indices aligned with smplx) across rotation modes.
- Fixed the left hand: `AxisAdaptiveLayer` now aligns the global rotation axis of the left hand, and `AxisLayerFK` composes correct axis angles for the left hand.
- Updated the demo scripts (`scripts/simple_app.py`, `scripts/simple_compose.py`) to support both hand sides, with new demo GIFs.
- Updated `setup.py` and `pyproject.toml` for correct dependencies, bumped the version to 0.0.3, and updated the README installation section.

### 2025-10-23

- Updated `manotorch/axislayer.py` and `manotorch/manolayer.py` for compatibility with PyTorch 2.7.
