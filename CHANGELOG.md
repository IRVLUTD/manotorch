# Changelog

This repository is a modified version of [lixiny/manotorch](https://github.com/lixiny/manotorch), forked by [IRVLUTD](https://github.com/IRVLUTD) from upstream commit `2f6a701` (2025-07-16).
As required by Section 5(a) of the [GNU GPL v3](LICENSE), this file lists the modifications made in this fork and their dates.
The full history is available in the git log.

## [Unreleased]

### 2026-10-03

- Rewrote `README.md` for the fork: fork notice, updated installation steps, and a License section that separates GPL-3.0 code, MANO-licensed files and third-party code. The previous README is archived as `README.old.md`.
- Added this `CHANGELOG.md`.
- `manotorch/utils/geometry.py`: added the PyTorch3D copyright and BSD license notice that this file was adapted from.
- Packaging moved entirely to `pyproject.toml` (PEP 621, setuptools backend); `setup.py` was removed. The original authors are credited alongside the fork maintainer. Core dependencies are `torch` and `numpy`; visualization packages moved to the `vis` extra and development tools to the `dev` dependency group. Only the `manotorch` package is installed.
- Added `manotorch/utils/mano_io.py`, which reads the official MANO pickles with numpy only, through a restricted unpickler that refuses any class a MANO file does not contain. `ManoLayer` no longer depends on chumpy, scipy, OpenCV or the bundled `mano/webuser` code, and no longer needs the `MANO_*_new.pkl` conversion. The model arrays, including the memory layout of `J_regressor`, are identical to the previous loader's.
- `ManoLayer` also loads `MANO_{SIDE}.npz` model files (dense arrays, no pickle), preferring them over `.pkl` when both exist; `tools/mano_pkl_to_npz.py` creates them. Both formats give bit-identical outputs.
- Replaced the conda environment files (`environment.yaml`, `environment.py310.cu121.yaml`) with a uv setup: `uv sync` creates a Python 3.12 + PyTorch 2.11.0 (CUDA 12.6) environment from the committed `uv.lock`. PyTorch comes from the CUDA 12.6 index on Linux and Windows; the version pin applies to the development environment only.

### Optimization (2026-10-03)

Apart from the removed deprecated code and the `AnchorLayer` default path listed below, the public API is unchanged: `ManoLayer` keeps its constructor arguments (including `**kargs`), its `th_*` buffers with their names, shapes and state-dict keys, `kintree_parents`, its methods, and `MANOOutput`; `AxisLayerFK` keeps its buffers and 3-tuple output. Outputs match the previous implementation to float32 rounding (max relative difference 2e-6 over 238 reference tensors, gradients included).

- `ManoLayer`: rewritten skinning: pose and shape blend shapes as single matrix products, forward kinematics on rotations and translations over the 5 finger chains, linear blend skinning without the (B, 4, 4, 778) intermediate, and constant index tensors registered as non-persistent buffers. No host-device synchronization remains, and the layer compiles into a single graph with `torch.compile`.
- `manotorch/utils/geometry.py`: ported the synchronization-free PyTorch3D conversions (`torch.sinc`, `torch.where`, `torch.gather`); `matrix_to_quaternion` still returns non-standardized quaternions, as before.
- New `ManoLayer(fix_left_shapedirs=False)` argument: when True, negates the x component of the left-hand shape blend shapes ([smplx#48](https://github.com/vchoutas/smplx/issues/48)) so that left and right hands with the same betas are mirrors. Off by default, matching the official model.
- `AxisLayerFK.compose()` no longer modifies its input tensor for the left hand; `AxisLayerFK` and `AxisAdaptiveLayer` build their constants once, as buffers.
- Removed the deprecated `AxisLayer` class and the deprecated `manotorch/utils/quatutils.py` and `manotorch/utils/rodrigues.py`, and with them the `deprecation` dependency.
- Anchor definitions moved into the package (`manotorch/assets/anchor`); `AnchorLayer()` loads them by default (its `anchor_root` default changed from `"assets/anchor"` to `None`), so it works from any directory and from a regular (non-editable) install.
- Removed the bundled official MANO code (`mano/webuser`, MANO license) and `tools/clean_ch.py`, both unused since the numpy-only loader.
- Added a pytest suite (`tests/`) that checks `ManoLayer` against an independent float64 MANO implementation, the npz/pickle loaders, `AxisLayerFK` and the left/right mirror symmetry.

Measured on an RTX 4090 with PyTorch 2.7 (shared machine, so absolute times are noisy):

| | before | after |
| --- | --- | --- |
| `ManoLayer` construction | 417 ms | 18 ms |
| CUDA forward, batch 1 | 8.8 ms | 5.0 ms (0.22 ms with `torch.compile(mode="reduce-overhead")`) |
| CUDA forward, batch 8192, peak memory (forward + backward) | 1607 MiB | 1151 MiB |
| CPU forward, batch 1024 | 488 ms | 311 ms |

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
