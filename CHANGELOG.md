# Changelog

This repository is a modified version of [lixiny/manotorch](https://github.com/lixiny/manotorch), forked by [IRVLUTD](https://github.com/IRVLUTD) from upstream commit `2f6a701` (2025-07-16).
As required by Section 4(b) of the [Apache License 2.0](LICENSE), this file lists the modifications made in this fork and their dates.
The full history is available in the git log.

## [Unreleased]

### Translation input and on-device closed faces (2026-10-04)

- `ManoLayer.forward(pose, betas, transl)` (upstream issue #19): a `(B, 3)` or `(1, 3)` translation in meters, added to `verts`, `joints` and `transforms_abs` after the `center_idx` centering, so the center joint lands at `transl`. With `center_idx=None` this matches manopth's `th_trans` (checked: 1.2e-7 m, float32 rounding); unlike manopth, a translation does not disable the centering. `joints_only` is now keyword-only.
- `ManoLayer.th_closed_faces` (upstream issue #4): the wrist-closed faces as a non-persistent buffer that follows the layer's device. `get_mano_closed_faces()` still returns a new CPU tensor, as before.

### Joints-only forward (2026-10-04)

- `ManoLayer.forward(pose, betas, joints_only=True)`: computes the joints from the joint-regressed shape basis and skins only the 5 fingertip vertices; `verts` is `None`. Joints and `transforms_abs` match the full forward to 1e-16 m in float64. On an RTX 4090 with N=16384: 1.7 ms instead of 19.5 ms, 104 MiB instead of 1.4 GiB peak memory; no gain below a few hundred hands. The full forward is unchanged (bit-identical). The derived bases are recomputed on every call, so in-place edits of the `th_*` buffers after construction (as done by downstream left-shapedirs fixes) are honored.

### PyTorch 2.x API (2026-10-04)

- `torch.norm` -> `torch.linalg.vector_norm`, `torch.cross` -> `torch.linalg.cross`, and `torch.Tensor(ndarray)` -> `torch.as_tensor(..., dtype=torch.float32)`.
- Every floating-point buffer and constant is created with an explicit `float32` dtype: `torch.Tensor(...)`, `torch.zeros(...)` and `torch.tensor(...)` followed the global default dtype, so after `torch.set_default_dtype(torch.float64)` some buffers became float64 and `AxisLayerFK` failed on mixed dtypes.
- Outputs are bit-identical to the previous commit (238 reference tensors), and `J_regressor` keeps its Fortran-order layout.

### Demo scripts (2026-10-04)

- `vis` extra: added `imageio`, which pyvista needs to write GIFs (the scripts failed at `open_gif`).
- `scripts/simple_anatomy_loss.py`: the optimization never reset the gradients (`optimizer.zero_grad()` was missing), so they accumulated over the 5000 iterations; fixed, now 1000 iterations at `lr=1e-2`, which bring all three index finger joints into their limits. Rendered with pyvista instead of Open3D, so it can write a GIF off-screen; removed unused imports and a loop variable that shadowed the iteration counter.
- `scripts/simple_app.py`, `scripts/simple_compose.py`: interactive window by default, `--gif <path>` to render a GIF off-screen (they always wrote `simple_app_new.gif` / `simple_compose_new.gif` to the working directory before); `simple_app.py` drew all 45 PCA coefficients from N(0, 1), which often pushed fingers into the palm; it now samples a natural pose in the anatomy aligned angle space (random per-finger curl with coupled joint flexion, small MCP spread, no twist, no global rotation) and composes both hands from it with `AxisLayerFK.compose`, on the CPU so the pose does not depend on the device; `simple_compose.py` checks that the composed left hand is the mirror of the right one, in PCA or axis-angle mode (`--no-pca`), and drops the obsolete left-hand angle flip comment.
- `scripts/test_compatibility.ipynb`: updated to the smplx fingertips (the joint assertion against manopth failed since the fork changed them; the comparison with Omid's MANO now needs only the joint reordering), puts this repository first on `sys.path` so an installed manotorch is not tested instead, and outputs cleared. Checked against manopth `4f1dcad` and MANO `5869ab0`: vertices and joints agree to 5e-8 m.
- New `scripts/_common.py` shared by the three scripts (device choice, colors, hand and axis drawing, legend, GIF recording). Rendering: light salmon / light blue hands (a color-blind safe pair) so the red / green / blue axes stand out, the two mirrored hands drawn 4 cm apart instead of overlapping at the wrist, back-face culling and depth peeling so translucent hands no longer show their inner faces as dark patches, thinner axis arrows, and a legend in every GIF.
- GIFs: 768 px, written by an explicit camera orbit (pyvista's `orbit_on_path` refits the view, so its orbit radius had no effect and the legend overlapped the hands), and re-encoded with one palette shared by all frames: about 1 MB each instead of 2.5 MB.
- Removed `manotorch/utils/visutils.py` (Open3D viewer helpers) and `open3d` from the `vis` extra: nothing in this repository or its known users imports them any more (about 900 MB less in the environment).
- Regenerated `doc/axis_new.gif`, `doc/simple_compose_new.gif` and `doc/pose_correction.gif` with the current code (the last one was still upstream's image).

### License (2026-10-03)

- The fork now follows upstream's license: upstream relicensed manotorch from GPL-3.0 to the Apache License 2.0 in commit `a2a70c5` (2026-02-03), which is merged here. Versions of this fork published before this change remain available under GPL-3.0.
- Added `NOTICE` with the required attributions (manotorch and manopth origin, PyTorch3D's BSD 3-Clause code, the MANO license), shipped with the package; `pyproject.toml` declares `license = "Apache-2.0"`.

### 2026-10-03

- Rewrote `README.md` for the fork: fork notice, updated installation steps, and a License section that separates GPL-3.0 code, MANO-licensed files and third-party code. The previous README is archived as `README.old.md`.
- Added this `CHANGELOG.md`.
- `manotorch/utils/geometry.py`: added the PyTorch3D copyright and BSD license notice that this file was adapted from.
- Packaging moved entirely to `pyproject.toml` (PEP 621, setuptools backend); `setup.py` was removed. The original authors are credited alongside the fork maintainer. Core dependencies are `torch` and `numpy`; visualization packages moved to the `vis` extra and development tools to the `dev` dependency group. Only the `manotorch` package is installed.
- Added `manotorch/utils/mano_io.py`, which reads the official MANO pickles with numpy only, through a restricted unpickler that refuses any class a MANO file does not contain. `ManoLayer` no longer depends on chumpy, scipy, OpenCV or the bundled `mano/webuser` code, and no longer needs the `MANO_*_new.pkl` conversion. The model arrays, including the memory layout of `J_regressor`, are identical to the previous loader's.
- `ManoLayer` also loads `MANO_{SIDE}.npz` model files (dense arrays, no pickle), preferring them over `.pkl` when both exist; `tools/mano_pkl_to_npz.py` creates them. Both formats give bit-identical outputs.
- Replaced the conda environment files (`environment.yaml`, `environment.py310.cu121.yaml`) with a uv setup: `uv sync` creates a Python 3.12 + PyTorch 2.11.0 (CUDA 12.6) environment from the committed `uv.lock`. PyTorch comes from the CUDA 12.6 index on Linux and Windows; the version pin applies to the development environment only.

### Optimization (2026-10-03)

Apart from the removed deprecated code, the `AnchorLayer` default path and the left-hand Euler angles of `AxisLayerFK.forward` listed below, the public API is unchanged: `ManoLayer` keeps its constructor arguments (including `**kargs`), its `th_*` buffers with their names, shapes and state-dict keys, `kintree_parents`, its methods, and `MANOOutput`; `AxisLayerFK` keeps its buffers and 3-tuple output. Outputs match the previous implementation to float32 rounding (max relative difference 2e-6 over 238 reference tensors, gradients included).

- `ManoLayer`: rewritten skinning: pose and shape blend shapes as single matrix products, forward kinematics on rotations and translations over the 5 finger chains, linear blend skinning without the (B, 4, 4, 778) intermediate, and constant index tensors registered as non-persistent buffers. No host-device synchronization remains, and the layer compiles into a single graph with `torch.compile`.
- `manotorch/utils/geometry.py`: ported the synchronization-free PyTorch3D conversions (`torch.sinc`, `torch.where`, `torch.gather`); `matrix_to_quaternion` still returns non-standardized quaternions, as before.
- New `ManoLayer(fix_left_shapedirs=False)` argument: when True, negates the x component of the left-hand shape blend shapes ([smplx#48](https://github.com/vchoutas/smplx/issues/48)) so that left and right hands with the same betas are mirrors. Off by default, matching the official model.
- `AxisLayerFK.compose()` no longer modifies its input tensor for the left hand; `AxisLayerFK` and `AxisAdaptiveLayer` build their constants once, as buffers.
- `AxisLayerFK.forward()` returns the left-hand rotations and Euler angles in the right-hand convention, the one `compose()` already expected: the twist and spread angles change sign for the left hand (`R -> P R P`, `P = diag(-1, -1, 1)`). Mirrored left and right poses now give equal angles, `compose(forward(T))` returns the pose for both hands (before, only for the right hand), and the `AnatomyConstraintLossEE` limits act in the same anatomical direction for both hands. `T_g_a` and the `TMPL_*` buffers are unchanged.
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
