# Changelog

This repository is a modified version of [lixiny/manotorch](https://github.com/lixiny/manotorch), forked by [IRVLUTD](https://github.com/IRVLUTD) from upstream commit `2f6a701` (2025-07-16).
As required by Section 4(b) of the [Apache License 2.0](LICENSE), this file lists the modifications made in this fork and their dates.
The full history is available in the git log.

## [Unreleased]

- Organize MANO_Poses QC into symmetric left/right data/result folders and three root-level PDFs: one
  combined statistics/paired-image report and two separate hand atlases. Preserve historical fitting,
  use portable manifest paths and expose a separate atlas PDF destination. Use a proper side-specific
  display rotation to show both hands upright while preserving handedness and measured errors.
- Keep subdivision caches created or transferred during inference warmup reusable for training backward;
  match indexed CUDA accuracy filenames in the right-only statistics publisher.

- Add paired right/left MANO_Poses QC bundles with full scanner-frame parameters, targets, independent NumPy
  references, raw dense models and a source snapshot. Portable tools support cached GPU atlas rendering,
  fresh inference, left statistics and relative-path checksum verification without the original dataset folder.
- Distinguish corrected MANO_LEFT implementation accuracy from left/right asset mirror residual; retain the
  official equivalent left articulation vectors, mirror global rotation/translation and verify triangle winding.

### Registration validation and script audit (2026-10-08)

- Add original-registration accuracy/stability/inference/fitting validation and opaque target/reconstruction/error
  QC rendering under `scripts/`. Read independent scanner-frame targets, map 16 MANO joints correctly, audit
  every file, and retain source/model hashes, per-hand errors and separate metric-free fitting timings.
- Add an optional QC statistics PDF builder with complete per-registration CSV/JSON and a proposed full-atlas index;
  document the separation between registration coverage, representative preview and synthetic sequence QC.
- Add complete A3 registration-atlas rendering with opaque lit surfaces, fixed global error scale, native text,
  subject bookmarks, linked worst-error index, selected-page PNG exports and checked sample/PDF provenance.
- Include the final optimizer update in fitting error curves and thresholds (result schema 2), capture peak
  fitting allocation before diagnostics, and guard CUDA-graph step markers on older PyTorch versions. Only
  compiled CUDA measurements invoke those markers; eager/CPU runs avoid unnecessary Inductor initialization.
- Interleave seeded isolated anatomy benchmarks and synchronize indexed CUDA devices. Improve missing/empty
  real-data diagnostics, solve PCA coefficients without explicitly inverting the basis, and clarify notebook
  working-directory and demo camera descriptions. Add asset-independent script regressions to public CPU CI.

### Follow-up (2026-10-05)

- Cache `UpSampleLayer` edge indices and child faces in nonpersistent device buffers. Existing two-argument
  calls invalidate by faces identity/version; explicit `prepare()` snapshots support repeated native tensor
  execution without CPU topology work. Preserve edge order, winding, output ownership and differentiable
  midpoint interpolation; add cache, gradient, device and minimum-version regression coverage.
- Enlarge README demo models and legends, use upright display rotations and bounded camera sweeps, highlight
  the composed index curl, and add an anchor interpolation illustration. Keep error-correction surfaces opaque
  and recompute displayed poses/losses after the labeled optimizer update.
- Add documented subdivision benchmarks and standalone Triton kernel feasibility experiments. Experimental
  kernels are not imported by the package, do not change the default eager backend and are not release-ready.

## [0.1.0] (2026-10-05)

Changes developed from 2026-10-03 to 2026-10-05. Release artifacts are the wheel and source distribution
on [GitHub Releases](https://github.com/IRVLUTD/manotorch/releases/tag/v0.1.0).

### Review follow-up (2026-10-05)

- Rotation coefficients use squared-angle Taylor branches near zero, avoiding NaN in second derivatives at the
  identity. Euler extraction handles all twelve conventions at exact gimbal lock by setting the last angle to zero;
  the selected inverse at a singularity is not smooth. Inactive atan2 branches are also protected for older PyTorch.
- Skinning supports `torch.func.jacrev`, `vmap` and `jvp`. Dynamo currently rejects custom JVPs, so compilation uses
  equivalent native operations that Inductor can fuse; eager execution keeps the custom backward.
- FK template rotations and wrist-closed faces are refreshed after state-dict loading, without changing persistent keys.
- Anatomy loss copies and validates configurations and vectorizes its interval penalties, preserving its output order,
  reduction modes and editable configuration lists. Device/dtype changes initialize its nonpersistent limit buffers;
  explicit module dtype conversions invalidate rounded limits so they are rebuilt from their degree strings.
- Model discovery also accepts flat asset folders and legacy `*_new.pkl` / `*_np.pkl` files. A restricted Latin-1 bytes
  reconstruction supports protocol-2 NumPy pickles without importing scipy or arbitrary codecs.
- Added regression coverage, explicit `--require-mano` validation, fixed Ruff rules, CPU CI and installed-wheel resource
  checks. Minimum supported PyTorch is now 2.0.1, validated with Python 3.10 / NumPy 1.26; CUDA validation uses 2.11.
- Added frozen-target MANO fitting benchmarks, selected ARCTIC/HO-Cap validation samples and downstream numerical audits.
  Raw licensed data stays outside version control. Rewrote the anatomical-feature documentation and corrected the
  float64 precision claim: model buffers are initially rounded to float32.
- Historical timings below describe the original optimization round. Follow-up performance and convergence results
  are documented in README and `doc/benchmark.md`; additional zero-rotation safeguards change eager operation counts.
- Eager rotation coefficients are evaluated together, sharing Taylor/sqrt/sinc work. The protected small-angle
  polynomial drops a term below float64 epsilon, retaining zero-angle second derivatives. The CUDA rotation probe
  decreases from 42 to 28 launches; interleaved full-mesh measurements reduce forward time by 6.1–9.1% and
  forward+backward by 7.5–10.6% versus the first correctness fix. Added a pinned eager comparison with upstream,
  manopth and smplx in `scripts/benchmark_layers.py` and README's "Other MANO layers".
- The error-correction demo now uses an opaque, smooth-shaded hand surface. Regenerated its README GIF and
  added descriptive captions and alternative text for the demo animations.
- Added an interleaved complete eager fitting benchmark with and without anatomy loss, using matched frozen
  targets, basis buffers, limits and Adam state across current/Claude/upstream manotorch. README reports full-loop
  timings; `scripts/README.md` documents dependencies, commands, input/output formats and scope for every script.
- Added a tag-triggered GitHub Release workflow: public CPU tests and lint must pass before the wheel and source
  distribution are built, checked and published. Licensed MANO models are excluded from release artifacts.
- Release code and documentation are in English. The development branch may retain its acceptance report;
  CI prevents that report from entering master or release tags and checks other source documents for CJK text.

`ManoLayer` keeps its constructor arguments (including `**kargs`), its `th_*` buffers with their names, shapes and state-dict keys, its methods and `MANOOutput`. Outputs match the previous version to float32 rounding and the official MANO code to 1.7e-4 mm in float32 (1.3e-5 mm in float64). Incompatible changes are listed under Behavior changes and Removed.

### Performance

`ManoLayer` (45 PCA components, pose and shape gradients) before (`ad515f0`) and after, on the same RTX 4090 with PyTorch 2.7 (shared machine, rounded):

| | before | after |
| --- | --- | --- |
| CUDA forward, batch 1 | 6.8 ms | 3.3 ms (0.09 ms with `torch.compile(mode="reduce-overhead")`, PyTorch 2.11) |
| CUDA forward + backward, batch 1 | 23 ms | 13 ms |
| CUDA forward, batch 8192 | 17 ms | 7.9 ms |
| CUDA forward + backward, batch 8192 | 33 ms | 20 ms |
| CUDA peak memory, forward + backward, batch 8192 | 1593 MiB | 1059 MiB |
| CPU (8 threads) forward / forward + backward, batch 1024 | 150 / 250 ms | 50 / 105 ms |

- Rewritten forward: pose and shape blend shapes as single matrix products, forward kinematics on rotations and translations over the five finger chains, and skinning without the `(B, 4, 4, 778)` intermediate. The last skinning step is a small `torch.autograd.Function` (elementwise forward and backward, itself differentiable).
- No host-device synchronization on the GPU (in `ManoLayer`, `AxisLayerFK`, `AnchorLayer` and `AnatomyConstraintLossEE`), and the layer compiles into a single graph with `torch.compile`.
- `manotorch/utils/geometry.py` rewritten from the textbook formulas (Rodrigues, Shepperd, closed-form Euler angles):
  the original optimized `axis_angle_to_matrix` launched 18 CUDA kernels instead of 53. The October 5 follow-up's
  zero-angle second-derivative safeguards initially increased this to 42 in an eager `(128,16,3)` probe; the subsequent
  coefficient batching reduces it to 28 while retaining those safeguards. Compile can fuse operations.
- New `joints_only=True` forward for callers that need only the joints: batch 16384 in 1.7 ms instead of 15 ms and 104 MiB instead of 1.4 GiB.

### Added

- `ManoLayer.forward(pose, betas, transl)`: a translation in meters added after the `center_idx` centering (upstream issue #19; with `center_idx=None`, as manopth's `th_trans`).
- `ManoLayer.forward(..., joints_only=True)`: skins only the 5 fingertip vertices and returns `verts=None`.
- `ManoLayer(fix_left_shapedirs=False)`: when True, mirrors the x component of the left-hand shape blend shapes ([smplx#48](https://github.com/vchoutas/smplx/issues/48)). Off by default, matching the official model.
- `ManoLayer.th_closed_faces`: the wrist-closed faces as a buffer that follows the layer's device (upstream issue #4).
- Model loading without chumpy, scipy or the MANO `webuser` code (`manotorch/utils/mano_io.py`): the official pickles are read through a restricted unpickler that refuses any class a MANO file does not contain. `MANO_{SIDE}.npz` files (no pickle) are preferred when present; `tools/mano_pkl_to_npz.py` creates them.
- Tests (`tests/`): against an independent float64 MANO implementation, model loaders, rotation conversions, left/right mirror symmetry, first- and second-order gradients, and runtime guards (no GPU synchronization, no warnings).
- `scripts/compare_mano_layers.py` and a README table comparing the conventions and accuracy of manopth, smplx `MANO` / `MANOLayer`, upstream manotorch and the official MANO code.

### Behavior changes

- `AxisLayerFK.forward()` returns left-hand Euler angles in the right-hand convention (twist and spread change sign): mirrored hands give equal angles, `compose()` inverts `forward()` for both hands, and the `AnatomyConstraintLossEE` limits act in the same anatomical direction for both hands.
- `AxisLayerFK.compose()` no longer modifies its input.
- `AnchorLayer()` loads the anchors shipped with the package (`manotorch/assets/anchor`); the `anchor_root` default changed from `"assets/anchor"` to `None`.
- Rotation conversions: `matrix_to_quaternion` returns `w >= 0`; `quaternion_to_axis_angle` accepts non-unit quaternions and returns angles in `[0, pi]`; `matrix_to_euler_angles` no longer returns NaN at gimbal lock.
- Buffers are float32 regardless of `torch.set_default_dtype`; deprecated `torch.norm`, `torch.cross` and `torch.Tensor(ndarray)` calls replaced by their PyTorch 2.x equivalents.

### Removed

- `AxisLayer`, `manotorch/utils/quatutils.py`, `manotorch/utils/rodrigues.py` (deprecated), `manotorch/utils/visutils.py` (Open3D viewers), the bundled MANO code (`mano/webuser`) and `tools/clean_ch.py`.
- Dependencies `chumpy`, `deprecation`, `open3d`, `opencv-python`, `scipy` and `matplotlib`; `pyvista`, `trimesh` and `tqdm` moved to the `vis` extra.

### Packaging, demos and documentation

- Packaging in `pyproject.toml` only (PEP 621; `setup.py` removed); core dependencies are `torch` and `numpy`.
- The conda environment files are replaced by a uv setup: `uv sync` creates Python 3.12 + PyTorch 2.11.0 (CUDA 12.6) from `uv.lock`.
- Demo scripts fixed (missing `zero_grad` in `simple_anatomy_loss.py`, fingers sampled into the palm in `simple_app.py`), rendered with pyvista (`--gif` for off-screen GIFs), and the README GIFs regenerated.
- `README.md` rewritten for the fork; the previous one is kept as `README.old.md`.

### License (2026-10-03)

- The fork follows upstream's relicensing from GPL-3.0 to the Apache License 2.0 (upstream commit `a2a70c5`, 2026-02-03, merged here). Versions of this fork published before this change remain available under GPL-3.0.
- Added `NOTICE` (manotorch and manopth origin, third-party code, MANO license), shipped with the package.

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
