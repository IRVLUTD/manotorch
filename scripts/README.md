# Script usage

Run the Python commands below **from the repository root**, rather than from `scripts/`:

```sh
uv sync
uv run python scripts/benchmark_fitting.py --help
```

The root README describes installation and the licensed MANO models. Most scripts expect
`assets/mano/models/MANO_RIGHT.pkl` and `MANO_LEFT.pkl` (canonical NPZ files are also accepted by this fork).
Use `--mano-assets-root /path/to/mano` where the script exposes it. The cross-library scripts additionally
expect canonical pickles for legacy implementations. Keep models, dataset samples, archived sources and raw
results under ignored `assets/mano/`, `data/` or `thirdparty/`; these resources are not distributed with the package.

## Choose a script

| Script | Purpose | Output |
| --- | --- | --- |
| `simple_app.py` | View mirrored hands with anatomical axes or 32 anchors | Interactive window or camera-sweep GIF |
| `simple_compose.py` | Compose both hands from the same anatomical Euler angles | Interactive window or camera-sweep GIF |
| `simple_anatomy_loss.py` | Correct an implausible index-finger pose using the anatomy prior | Interactive window or optimization GIF |
| `benchmark_registrations.py` | Validate original MANO registration targets, derivatives, inference and batched fitting | Manifest, per-hand accuracy CSVs, diagnostic NPZs and JSON |
| `render_registration_qc.py` | Render matched target/reconstruction/error views of representative registrations | QC PNG and selection/error manifest |
| `render_registration_atlas.py` | Render all registrations with global errors and linked PDF navigation | Complete A3 QC PDF, page index/manifest and selected PNG pages |
| `summarize_registration_qc.py` | Summarize every registration and build a proposed atlas index | Statistics PDF, complete CSV and JSON |
| `prepare_registration_bundle.py` | Store paired right/left full poses, targets, references, raw models and source snapshot | Self-contained local NPZ bundle and complete accuracy/stability CSV/JSON |
| `reinfer_registration_bundle.py` | Recompute cached outputs using only the bundle's models/poses/source | New validated prediction archive; original archive preserved |
| `summarize_registration_mirror.py` | Summarize both hands, paired visuals, mirror residual and historical fitting | One combined four-page QC PDF; root and per-side CSV/JSON |
| `verify_registration_bundle.py` | Write or verify a portable relative-path SHA-256 inventory | Bundle manifest or verification result; standard library only |
| `_registration_geometry.py` | NumPy-only Rodrigues and hashing for portable QC tools | Helper module; not a standalone command |
| `registration_bundle_README.md` | Usage/schema template copied into each self-contained bundle | Bundle-local README; not a command |
| `test_benchmark_scripts.py` | Regression checks for benchmark bookkeeping and joint order | Pytest results; no MANO models needed |
| `benchmark_fitting.py` | Measure MANO model runtime and complete fitting, without anatomy loss | Frozen targets, manifest and result JSON |
| `benchmark_fitting_anatomy.py` | Compare complete eager fitting with/without anatomy loss across three manotorch revisions | Interleaved timing and convergence JSON |
| `benchmark_layers.py` | Compare eager layer runtime with upstream, manopth and smplx | Timing groups, numerical errors and source hashes in JSON |
| `benchmark_anatomy.py` | Isolate the anatomy-loss forward/backward cost | Baseline/current timing JSON |
| `benchmark_upsample.py` | Compare uncached subdivision with automatic and prepared topology caches | Cold preparation and warm timing JSON |
| `benchmark_kernels.py` | Evaluate optional Triton forward prototypes against eager/compiled operations and full workloads | CUDA float32 kernel feasibility JSON |
| `prepare_fitting_data.py` | Select valid ARCTIC/HO-Cap observations from UHAS exports | Target NPZs and selection manifests |
| `compare_benchmarks.py` | Match two `benchmark_fitting.py` runs and export their comparison | CSV plus printed error summary |
| `check_downstream.py` | Audit two downstream vendors numerically | JSON of geometry/gradient and anatomy-convention checks |
| `compare_mano_layers.py` | Compare numerical accuracy against the official chumpy model | Printed error table |
| `test_compatibility.ipynb` | Interactive comparison with manopth and Omid's MANO | Notebook assertions and outputs |
| `_common.py` | Shared rendering helpers used by the demos | Helper module; not a standalone command |

These commands are research demos and benchmarks, not a production fitting API. Default configuration and loss
weights are demonstration choices. For benchmark methodology and interpretation, see [doc/benchmark.md](../doc/benchmark.md).

## Visualization demos

Install the rendering extras once:

```sh
uv sync --extra vis
uv run --extra vis python scripts/simple_app.py
uv run --extra vis python scripts/simple_app.py --mode anchor --seed 0
uv run --extra vis python scripts/simple_compose.py
uv run --extra vis python scripts/simple_anatomy_loss.py
```

Without `--gif`, a script opens an interactive PyVista window. With `--gif`, it renders off-screen;
the machine still needs a working VTK/OpenGL rendering backend. Write new previews under `data/`:

```sh
uv run --extra vis python scripts/simple_app.py --gif data/axis.gif
uv run --extra vis python scripts/simple_app.py --mode anchor --gif data/anchors.gif
uv run --extra vis python scripts/simple_compose.py --no-pca --gif data/compose.gif
uv run --extra vis python scripts/simple_anatomy_loss.py \
  --iters 1000 --lr 0.01 --draw-every 10 --gif data/pose_correction.gif
```

- `simple_app.py`: `--mode axis` is the default; `--seed 0` controls the sampled pose. The seed is applied on
  CPU so the pose selection does not depend on which device renders it. Anchor mode shows 32 purple anchors
  per hand, highlighting anchor 0 in gold and outlining the triangle used for its barycentric interpolation.
- `simple_compose.py`: defaults to expressing the composed articulation as 45 PCA coefficients;
  `--no-pca` feeds the full axis-angle pose directly. The script prints whether the resulting poses are mirrored.
  The gold index finger has MCP spread 30 degrees and MCP/PIP/DIP bends of 90 degrees; only its axes are drawn.
- `simple_anatomy_loss.py`: defaults to 1000 Adam updates, learning rate 0.01 and a frame every 10 updates;
  it optimizes articulation with anatomy loss alone. It is **not** an observation-fitting benchmark. Keep
  `--iters >= 5` because its learning-rate scheduler uses `iters // 5` as the step size. The surface is opaque;
  each frame's pose and loss are recomputed after the labeled number of optimizer updates.
- All three automatically choose CUDA, then MPS, then CPU, and accept `--mano-assets-root`.
  The GIF colors are red/twist, green/spread and blue/bend. GIF playback does not measure runtime.

Axis/anchor/compose GIFs use 960-by-640 frames, large legends and 48 frames at 10 fps. Proper display rotations
place both hands upright; these rotations only affect illustrations. A bounded orthographic camera sweep keeps
the models large throughout playback and avoids edge-on views. Compose uses a more oblique angle to expose the
index curl. Error correction uses a fixed profile view, 960-by-600 frames and 101 frames with default options.
Inspect the first/middle/last frames and the camera extrema before replacing a README asset.

To deliberately replace the root README assets, use its listed `doc/*.gif` output paths and inspect the result.
Providing an existing output path replaces that file.

## Prepare benchmark sources and frozen MANO targets

The comparison baseline is Claude's optimized revision `c936b59`, not upstream manotorch. Archive it once:

```sh
mkdir -p data/benchmarks/baseline_source
git archive c936b59 | tar -x -C data/benchmarks/baseline_source
```

The fitting benchmark also needs the official training articulation arrays:

```text
data/MANO_Poses/mano_poses_v1_0/
  handsOnly_REGISTRATIONS_r_lm___POSES___R.npy
  handsOnly_REGISTRATIONS_r_lm___POSES___L.npy
```

Generate frozen targets with the archived implementation first, then measure current code with identical options:

```sh
uv run python scripts/benchmark_fitting.py \
  --implementation-root data/benchmarks/baseline_source \
  --modes eager --sides right --batches 1 128 --steps 100 --repeats 20 \
  --output data/benchmarks/baseline_eager_small.json
uv run python scripts/benchmark_fitting.py \
  --modes eager --sides right --batches 1 128 --steps 100 --repeats 20 \
  --output data/benchmarks/current_eager_small.json
```

`--targets` defaults to `data/benchmarks/mano_targets.npz`. If it already exists, the script reads it;
`--dataset`, `--samples` and `--seed` do not regenerate that file. If absent, the first run selects up to
128 training poses per hand with seed `20261005`, generates targets/initial parameters and writes a manifest.
Do not delete or replace targets between implementations. Batches larger than the stored bank repeat samples.

For cross-library benchmarks, obtain the pinned upstream sources using
[the clone/checkout commands](../doc/benchmark.md#eager-comparison-with-other-mano-layers).
The expected defaults are `data/benchmarks/manotorch_upstream`, `manopth` and `smplx`.
No benchmark installs these libraries as core manotorch dependencies or edits their timed numerical source.

## `benchmark_fitting.py`: model runtime and fitting without anatomy loss

By default this runs both hands, batches 1/32/128/1024, three tasks and both eager/compile modes, with 100 fitting
steps and 30 timing repetitions. It reports model forward, forward+backward, complete Adam steps, extra allocated
CUDA memory, first differentiated compile time, error curves and steps to a 1 mm threshold.

| Option | Meaning |
| --- | --- |
| `--tasks joints` | Fit all 21 synthetic joints while evaluating the full mesh |
| `--tasks joints_only` | Same synthetic targets; skip the full mesh using `joints_only=True` |
| `--tasks vertices` | Fit the full mesh; only supported for MANO synthetic targets |
| `--modes eager compile` | Eager and compiled prediction; **loss and Adam remain eager** |
| `--device cpu --modes eager` | CPU diagnostic without CUDA compilation |
| `--steps`, `--repeats` | Adam updates per fit and calls per runtime measurement |
| `--threshold-mm` | Error threshold used to report the first matching curve index |
| `--implementation-root` | Source folder of the implementation under test |
| `--source real --targets ...` | Read separately prepared real observations; fit only 16 common joints |

Example focused compiled comparison:

```sh
uv run python scripts/benchmark_fitting.py --sides right --batches 128 \
  --tasks joints_only --modes eager compile --steps 100 --repeats 20 \
  --output data/benchmarks/joints128.json
```

Forward retains autograd. The observation loss is coordinate MSE in metres; Adam optimizes pose, shape and
translation with learning rates 0.01/0.02/0.001. Geometry is reported as RMS Euclidean point distance in mm.
Compile targets `predict` with `fullgraph=True, mode="reduce-overhead"`; startup depends on the disk cache.
The curve index equals completed optimizer updates, including the final update (`steps + 1` entries).
Result schema version 2 fixes the old omission of the final prediction; rerun both sides before comparing
thresholds with historical schema-1 results. Threshold time is an estimate from the curve index and mean step time.
The peak fitting allocation is captured before final diagnostics. Extra allocated memory is not total
reserved memory, and CUDA-graph workspaces can already be live before a measurement.

## `benchmark_fitting_anatomy.py`: complete fitting with the anatomy prior

After preparing the frozen MANO targets and upstream/baseline folders:

```sh
uv run --with deprecation python scripts/benchmark_fitting_anatomy.py \
  --batches 1 128 1024 --steps 100 --groups 6 --anatomy-weight 0.0001 \
  --output data/benchmarks/fitting_anatomy.json
```

`deprecation` is required by the archived upstream anatomy module, not by this fork. This eager-only benchmark
uses current, Claude `c936b59` and upstream `a2a70c5`, each with its own MANO/FK/loss implementation. It always
tests the right hand and full mesh. All three receive the same right-hand reference basis buffers and default
anatomical limits at construction; legacy loading/absolute-import adapters are outside timing. manopth and smplx
are not included because they do not provide this same anatomy chain.

The objective is `mean((predicted_16_joints_m - target_16_joints_m)^2) + weight * mean_anatomy_penalty_rad`.
The 16 joints exclude all fingertips. Pose, shape and translation share frozen initialization and the Adam
settings above. Both weight zero and `--anatomy-weight` are tested; weight zero skips FK and anatomy loss.
The default weight `1e-4` is a benchmark choice, not a tuned recommendation for other datasets or loss units.

Defaults are batches 1/128/1024, 100 steps, 5 groups and 10 warmup steps. `--groups 6` balances the three
implementation orders over two rotations. Every group starts a new fit from the frozen parameters and zeroed
Adam state; warmup allocates that state before timing. The measured loop includes MANO, optional FK/Euler/loss,
backward and Adam, with synchronization only at the boundaries. Loading, construction, validation, state reset,
warmup and metric collection are excluded. Output stores every group's total ms, median total/step times,
initial/final geometric error and anatomy penalty, initial loss/gradient agreement and source/target hashes.
An anatomy prior can increase observation error while reducing its penalty; inspect both metrics.

For a quick CPU check of adapters and numerical agreement:

```sh
uv run --with deprecation python scripts/benchmark_fitting_anatomy.py \
  --device cpu --batches 1 --groups 1 --steps 2 --warmup 1 \
  --output data/benchmarks/fitting_anatomy_smoke.json
```

## `benchmark_layers.py`: eager layer comparison

```sh
uv run python scripts/benchmark_layers.py --groups 7 --repeats 20 \
  --output data/benchmarks/eager_layers.json
```

Defaults include both hands and batches 1/128/1024, 6 groups and 20 calls per group. Optional source locations
are `--upstream`, `--manopth`, `--smplx` and `--baseline`; use `--device cpu` for CPU execution. This compares
full-mesh float32/flat-mean/full-axis-angle forward and backward of a vertex MSE for pose/betas. It includes
unit/joint-order adapters and smplx MANOLayer's AA-to-matrix conversion, but excludes Adam, anatomy loss and
compilation. Construction-only array-loading shims avoid chumpy/SciPy. Ten warmups precede measurements,
implementation order rotates across groups, and results include all timings and vertex/16-joint/gradient errors.
`--before-geometry` is an optional local saved module; it is not required for upstream/current comparisons.

## `benchmark_anatomy.py`: anatomy-loss cost alone

```sh
uv run python scripts/benchmark_anatomy.py --device cuda \
  --output data/benchmarks/anatomy.json
```

This loads the archived loss from `--baseline` (default
`data/benchmarks/baseline_source/manotorch/anatomy_loss.py`) and compares it with the vectorized current loss
on random angle tensors at batches 1/128/1024. It measures loss forward+backward after 10 warmups, using
6 rotating groups of 30 calls with a fixed seed; loss and input gradients are checked before timing.
Use `--device cpu` for CPU or `--device cuda:0` for an indexed CUDA device. Optional `--batches`, `--groups`,
`--repeats` and `--seed` control the workload. It has no MANO model, FK, optimizer or fitting targets;
its speedup must not be reported as an end-to-end fitting speedup. Historical results used five groups in
baseline-then-current order; the updated script records its interleaved protocol in each JSON row.

## `benchmark_upsample.py`: repeated mesh subdivision

Prepare `baseline_source` with the archive command above, then run:

```sh
uv run python scripts/benchmark_upsample.py --devices cpu cuda --batches 1 32 128 \
  --groups 5 --repeats 3 --output data/benchmarks/upsample_cache.json
```

This compares Claude's uncached `UpSampleLayer` with the current automatic cache (`layer(vertices, faces)`)
and explicit snapshot (`prepare(faces, vertex_count)` followed by `layer(vertices)`). It uses the right MANO
open-wrist mesh, with one shared faces tensor expanded across the batch. Geometry and face order must match
exactly. Cold topology preparation is reported separately. Warm timings include interpolation and a fresh faces
output, excluding MANO evaluation and input allocation. Cached blocks use ten times more repetitions to
amortize their shorter latency. Five groups rotate method order; the JSON retains raw groups and source hashes.
Use `--devices cpu` on a machine without CUDA, and `--baseline PATH` for a different archived source folder.
This is subdivision speed, not a MANO fitting speedup. Cache usage and invalidation are documented in
[the mesh-subdivision feature](../doc/features.md#mesh-subdivision).

## `benchmark_kernels.py`: optional kernel feasibility experiment

```sh
uv run python scripts/benchmark_kernels.py --batches 1 128 1024 \
  --groups 6 --repeats 20 --steps 100 --output data/benchmarks/kernel_feasibility.json
```

This standalone experiment requires a CUDA GPU, a compatible Triton installation and the frozen target bank
prepared above. Triton is not a core manotorch dependency; the script never enables a production backend.
The tested kernels accept contiguous CUDA float32 tensors. A fused skinning forward retains the existing
PyTorch backward; a fused axis-angle conversion is **inference only**, with no custom backward. The experiment
does not establish AMP, vmap/JVP, second derivatives, compiled full-model or cross-device compatibility.

Microbenchmarks compare eager, warmed `torch.compile(fullgraph=True)` and Triton forwards for rotation and
skinning. Compile/JIT startup is excluded. Complete-model comparisons separately measure autograd-enabled
forward, forward+backward and no-grad inference. Skinning additionally runs complete 100-step right-hand fits
with and without anatomy loss, using the same `FittingCase`, targets, initialization and Adam buffers as
`benchmark_fitting_anatomy.py`. Rotation fusion is never used for these fits. Six groups alternate method order;
`--repeats` controls model/micro blocks, while `--steps` controls fitting updates. Numerical validation is outside
timing: skinning geometry/gradients must match exactly, and fitting metrics must agree within `1e-7`.
The JSON stores group timings, geometry/quality metrics, environment and target/source hashes. It has a separate
schema from `compare_benchmarks.py`. See [the kernel study](../doc/benchmark.md#optional-kernel-feasibility)
before interpreting a microbenchmark as an end-to-end speedup.

## Real observations and downstream audits

`prepare_fitting_data.py` reads UHAS `hand_poses` exports containing `arctic/*.npz` and `hocap/*.npz`.
It selects the first file in sorted order from each folder, samples up to 32 valid frames per available hand,
keeps the exported observations, and perturbs the initial parameters using seed `20261005`.
It expects `meta` JSON with `units="metres"`, `rotation="axis-angle"` and `hands`, plus `valid`,
`frame_index`, `global_orient`, `hand_pose`, `betas`, `transl` and `joints` arrays in the UHAS layout.

```sh
uv run python scripts/prepare_fitting_data.py --root /path/to/UHAS_HumanData/data/hand_poses \
  --samples 32 --output-dir data/benchmarks
uv run python scripts/benchmark_fitting.py --source real \
  --targets data/benchmarks/arctic_targets.npz --batches 32 --tasks joints joints_only \
  --output data/benchmarks/arctic_fitting.json
uv run python scripts/benchmark_fitting.py --source real --sides left \
  --targets data/benchmarks/hocap_targets.npz --batches 32 --tasks joints joints_only \
  --output data/benchmarks/hocap_fitting.json
```

Choose `--sides` according to the manifest; the selected HO-Cap sequence has valid left hands only.
The preparation writes `<dataset>_targets.npz` and `.manifest.json`, and replaces existing files in its output
directory. Original UHAS source files are read only. Matching real-data poses requires checking mean pose,
units, centering, shapedirs and joint conventions as described in [the benchmark document](../doc/benchmark.md).

`check_downstream.py` expects a parent directory containing `dr_egostereo` and `FoundationEgo_Annotation`,
each with `third_party/manotorch/manotorch` and licensed `weights/mano` assets; Foundation also needs `src`.

```sh
uv run --with scipy --with deprecation python scripts/check_downstream.py \
  --root /path/to/Deepreach_AI --output data/benchmarks/downstream.json
```

It compares vertices, 16 joints and pose/shape gradients on CPU; left shapedirs are corrected exactly once
to match the existing wrappers. It also checks Foundation's left-hand Euler convention and `LIMITS_WIDE`.
It does not modify either downstream repository or run their complete pipelines. It is tailored to these two
vendor layouts, rather than a general repository scanner.

## Result comparison and legacy accuracy checks

```sh
uv run python scripts/compare_benchmarks.py \
  --before data/benchmarks/baseline_eager_small.json \
  --after data/benchmarks/current_eager_small.json \
  --output data/benchmarks/eager_small_comparison.csv
```

`compare_benchmarks.py` requires identical target hash, steps, repetitions, source, PyTorch, device and case
lists. It matches side/batch/task/mode and exports step time, final error and threshold indices. It accepts
**only the `benchmark_fitting.py` result schema**; anatomy/layer benchmarks already store their comparisons
inside their own JSON. Inspect geometry agreement and shared-machine timing variation before claiming a speedup.

`compare_mano_layers.py` needs a separate environment able to import legacy chumpy, SciPy and the third-party
libraries, plus the licensed official `assets/mano/webuser` code. This is not part of the modern core setup.
Its source folder contains `manopth`, `smplx` and `manotorch_upstream` clones; use the pinned commits above.

```sh
python scripts/compare_mano_layers.py --thirdparty data/benchmarks \
  --mano-assets-root assets/mano --samples 32
```

The script adapts the official Python 2 syntax in memory and compares both hands in full axis-angle,
15-component PCA and rotation-matrix settings, float32 and float64. It prints largest vertex/16-joint errors
in mm. It excludes fingertips and is an accuracy check, not the steady-state benchmark used in README.

`test_compatibility.ipynb` is a separate interactive legacy check with manopth and Omid's MANO. Its first
Markdown cell lists pinned repositories and dependencies. Clone them into root `thirdparty/`; use a kernel
with **`scripts/` as its working directory**, because the cells use `../assets/mano` and insert the parent
folder into `sys.path`. The notebook needs chumpy and manopth's bundled loader on the import path. It is
not collected by pytest. Prefer the benchmark scripts for scripted reproducibility.

All Python entry points expose `--help`. JSON/CSV/GIF outputs replace existing files at the selected paths.
Shared-machine timing varies; retain raw groups and record environment/source hashes for comparisons.

## Original registration accuracy, stability, inference and fitting

`benchmark_registrations.py` reads the official `mano_poses_v1_0` PKLs directly. Every file, including
`l_mirrored`, is a right-model fit with absolute 48-value axis-angle pose, per-registration betas and scanner
translation. The layer uses `use_pca=False, flat_hand_mean=True, center_idx=None` and `transl=trans`.
The original `v` and `J_transformed` are independent targets; this does not reuse `mano_targets.npz` generated
by an archived implementation. The 16 target joints are compared in MANO order using
`[0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3]`, not `out.joints[:, :16]`.

All runs audit fields, finite values, right-model/full-pose metadata, R.npy sorted-file correspondence and model/data
hashes. `--dataset` points directly to `data/MANO_Poses/mano_poses_v1_0`. Choose separate output folders for
stage-specific runs: outputs replace existing files, and each `report.json` describes that invocation only.

```sh
uv run python scripts/benchmark_registrations.py --stages audit accuracy stability \
  --output-dir data/benchmarks/mano_registrations/validation
uv run python scripts/benchmark_registrations.py --stages runtime --modes eager compile \
  --output-dir data/benchmarks/mano_registrations/runtime
uv run python scripts/benchmark_registrations.py --stages fitting --fit-samples 128 \
  --fit-batches 32 128 --fit-steps 100 300 --fit-groups 3 \
  --output-dir data/benchmarks/mano_registrations/fitting
```

- Accuracy: all registrations, CPU/CUDA and float32/float64, vertex/16-joint Euclidean errors in mm,
  per-hand CSVs and worst-case target/prediction NPZs. Default acceptance is 0.001 mm maximum error.
  Float64 buffers retain the initial float32 model rounding; no machine-precision match to original arrays is assumed.
- Stability: all float32 forward/backward outputs must be finite. A deterministic representative/extreme set checks
  batching, permutation, shared betas, joints-only outputs/gradients and CPU/CUDA agreement. Float64 directional
  finite differences and model second derivatives cover real poses, zero, Taylor boundaries and near-pi rotations.
  An independent NumPy Rodrigues/sequential FK/homogeneous LBS reference verifies original right targets and both
  left shapedirs policies. L.npy is compared as rotation matrices because equivalent axis-angle vectors can differ.
  Left-hand mirror residuals are reported separately from agreement with the left model; there are no independent
  original left scanner targets in this archive. Accuracy/stability failures stop before later stages.
- Runtime: default batches 1/8/32/128/512/1024/1554, full mesh and joints-only, float32 `inference_mode`, eager by
  default; `--modes eager compile` adds optional compiled prediction. All batch members are independent records.
  Ten warmups and six rotating groups of 30 calls precede medians, throughput and CUDA-event spans. Startup is
  the first compiled forward plus synchronization (disk cache dependent), excluding numerical comparison.
  CUDA event spans include idle time between host launches, not just summed kernel durations. Loading, transfer
  and construction are excluded. Extra allocated memory excludes already live workspaces; reserved memory can
  include other methods and prior cases in this process and is not a clean per-backend footprint comparison.
- Fitting: eager only, vertices and 16 joints, weights 0 and 1e-4. A subject-stratified bank includes the most extreme
  shape/articulation, with fixed perturbations of 0.05 rad pose, 0.1 shape and 0.005 m translation (seed 20261008).
  Adam uses pose/shape/translation learning rates 0.01/0.02/0.001, `foreach=False`. Per-hand coordinate MSE and
  per-hand mean anatomy penalty are summed across the batch so batching does not rescale gradients relative to
  Adam epsilon. Each timed group resets parameters and zeroes warmed Adam state. The same selected bank is
  processed in shards; reported total time sums median shard durations, rather than timing just one batch.
  Curves, final mesh errors and actual first-threshold timestamps come from a separate synchronized diagnostic
  replay, whose time includes metric/host overhead and must not be substituted for metric-free fitting latency.
  Success means reaching 0.1 mm vertex or 1 mm joint RMSE at any recorded update; final errors are also reported.
  Joint fitting does not uniquely recover pose/shape, and a prior may trade geometric error for its penalty.

`--devices cpu`, `--runtime-devices cpu` and `--fit-device cpu` select CPU independently. Other controls are
`--threads`, `--chunk`, `--groups`, `--repeats`, `--derivative-samples`, `--fit-samples`, `--fit-batches`,
`--fit-steps`, `--fit-groups` and `--anatomy-weights`. The default stages include fitting, which can be lengthy.
Batch 1 processes each selected hand separately; use an eight-hand smoke bank before running a large bank:

```sh
uv run python scripts/benchmark_registrations.py --stages fitting --fit-samples 8 \
  --fit-batches 1 8 --fit-steps 20 --fit-groups 1 \
  --output-dir data/benchmarks/mano_registrations/fitting_smoke
```

For complete registration coverage in fitting, set `--fit-samples 1554 --fit-batches 128`. Short final shards are
kept, not padded with duplicate hands. Runtime batches larger than the dataset are rejected.

Render QC with the visualization extra after accuracy validation:

```sh
uv run --extra vis python scripts/render_registration_qc.py \
  --accuracy-csv data/benchmarks/mano_registrations/validation/accuracy_cuda_float32.csv \
  --output-dir data/qc/MANO_Poses/right
```

This writes `registration_qc.png` and `.json`, selecting the measured worst file, extreme shape/articulation and
representative registrations (`--samples 6` by default). Each row has target, reconstruction and a shared-scale
Euclidean vertex-error heatmap, with opaque surfaces and matched cameras. Errors are computed in the original
scanner frame; wrist centering and inverse global rotation affect display only. If `--accuracy-csv` is absent,
selection uses extremes/representatives without claiming a worst case. VTK/OpenGL off-screen support is required.

Summarize the complete accuracy/stability run independently of the preview selection:

```sh
uv run --with reportlab --with matplotlib python scripts/summarize_registration_qc.py \
  --results-dir data/benchmarks/mano_registrations/validation \
  --fitting-dir data/benchmarks/mano_registrations/fitting \
  --output-dir data/benchmarks/mano_registrations/right_qc_audit
```

This optional publishing script writes `qc_summary.pdf`, `qc_statistics.json` and `qc_metrics.csv`.
It requires ReportLab and matplotlib, without adding either to the core package dependencies. The PDF shows
dataset coverage, all device/dtype error distributions, worst registrations, stability/reference checks and
the existing six-hand preview. Optional `--fitting-dir` adds the completed batched fitting results and methodology.
Units are explicit: PDF overview values use micrometres; CSV/JSON use mm.
All accuracy CSVs must contain exactly the manifest's unique filenames with finite nonnegative metrics.
JSON provenance records the source report, manifest, accuracy CSVs and summary-script hashes.

The proposed full atlas uses A3 portrait alphabetical six-hand pages, three columns, large models,
native PDF labels of at least 9 pt and one global error colour scale;
subject bookmarks and an error-ranked index help navigation. For 1,554 registrations this is 259 grid pages,
excluding a cover/statistics section. CSV `proposed_atlas_grid_page` and `proposed_atlas_row` identify these
planned locations. The statistics builder does **not** render the atlas; `render_registration_qc.py` produces
a preview and `render_registration_atlas.py` below produces the complete atlas.
Prefer a multi-page PDF for complete coverage and export individual PNG pages only when needed. Synthetic
sequence frames need a separate temporal QC design because they lack original registration mesh targets.

Render the full atlas using CUDA MANO inference and an existing GPU OpenGL context:

```sh
uv run --extra vis --with reportlab --with matplotlib python scripts/render_registration_atlas.py \
  --device cuda --require-gpu-render --output-dir data/qc/MANO_Poses/right \
  --pdf-path data/qc/MANO_Poses/registration_atlas_right.pdf
```

`--device` controls PyTorch inference; VTK/OpenGL selects its rendering device independently. The manifest
records its actual vendor/renderer/version; `--require-gpu-render` rejects software OpenGL. A working VTK/OpenGL
context is required. Opaque surfaces, light kits and camera settings are reused consistently across pages.
The atlas has two cover/index pages followed by alphabetical six-hand grids: 261 PDF pages for 1,554 hands.
Native PDF text keeps 9-12 pt captions legible. Subject bookmarks and the top-20 error index provide clickable
navigation. The shared colour maximum is derived once from all vertices, never rescaled per page.

The root `registration_atlas_right.pdf` and side-folder `registration_atlas_index.csv` / `registration_atlas.json` record full coverage,
actual PDF/grid page and row, errors and source/model/PDF hashes. PNG exports under `atlas_pages/` default to
first/middle/last grid pages plus the worst error and extreme shape/articulation pages. Use `--export-pages 1 130 259`
to choose grid pages, or `--export-pages` with no numbers to skip exports; exports require Poppler's `pdftoppm`.
`--max-pages 1 --output-dir data/qc/MANO_Poses/atlas_smoke` makes a separate incomplete layout/lighting smoke PDF.
The full atlas is built through a `.partial.pdf` and replaces the final PDF only after successful completion.

The default inference chunk 128 matches the complete CUDA float32 accuracy CSV. Supply the matching
`--accuracy-csv` when changing inference device/chunk; coverage and metric agreement are checked. Maximum
scanner-frame vertex/joint reconstruction error must remain below 0.001 mm. Dataset files remain local.
After rendering, pass `--atlas-manifest data/qc/MANO_Poses/right/registration_atlas.json` to the statistics builder
to include completed-atlas provenance and actual PDF page/row columns in `qc_metrics.csv`.

`--full-batch-validation-dir` adds a completed accuracy/stability run with `--chunk 1554`, confirming single-batch
coverage separately from the chunked audit. Repeat `--fitting-dir` to include multiple completed banks;
`--timing-note` records resource contention or other timing qualifications. For a full independent fitting bank:

```sh
uv run python scripts/benchmark_registrations.py --stages accuracy stability --devices cuda --chunk 1554 \
  --output-dir data/benchmarks/mano_registrations/full_batch_validation
uv run python scripts/benchmark_registrations.py --stages fitting --fit-samples 1554 --fit-batches 1554 \
  --fit-steps 100 300 --fit-groups 3 --anatomy-weights 0 \
  --output-dir data/benchmarks/mano_registrations/full_batch_fitting
```

Run performance measurements when GPU resources are available. Chunking is not a MANO batch-size limit.
Calibrate anatomy weights on a small quality-controlled bank before expanding prior-based fitting to all hands.
The 2026-10-08 run's full-batch accuracy/finite-gradient checks passed; full-batch fitting was deferred after GPU
resources were reported occupied. Its existing shared-machine timing figures remain provisional.

Run asset-independent bookkeeping regressions with `uv run pytest scripts/test_benchmark_scripts.py`.
They cover the final-update threshold, indexed CUDA synchronization, MANO joint mapping and independent Rodrigues
known rotations. Public CI includes them alongside the core suite. Full licensed-dataset and CUDA runs remain local.

## Self-contained right/left registration QC

The original dataset contains right scans and left scans already mirrored into MANO_RIGHT. Its official
`L.npy` supplies mirrored articulation for MANO_LEFT. The bundle preparer validates every joint rotation as
`S R S`, also mirrors global rotation, translation, mesh and joints, and retains betas. Left inference uses
`fix_left_shapedirs=True`; raw stored left model arrays are not pre-corrected. Derived left targets are not
new independent scans. Implementation error and small model asset mirror differences are reported separately.

```sh
uv run python scripts/prepare_registration_bundle.py --device cuda --output-dir data/qc/MANO_Poses
uv run --extra vis --with reportlab --with matplotlib python scripts/render_registration_atlas.py \
  --pose-archive data/qc/MANO_Poses/right/poses.npz --output-dir data/qc/MANO_Poses/right \
  --pdf-path data/qc/MANO_Poses/registration_atlas_right.pdf --require-gpu-render
uv run --extra vis --with reportlab --with matplotlib python scripts/render_registration_atlas.py \
  --pose-archive data/qc/MANO_Poses/left/poses.npz --output-dir data/qc/MANO_Poses/left \
  --pdf-path data/qc/MANO_Poses/registration_atlas_left.pdf --require-gpu-render
uv run --with reportlab --with matplotlib --with pypdf python scripts/summarize_registration_mirror.py \
  --bundle-dir data/qc/MANO_Poses
python3 scripts/verify_registration_bundle.py data/qc/MANO_Poses --write-manifest
```

Use a fresh folder for a new preparation run: archives/models are replaced, invalidating earlier atlas hashes.
Preparation audits every sample against independent NumPy FK/LBS on CPU and the selected device, in
float32/float64; checks finite first derivatives, full batch vs chunks, permutation, joints-only and 16 single
samples per side. It performs no timing/fitting benchmark.

Both `right/poses.npz` and `left/poses.npz` contain full 48-axis-angle poses, shape, translation, targets,
MANO 16 joints, faces, cached predictions, 21 reconstructed joints and independent references. Each side
folder also has metrics, statistics, atlas manifest/page index and selected PNGs. All PDFs are at root:
`registration_atlas_left.pdf`, `registration_atlas_right.pdf`, and **one combined** `qc_summary.pdf`.
The combined report has paired images, distributions, numerical reference and mirror residuals, stability
and historical right fitting; it performs no new fitting. Historical measurements are retained in
`right/audit_statistics.json`; if absent, the report explicitly states no historical fitting is included.

`--pdf-path` separates the PDF destination from `--output-dir`'s side-specific CSV/JSON/PNG outputs.
Atlas paths are relative to the manifest folder; its `path_base` field identifies this convention.
The legacy `summarize_registration_qc.py` builds a right-only audit; use a separate output folder for it,
then retain its statistics as `right/audit_statistics.json` when building a combined bundle.

[The bundle README template](registration_bundle_README.md) documents arrays and portable commands.
Models, source snapshot, official README/aggregate arrays and small tools support operation after moving
this folder. Cached rendering needs no Torch or original dataset. Fresh inference uses bundled source/models.
Python packages, rendering drivers and Poppler remain runtime dependencies. Verify the folder inventory
with `verify_registration_bundle.py FOLDER`.

Each atlas has 261 pages. Both hands use upright display rotations; the left display basis adds a proper
180-degree rotation about display z, without changing poses, winding or measurements. The summary uses
the new atlas panels directly. The left error column is a **mirror residual**: cached independent-reference
agreement must remain below 0.001 mm; a separate 0.05 mm mirror guard detects transformation/asset mistakes.
The measured corrected asset residual is approximately 0.0106 mm. Use independent left references for exact
parameter recovery and mirrored targets to assess symmetry/transfer; identify the chosen target explicitly.
