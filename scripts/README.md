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
| `simple_app.py` | View mirrored hands with anatomical axes or 32 anchors | Interactive window or orbiting GIF |
| `simple_compose.py` | Compose both hands from the same anatomical Euler angles | Interactive window or orbiting GIF |
| `simple_anatomy_loss.py` | Correct an implausible index-finger pose using the anatomy prior | Interactive window or optimization GIF |
| `benchmark_fitting.py` | Measure MANO model runtime and complete fitting, without anatomy loss | Frozen targets, manifest and result JSON |
| `benchmark_fitting_anatomy.py` | Compare complete eager fitting with/without anatomy loss across three manotorch revisions | Interleaved timing and convergence JSON |
| `benchmark_layers.py` | Compare eager layer runtime with upstream, manopth and smplx | Timing groups, numerical errors and source hashes in JSON |
| `benchmark_anatomy.py` | Isolate the anatomy-loss forward/backward cost | Baseline/current timing JSON |
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
  CPU so the pose selection does not depend on which device renders it.
- `simple_compose.py`: defaults to expressing the composed articulation as 45 PCA coefficients;
  `--no-pca` feeds the full axis-angle pose directly. The script prints whether the resulting poses are mirrored.
- `simple_anatomy_loss.py`: defaults to 1000 Adam updates, learning rate 0.01 and a frame every 10 updates;
  it optimizes articulation with anatomy loss alone. It is **not** an observation-fitting benchmark. Keep
  `--iters >= 5` because its learning-rate scheduler uses `iters // 5` as the step size. The surface is opaque.
- All three automatically choose CUDA, then MPS, then CPU, and accept `--mano-assets-root`.
  The GIF colors are red/twist, green/spread and blue/bend. GIF playback does not measure runtime.

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
Threshold time is an estimate from the curve index and mean step time. Extra allocated memory is not total
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
5 groups of 30 calls. Use `--device cpu` for CPU. It has no MANO model, FK, optimizer or fitting targets;
its speedup must not be reported as an end-to-end fitting speedup. It uses baseline-then-current ordering,
unlike the interleaved comparisons above.

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
