# Reproducible MANO fitting benchmark

The benchmark compares an archived implementation and the current code against identical frozen targets and
initial parameters. It separates model runtime, complete fitting steps and convergence. The optimizer remains eager
in both modes; `compile` compiles the model prediction with `fullgraph=True, mode="reduce-overhead"`.

The [scripts usage guide](../scripts/README.md) lists each entry point, dependencies, inputs/outputs and examples.

## Data and conventions

- MANO training poses: 128 deterministically selected poses per hand from the 1554-row R/L arrays under
  `data/MANO_Poses/mano_poses_v1_0`. Global orientation, shape, translation and initialization perturbations are
  generated with seed `20261005`. These are controlled fitting targets, not independent measured joint observations.
- Targets must first be generated with the archived `c936b59` source. All subsequent runs read the same NPZ.
  At batch 1024, the fixed 128-sample bank is repeated to measure throughput; it is not 1024 independent hands.
- Real validation: 32 valid frames per available hand from the UHAS ARCTIC and HO-Cap exports. The selected HO-Cap
  sequence has only valid left hands. The 16 MANO joints are used, since the exports use different fingertip vertices.
  Exported metadata specifies metre units, absolute axis-angle articulation, uncorrected left shapedirs and no centering.
- `benchmark_*_samples.json` stores source hashes and selected frame/array indices. Actual data, generated targets
  and raw results remain under git-ignored `data/benchmarks`; no model or pose dataset is distributed with the package.

## Commands

Run from the repository root with the licensed MANO models installed:

```sh
mkdir -p data/benchmarks/baseline_source
git archive c936b59 | tar -x -C data/benchmarks/baseline_source
uv run python scripts/benchmark_fitting.py \
  --implementation-root data/benchmarks/baseline_source \
  --output data/benchmarks/baseline.json
uv run python scripts/benchmark_fitting.py --output data/benchmarks/current.json
```

Defaults cover both sides, batches 1/32/128/1024, joints/full, joints-only and vertices fitting, eager and compiled.
The first run creates `mano_targets.npz`; do not remove or regenerate it between implementations. Its SHA256 is
recorded in every result. To shorten a diagnostic run, use `--batches 1 128 --steps 100 --repeats 20 --sides right`.
Use `--device cpu --modes eager` for CPU measurements. Change `--mano-assets-root` for another model collection.

Prepare the real observations without modifying the UHAS sources:

```sh
uv run python scripts/prepare_fitting_data.py --root /path/to/UHAS_HumanData/data/hand_poses
uv run python scripts/benchmark_fitting.py --source real \
  --targets data/benchmarks/arctic_targets.npz --tasks joints joints_only --batches 32 \
  --output data/benchmarks/arctic_fitting.json
uv run python scripts/benchmark_fitting.py --source real --sides left \
  --targets data/benchmarks/hocap_targets.npz --tasks joints joints_only --batches 32 \
  --output data/benchmarks/hocap_fitting.json
```

For the anatomy loss and downstream numerical audit:

```sh
uv run python scripts/benchmark_anatomy.py
uv run --with scipy --with deprecation python scripts/check_downstream.py --root /path/to/Deepreach_AI
```

SciPy and deprecation are only needed by the old FoundationEgo vendor during that audit. The current MANO loader needs
neither SciPy nor chumpy. The audit does not replace downstream source or run the full annotation/stereo pipeline.

## Metrics and interpretation

The forward measurement keeps autograd enabled, as fitting does. Forward+backward clears gradients each iteration.
Five warmups precede each timed block. CUDA is synchronized at block boundaries; the measurement uses wall time and
therefore includes Python/kernel-launch overhead. Peak extra CUDA **allocated** memory is reported relative to memory
already live before a block; it is not total reserved device memory. Compile/CUDA-graph workspaces can already be live
in that baseline, so cross-mode peak-extra numbers must not be interpreted as total-memory comparisons.

Adam has learning rates 0.01 for pose, 0.02 for shape and 0.001 for translation; it runs 100 steps by default.
The error is root mean squared Euclidean point distance, in millimetres. Record initial/final error and the full curve.
`steps_to_threshold` uses 1 mm by default; `estimated_ms_to_threshold` multiplies it by the measured mean step time.
It is an estimate, not a separately instrumented first-passage timestamp. Shape/pose identifiability is not assumed;
the benchmark evaluates geometric error rather than exact recovery of every latent parameter.

`compile_seconds` measures the first differentiated prediction plus backward and synchronization. It depends on the
existing Inductor disk cache and is not an installation-independent cold-compilation guarantee. Each compiled case
resets Dynamo's in-process cache, avoiding its specialization limit across different shapes/tasks.

Shared-machine scheduling and GPU contention can produce large timing fluctuations. Repeat measurements and compare
ranges before claiming a small speedup or slowdown. Convergence comparisons use the exact same targets and initial
parameters, even when timings fluctuate. Consult the README performance tables and the methodology below
for measured results and limitations.

## Eager comparison with other MANO layers

`scripts/benchmark_layers.py` compares current eager, Claude's `c936b59`, upstream manotorch, manopth and
smplx MANO/MANOLayer with the same licensed model arrays. Obtain these pinned numerical sources:

```sh
git clone https://github.com/lixiny/manotorch.git data/benchmarks/manotorch_upstream
git -C data/benchmarks/manotorch_upstream checkout a2a70c591f91551078b7bb2af9b5d9f275b626e0
git clone https://github.com/hassony2/manopth.git data/benchmarks/manopth
git -C data/benchmarks/manopth checkout 4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b
git clone https://github.com/vchoutas/smplx.git data/benchmarks/smplx
git -C data/benchmarks/smplx checkout 1265df7ba545e8b00f72e7c557c766e15c71632f
uv run python scripts/benchmark_layers.py --groups 7 --repeats 20 \
  --output data/benchmarks/eager_layers.json
```

Prepare `baseline_source` with the archive command above. No upstream runtime numerical source is edited.
A construction-only replacement for legacy `ready_arguments` supplies the same model arrays through this fork's
restricted loader; smplx receives a `Struct` with those arrays. Deserialization, construction and validation are
outside timing. The shim does not claim compatibility of the upstream loaders themselves with modern dependencies.

Both hands and batches 1/128/1024 use seeded random full axis-angle poses and shapes, float32 and a flat mean;
all implementations return full meshes. `MANOLayer`'s matrix conversion is included. Unit normalization and
selection of the 16 common joints are included; five fingertips are excluded from numerical comparisons.
Forward keeps autograd on; backward differentiates mean squared vertices in metres for pose and betas, clearing
their gradients per iteration. It excludes translation optimization, Adam, anatomy loss and compilation.
Ten warmups precede each case; the implementation order rotates across groups and each group contains 20
synchronized wall-clock calls. The output stores every group's timing, medians, errors, source hashes and pins.
This measures latency of this workload, not convergence or inference under `no_grad`.

The optional `--before-geometry PATH` loads a local saved geometry module to compare the initial correctness
fix with the latest eager optimization. This snapshot is under ignored data and is not required for the
public upstream/current comparison. The review used `data/benchmarks/geometry_before_eager.py`.
README reports the right-hand table from this run; the complete two-hand data stays in the output JSON.
Across the six paired cases, current/upstream speedups are 1.70–2.19× forward and 1.48–2.10× forward+backward.
The largest cross-layer discrepancies are 7.46e-5 mm for vertices, 5.97e-5 mm for common joints and
4.08e-10 for pose/betas gradients. These are float32 comparisons and do not replace the independent
official-model accuracy check. Shared-machine scheduling changed substantially between right/left cases;
compare implementations within each case instead of attributing cross-case timing differences to hand side.

## Complete eager fitting with anatomy loss

After preparing the same frozen target NPZ and pinned sources above:

```sh
uv run --with deprecation python scripts/benchmark_fitting_anatomy.py \
  --batches 1 128 1024 --steps 100 --groups 6 --anatomy-weight 0.0001 \
  --output data/benchmarks/fitting_anatomy.json
```

This compares the current fork, Claude `c936b59` and upstream `a2a70c5`, each using its own `ManoLayer`,
`AxisLayerFK` and `AnatomyConstraintLossEE`. Upstream's module needs `deprecation`; it is not a core dependency.
Legacy array-loading and absolute-import adapters operate at construction only. Each native FK layer receives
the same current right-hand reference basis, and all use the same default limits, so differences in fingertip-based
template axes do not change the objective. Cached basis products in the archived implementation are aligned too.
Timed numerical source is not edited. The right hand avoids the upstream's different left-hand angle convention.
manopth and smplx do not expose this same anatomy chain and are not included in this table.

Full float32 meshes are evaluated from flat-mean 48-value axis-angle poses, while the observation term fits
the 16 common MANO joints (all fingertips excluded). The loss is coordinate MSE in metres plus `1e-4` times
the mean anatomy penalty in radians. The prior weight is a benchmark choice, not a tuned fitting policy.
Weight zero is measured separately and skips FK/anatomy evaluation; in those JSON rows `anatomy_rad=0`
means disabled, not a measured zero penalty. Pose, shape and translation use the shared initialization and
Adam settings described above. Batch 1024 repeats the 128-sample bank.

Each group fits 100 steps from the same initial parameters and zeroed Adam state. Ten warmup steps allocate
Adam buffers first; allocation/reset/warmup are outside timing. The order rotates across six groups, balancing
the three implementations. Synchronized wall time covers MANO, optional FK/Euler/prior, backward and Adam;
there is no per-step metric collection. Initial/final metrics and gradient agreement are evaluated outside timing.
The JSON stores each group's total milliseconds, median complete-fit and per-step times, geometry/prior metrics,
validation errors, and model/source/benchmark hashes. It uses its own result schema, not `compare_benchmarks.py`.

The README's table uses median **total seconds for 100 updates**, rather than extrapolating a standalone loss
speedup. These fits have 16-joint MSE and a prior, whereas the eager layer table differentiates vertex MSE without
Adam; their timings are different workloads. Shared-machine load varied within some groups, so retain the raw
timing ranges and compare within each case. The anatomy penalty can decrease while observation RMSE increases.
