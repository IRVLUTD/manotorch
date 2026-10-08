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

## Topology cache

`UpSampleLayer` is a separate triangle-subdivision layer, unused by the measured MANO forward/fitting paths.
Its original implementation rebuilt a Python edge map on CPU for every mesh on every call, including repeated
identical batch topologies. The current layer stores a bounded, nonpersistent topology cache on the tensor device.
A prepared snapshot runs midpoint interpolation with native tensor operations and no warm CPU topology transfers.
See [the API and invalidation rules](features.md#mesh-subdivision) and
[the benchmark command](../scripts/README.md#benchmark_upsamplepy-repeated-mesh-subdivision).

Measured on 2026-10-05: RTX 4090, PyTorch 2.11.0+cu126, four CPU threads, right open-wrist MANO mesh
(778 vertices, 1,538 triangles), one faces tensor expanded across each batch. Values are median synchronized
wall-clock milliseconds per subdivision over five rotating groups. The uncached baseline is `c936b59`.
Timings include interpolation and fresh output faces, excluding MANO evaluation. Cached methods use 30 calls
per block versus three for the much slower baseline. Geometry and output face order match exactly.

| Device | Batch | Uncached (ms) | Automatic cache (ms) | Prepared snapshot (ms) |
| --- | --- | --- | --- | --- |
| CPU | 1 | 27.491 | 0.437 | 0.432 |
| CPU | 32 | 876.458 | 2.537 | 2.541 |
| CPU | 128 | 3515.005 | 9.455 | 9.438 |
| CUDA | 1 | 27.628 | 0.228 | 0.223 |
| CUDA | 32 | 874.356 | 0.239 | 0.227 |
| CUDA | 128 | 3539.932 | 0.241 | 0.226 |

Cold shared-topology preparation takes 20.7-22.9 ms here and is excluded from warm figures. The very large
batch speedups chiefly remove the baseline's repeated Python traversal, rather than accelerating MANO kernels.
One-off or continually changing topologies still pay preparation costs. Raw groups, device/source metadata and
hashes are saved locally in `data/benchmarks/upsample_cache.json` and are not distributed with the package.

## Optional kernel feasibility

The standalone [kernel experiment](../scripts/benchmark_kernels.py) evaluates two contiguous CUDA float32
Triton forward prototypes; neither is imported by manotorch. Skinning fuses the final per-vertex affine transform
and keeps the existing native PyTorch backward. Rotation fuses the protected squared-angle/Taylor axis-angle
conversion, but **has no backward and is used only under `no_grad`**. No production backend or C++/CUDA
extension was added. Separate CUDA profiling confirms rotation launches decrease from 28 to one and final
skinning launches from three to one. Commands, requirements and excluded compatibility claims are in
[the scripts guide](../scripts/README.md#benchmark_kernelspy-optional-kernel-feasibility-experiment).

Two complete runs used RTX 4090 / PyTorch 2.11.0+cu126 / Triton 3.6.0, four CPU threads, right full meshes,
batches 1/128/1024 and six rotating groups of 20 calls. Compile and Triton startup, construction and validation
are excluded. Synchronized wall time includes Python launch overhead. Timings varied markedly between some
blocks on this shared machine; compare methods within each workload rather than comparing batch latencies.

Warmed isolated forward medians from the second run:

| Batch | Operation | Eager (ms) | Compile (ms) | Triton (ms) |
| --- | --- | --- | --- | --- |
| 1 | Rotation | 1.046 | 0.390 | 0.084 |
| 128 | Rotation | 1.049 | 0.412 | 0.083 |
| 1024 | Rotation | 0.301 | 0.148 | 0.028 |
| 1 | Final skinning transform | 0.289 | 0.154 | 0.167 |
| 128 | Final skinning transform | 0.286 | 0.178 | 0.169 |
| 1024 | Final skinning transform | 0.088 | 0.069 | 0.068 |

Rotation's isolated speedup is much larger than the full-model improvement. Complete MANO **no-grad inference**:

| Batch | Run 1 eager / rotation Triton (ms) | Run 2 eager / rotation Triton (ms) | Latency reduction across runs |
| --- | --- | --- | --- |
| 1 | 3.720 / 2.787 | 4.010 / 3.045 | 24.1-25.1% |
| 128 | 1.303 / 0.971 | 3.968 / 3.009 | 24.2-25.5% |
| 1024 | 1.302 / 0.973 | 1.307 / 0.973 | 25.3-25.5% |

The maximum full-model vertex discrepancy from rotation fusion was `4.47e-5 mm`. Forward probes cover zero,
near-zero, both Taylor branch boundaries and larger angles. Skinning geometry and pose/shape gradients matched
exactly. Complete 100-step skinning fits, with and without anatomy loss, also produced exactly equal final
RMSE/prior/objective in all measured cases. They use the frozen bank and Adam setup from the anatomy-fitting
benchmark; rotation fusion is never used in fitting.

Skinning did **not** produce a consistent complete-fitting speedup. In the stable batch-128 anatomy case,
run 1 took 0.819 s eager versus 0.822 s Triton and run 2 took 0.820 s versus 0.824 s (about 0.5% slower).
Autograd-enabled model forward improved about 1.4-1.6% at batches 1/128 but regressed about 2.3-2.4% at 1024.
Other fitting groups crossed major host-load transitions: for example run-1 batch-1 anatomy groups spanned
1.426-2.337 s eager and 1.426-2.381 s Triton. A 19% difference between those medians is inconclusive;
raw ranges and the repeat run prevent treating it as a reproducible kernel regression or improvement.

The recommendation is to retain the native default. A narrowly scoped optional rotation-inference backend is
worth a follow-up if inference is a real downstream hot path; training needs a separately validated backward.
Prioritize complete fitting and existing `joints_only` support over the final skinning forward kernel. C++ alone
would not fuse the GPU operations; a fused CUDA implementation could be evaluated later, without assuming it
will beat the Triton prototype.

Production difficulty lies in preserving dtype/layout fallbacks, zero-angle second derivatives, transform rules
and compiled behavior. PyTorch documents the required transformable methods and batching/JVP support for
[custom autograd functions](https://docs.pytorch.org/docs/2.11/notes/extending.func.html). C++/CUDA adds compiler,
CUDA architecture and binary-distribution work. The current [official C++ tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
uses a stable-ABI target requiring PyTorch 2.10; this package's minimum is 2.0.1, so that route does not cover the
full supported range. Recent tutorial APIs must not be assumed available in the minimum environment.

Rough planning estimates for a developer familiar with this code: a narrow optional Triton inference path and
fallback tests, 3-5 working days; a training path preserving higher derivatives/torch.func and version coverage,
1-3 weeks; a distributed C++/CUDA backend, 2-4 weeks. These are engineering estimates, not measured runtimes
or commitments; additional devices and environments expand the scope.

Raw groups, numerical checks, targets/source hashes and environment metadata are retained locally in
`data/benchmarks/kernel_feasibility.json` and `kernel_feasibility_repeat.json`. Profiling and compilation probes
are separate from timing. The experimental script makes no AMP, vmap/JVP, second-derivative, compiled-full-model
or cross-device compatibility claim. Its kernels remain outside the public runtime API.

## Original MANO registration validation (2026-10-08)

The [paired QC bundle](../scripts/README.md#self-contained-rightleft-registration-qc) extends this audit to
1,554 derived left-hand poses, including full global rotation and translation. Targets are geometric mirrors
of the original right-model registrations, not independent left-scan fits. With `fix_left_shapedirs=True`,
CUDA float32 left reconstruction agrees with independent left NumPy FK/LBS to 0.000105672 mm vertex max and
0.000090054 mm joint max. Independent corrected-left vs mirrored-right asset vertex residual is 0.010575863 mm
in the full scanner frame; without the shape correction it reaches 50.449993 mm. These are Euclidean distances,
unlike the earlier canonical-frame maximum-coordinate residual. All samples pass finite-gradient checks;
B=1,554, chunks, permutation and joints-only are consistent within 0.001 mm. A full 261-page left atlas and
portable archives/models/source/tools live locally under `data/qc/MANO_Poses`. No new fitting/timing claim
is made by this extension.

The new [registration workflow](../scripts/README.md#original-registration-accuracy-stability-inference-and-fitting)
uses all 1,554 original PKLs from `mano_poses_v1_0`: 31 subjects, 895 right hands and 659 left hands already
mirrored into the right model. These original scanner-frame vertex/16-joint targets are independent of our
generated frozen targets. Every registration is reconstructed with its stored 48 absolute axis-angle values,
ten betas and translation, `use_pca=False, flat_hand_mean=True, center_idx=None`. Numerical helpers for the
independent NumPy Rodrigues/sequential-FK/homogeneous-LBS reference do not call manotorch's geometry routines.

RTX 4090, PyTorch 2.11.0+cu126, NumPy 2.5.3, four CPU threads. Maximum Euclidean errors in **mm**, all hands:

| Device / dtype | Maximum vertex error | Maximum 16-joint error |
| --- | --- | --- |
| CPU / float32 | 0.00007823 | 0.00006005 |
| CUDA / float32 | 0.00012732 | 0.00010156 |
| CPU / float64 | 0.00001029 | 0.000001831 |
| CUDA / float64 | 0.00001029 | 0.000001831 |

All pass the 0.001 mm threshold. Float64 still uses model buffers rounded to float32 at construction.
The independent original-array NumPy reference matches stored vertices/joints to a maximum **coordinate**
error of 1.94e-13 mm. CPU/CUDA float32 forward/backward are finite on every registration; representative
batch/single, permutation, shared-shape, joints-only gradient/output and cross-device checks pass. Model
directional finite differences and finite second derivatives cover real poses, zero, Taylor thresholds and near pi.

A separate `--chunk 1554 --devices cuda` run sends the entire dataset through one batch for accuracy and
finite forward/backward checks. Maximum float32 vertex/joint errors are 0.00012794/0.00011515 mm and pass.
Chunking the main audit at 128 facilitates diagnostics; it is not a MANOLayer batch-size restriction.

The left model is verified separately against its own NumPy reference under both shapedirs policies. Correcting
left shapedirs reduces maximum mirrored-coordinate residual from 44.796 mm to 0.008993 mm across this bank.
That residual reflects left/right asset differences and is not the right-model reconstruction error above.
The archive does not supply independent original left scanner meshes.

### Registration inference

All members of each runtime batch are independent registrations, including `B=1554`. True `inference_mode`,
ten warmups, six rotating groups of 30 calls; fullgraph reduce-overhead compilation; loading, construction,
transfers and first-call compilation excluded from warm timings. Selected synchronized wall-clock medians:

| Batch | Full mesh eager / compile (ms) | Joints-only eager / compile (ms) |
| --- | --- | --- |
| 1 | 3.637 / 0.252 | 3.689 / 0.248 |
| 128 | 3.771 / 0.252 | 3.937 / 0.246 |
| 1554 | 3.730 / 1.493 | 3.887 / 0.253 |

First compiled inference plus synchronization takes 5.66-9.94 s per case here, depending on the disk cache.
Prediction numerics are checked before timing; this does not establish compiled fitting/high-order compatibility.
These are shared-machine measurements, with source hashes and every raw timing group retained. They are not
directly comparable to the earlier different-input, autograd-enabled README layer workload. GPU resources were
subsequently reported occupied; repeat performance runs when idle before making optimization decisions.
The latest scripts also skip CUDA-graph markers for eager/CPU execution; those corrected GPU timing runs are pending.

### Registration fitting and quality

A fixed bank of 128 independent hands includes all subjects and extreme shape/articulation, with seed 20261008.
Perturb pose by 0.05 rad, shape by 0.1 and translation by 0.005 m. Eager float32 Adam optimizes all three with
learning rates 0.01/0.02/0.001 and `foreach=False`. The sum of independent per-hand coordinate-MSE/prior losses
keeps gradient scale relative to Adam epsilon independent of batching. Shards of 32 and one batch of 128 process
the same bank. Each case runs 100/300 updates and three reset timing groups; curves and threshold timestamps
come from a separate diagnostic replay. This tests near-target convergence, not image-based initialization.

Selected 300-update **mean per-hand** errors in mm, batch 128:

| Observation | Anatomy weight | Observation RMSE | Mesh RMSE | Ever reached threshold |
| --- | --- | --- | --- | --- |
| Vertices | 0 | 0.07157 | 0.07157 | 81.25% at 0.1 mm |
| 16 joints | 0 | 0.03739 | 0.94249 | 100% at 1 mm |
| Vertices | 1e-4 | 3.23511 | 3.23511 | 0% at 0.1 mm |
| 16 joints | 1e-4 | 2.16763 | 6.62401 | 3.125% at 1 mm |

Batch 32 gives close quality agreement (without prior: 0.07224 mm vertex / 0.03739 mm joint RMSE); finite
precision and Adam near convergence can produce small differences. The joint-only observations leave mesh
and latent pose/shape underdetermined. Weight 1e-4 lowers mean anatomy penalty from about 0.4601 to 0.0103 rad
for vertex fitting and 0.4660 to 0.00654 rad for joint fitting, while substantially worsening geometry. It needs
dataset-specific calibration; the prior benchmark's default weight is not a fitting recommendation.

Batch-128 complete times for 300 updates are 3.319/4.884 s for vertex fitting without/with prior and
3.216/5.089 s for joint fitting. These timings are **provisional**: group/shard ranges show major shared-host
variation, and resource contention was reported. They do not support a controlled prior-overhead or batch-speedup
claim. Peak extra allocation is at most 17.13 MiB across the bank's cases, excluding already live buffers.
Full `B=1554` fitting is deferred until GPU resources are available; its command is in the script guide.

Local raw outputs are in `data/benchmarks/mano_registrations/2026-10-08`, `runtime_2026-10-08`,
`fitting_2026-10-08` and `full_batch_validation_2026-10-08`. The solid-surface six-hand preview and full-dataset
statistics PDF/CSV/JSON are under `data/qc/MANO_Poses`. The complete A3 atlas contains 259 six-hand grids plus
cover/index (261 PDF pages), rendered on RTX 4090 OpenGL with one global error scale and native captions.
Its companion CSV maps every filename to actual PDF/grid page and row; the manifest records source/PDF hashes.
All samples are covered, with clickable subject bookmarks and a top-20 worst-error index. First/middle/last,
worst-error and extreme-shape/articulation grid pages are also exported as PNGs under `atlas_pages/`.
Synthetic sequence frames need separate temporal QC because original registration meshes are unavailable.

### Combined right/left QC report layout

The paired bundle uses matching `left/` and `right/` directories for poses, metrics, atlas page indices and
selected PNGs. All PDFs are at `data/qc/MANO_Poses/`: `registration_atlas_left.pdf`,
`registration_atlas_right.pdf`, and one combined `qc_summary.pdf`. The four-page report includes eight
device/dtype accuracy cases, both-side distributions/stability, paired first/worst-mirror/extreme-shape
images and the existing 16-case right fitting table. Historical audit details remain in
`right/audit_statistics.json`; no new fitting/timing was run. The right atlas retains its original images; the left atlas uses a side-specific proper upright rotation,
with a 180-degree rotation about display z to align finger direction while preserving handedness. Summary
previews use the atlas panels directly, with no additional roll. Paths in migrated atlas manifests resolve
relative to each side directory, while original paths and historical source hashes remain recorded.
