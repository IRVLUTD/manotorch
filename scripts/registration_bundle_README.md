# Self-contained MANO_Poses QC bundle

This folder contains 1,554 right-model registrations and 1,554 paired derived left-hand poses.
The original training registrations comprise 895 right scans and 659 left scans already mirrored into
MANO_RIGHT. The left set is their geometric reflection, **not additional independent left scan data**.
Both sets retain the full global orientation and translation in the scanner coordinate frame.

All data/model inputs and a manotorch source snapshot are included. Analysis, fresh inference and cached
re-rendering need no original dataset directory or repository checkout. Python packages and the rendering
driver are external runtime dependencies; PNG exports require Poppler's `pdftoppm`.
The MANO license still applies to the models and derived data. Keep this folder local; do not redistribute it.

## PDFs and symmetric side folders

All PDF files are regular files at the bundle root:

- [Combined right/left QC summary](qc_summary.pdf): accuracy, stability, distributions, paired hand images,
  mirror residuals and the existing right-model fitting results. Both hands use side-specific proper upright
  display rotations; atlas data and measured errors are unchanged.
- [Left-hand image atlas](registration_atlas_left.pdf): 261 pages, all 1,554 derived left poses.
- [Right-hand image atlas](registration_atlas_right.pdf): 261 pages, original right-model registrations.

Hand images in each atlas start at PDF page 3; pages 1-2 are the cover/index.
Each side has an identical primary result layout:

```text
MANO_Poses/
  qc_summary.pdf
  registration_atlas_left.pdf
  registration_atlas_right.pdf
  qc_metrics.csv                 # both sides, all device/dtype cases, actual atlas pages
  qc_statistics.json             # combined report and provenance
  mirror_validation.csv/json     # original paired validation inputs
  left/
    poses.npz
    qc_metrics.csv
    qc_statistics.json
    registration_atlas.json
    registration_atlas_index.csv
    registration_atlas_progress.json
    atlas_pages/
  right/
    poses.npz
    qc_metrics.csv
    qc_statistics.json
    registration_atlas.json
    registration_atlas_index.csv
    registration_atlas_progress.json
    atlas_pages/
    audit_statistics.json        # historical right audit/fitting measurements
    audit_metrics.csv
    registration_qc.png/json     # historical six-hand preview
  models/                        # shared raw MANO_LEFT/RIGHT assets
  code/                          # library source snapshot for optional fresh inference
  tools/                         # portable rendering, summary, inference and verification tools
  sources/                       # original README, official arrays and executed source snapshots
  bundle_manifest.json
  README.md
```

`left/audit_statistics.json` preserves the earlier left publication's provenance. Historical audit records
may contain original paths; current atlas manifests resolve their PDF/archive/index paths relative to their
own side folder (`path_base=manifest_directory`). Root PDFs do not have duplicate side-folder copies or aliases.
The model license still applies. Library/source snapshots support independent inference; cached analysis and
re-rendering use the stored NPZ arrays and the portable tools.

The original right atlas predates this bundle; its script hash refers to that historical run. Its registration
targets and inference workload agree with the right archive. The left atlas records the new archive hash.
Cached rendering reproduces the same geometry/metrics; GPU/VTK versions can change rasterization and PDF bytes.
The combined summary retains original right fitting measurements; **no left-hand fitting or controlled timing run is
included in this bundle**. Current GPU resources are shared.

## Array schema

Load with `np.load("left/poses.npz", allow_pickle=False)`; Unicode metadata/names need no pickle.
Rows use sorted original PKL filenames. These filenames are provenance identifiers: a suffix `r` does not
make a row in the derived left archive a right-model pose. Read `model_side` for the actual model side.

| Key | Shape | Convention |
|---|---|---|
| `pose` | `(1554, 48)` | Absolute axis-angle in radians; first 3 global, then 15 joint rotations |
| `betas` | `(1554, 10)` | Shape per registration, unchanged between paired sides |
| `trans` | `(1554, 3)` | Translation after LBS, metres; full scanner frame |
| `v` | `(1554, 778, 3)` | Original right target or geometrically mirrored left target, metres |
| `J_transformed` | `(1554, 16, 3)` | Target joints in MANO order, including wrist, no fingertips |
| `faces` | `(1538, 3)` | Model triangle vertex indices, with side-appropriate winding |
| `predicted_vertices`, `reference_vertices` | `(1554, 778, 3)` | Cached manotorch float32 output stored as float64; independent NumPy float64 reference |
| `predicted_joints`, `reference_joints` | `(1554, 16, 3)` | Reconstructed/reference MANO 16 joints |
| `predicted_joints21` | `(1554, 21, 3)` | Cached manotorch joints with fingertips in manotorch order |
| `mano16_indices` | `(16,)` | Select these from manotorch 21 joints to obtain MANO order |
| `names`, `source_sha256`, `source_side` | `(1554,)` | Original PKL identifiers, hashes and original scan-side label |
| `model_side`, `metadata` | scalar strings | Actual model side and JSON inference/convention provenance |

For reflection about `x=0`, positions and translation multiply `(-1,1,1)`. Rotations obey
`R_left = S @ R_right @ S`, with `S=diag(-1,1,1)`. Axis-angle axial vectors multiply `(1,-1,-1)`.
This applies to the global rotation too. The official `L.npy` articulation is used: 115 samples contain
equivalent canonical axis-angle vectors, so compare rotation matrices rather than raw vectors.
Use `use_pca=False`, `flat_hand_mean=True`, `center_idx=None`, and **`fix_left_shapedirs=True` for left**.
Do not negate PCA coefficients or add a mean pose. Do not correct the stored raw left model twice.

Left/right model assets are approximate mirrors. The left atlas's third column measures the mirror residual;
the validation CSV separately measures manotorch vs independent corrected-left NumPy FK/LBS. A display-only
wrist centering, inverse global rotation and side-specific upright rotation make hands easier to view.
The left display adds a proper 180-degree rotation about display z, keeping fingers up and preserving
handedness. These transformations do not change measured errors. The summary uses atlas panels directly.

## Analyze without Torch

From this folder:

```sh
python3 tools/verify_registration_bundle.py .
```

This verifies the complete packaged inventory after moving or copying the folder.

```python
import numpy as np

with np.load("right/poses.npz", allow_pickle=False) as f:
    right_pose = f["pose"]
    right_targets = f["v"]
with np.load("left/poses.npz", allow_pickle=False) as f:
    left_pose = f["pose"]
    left_targets = f["v"]
    numerical_error_mm = np.linalg.norm(f["predicted_vertices"] - f["reference_vertices"], axis=-1) * 1000
    mirror_residual_mm = np.linalg.norm(f["predicted_vertices"] - f["v"], axis=-1) * 1000
np.testing.assert_array_equal(left_targets, right_targets * [-1, 1, 1])
print(numerical_error_mm.max(), mirror_residual_mm.max())
```

## Re-render from cached geometry

Run from this folder. These commands ignore any surrounding Python project and need no Torch installation.
GPU OpenGL rendering is verified by `--require-gpu-render`; inference device flags do not affect cached data.
Write to new directories to preserve the delivered atlases and hashes.

```sh
uv run --no-project --with numpy --with pyvista --with pillow --with reportlab --with matplotlib \
  python tools/render_registration_atlas.py --pose-archive left/poses.npz \
  --output-dir rerender/left --pdf-path rerender/registration_atlas_left.pdf --require-gpu-render
uv run --no-project --with numpy --with pyvista --with pillow --with reportlab --with matplotlib \
  python tools/render_registration_atlas.py --pose-archive right/poses.npz \
  --output-dir rerender/right --pdf-path rerender/registration_atlas_right.pdf --require-gpu-render
```

Each full atlas has 259 six-hand/three-column pages plus cover/index = 261 PDF pages. Try
`--max-pages 1 --export-pages 1` in a separate smoke output directory first. Use `--export-pages 1 130 259`
for selected PNGs, or `--export-pages` alone to skip PNG export. Full numerical references remain checked.

## Fresh MANO inference / fitting inputs

Use the source snapshot and stored raw models for fresh inference. Float32 chunked inference is recomputed,
validated against the independent references, and saved to a **new** archive:

```sh
uv run --no-project --with 'torch==2.11.0' --with numpy \
  python tools/reinfer_registration_bundle.py --bundle-dir . --side left --device cuda
# Use --side right for the right hand, or --device cpu without available CUDA.
```

Pass `left/poses_recomputed.npz` to the cached renderer to view the new outputs. For your own fitting code,
add `code/` to `sys.path`, load the arrays above, and initialize:

```python
import sys
sys.path.insert(0, "code")
from manotorch.manolayer import ManoLayer

layer = ManoLayer(side="left", mano_assets_root="models", use_pca=False,
                  flat_hand_mean=True, center_idx=None, fix_left_shapedirs=True)
# out = layer(pose_tensor, betas_tensor, trans_tensor), tensors shaped (B,48)/(B,10)/(B,3).
# Targets are f['v'] / f['J_transformed']; compare out.joints[:, f['mano16_indices']].
```

Mirrored targets have a small irreducible asset residual, so prefer `reference_vertices`/`reference_joints`
for an exact MANO_LEFT parameter-recovery benchmark and use mirrored `v` for symmetry/transfer evaluation.
Clearly identify the chosen target. Neither constitutes an independent left-scan fitting benchmark.

Rebuild the combined right/left statistics document from the stored report and completed atlas:

```sh
uv run --no-project --with numpy --with reportlab --with matplotlib --with pypdf \
  python tools/summarize_registration_mirror.py --bundle-dir .
```

PDF publication checks source hashes. Regenerated outputs alter their hashes; preserve the original bundle
manifest or explicitly update an inventory for your new run rather than treating it as the original result.
