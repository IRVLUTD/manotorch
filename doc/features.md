# Anatomical hand utilities

All spatial outputs use metres, all angle tensors use radians, and quaternions use `(w, x, y, z)`.
Match the hand side, device and dtype of the MANO and anatomical layers.

## Anatomical Consistent Basis

MANO articulation contains 15 local joint rotations plus the global wrist orientation. The anatomical basis labels
three directions at each joint: back/twist, up/spread, and left/bend. These are the columns of its rotation matrix.

`AxisAdaptiveLayer` derives those axes from the supplied posed joints and transforms. `AxisLayerFK` instead builds
fixed reference bases from a flat, mean-shape hand during initialization, then expresses the supplied joint rotations
relative to them. The latter is the basis used for the Euler loss and hand composition. It is independent of the pose
and shape being evaluated, and it does not adapt the reference basis to each predicted shape.

```python
import torch
from manotorch.manolayer import ManoLayer
from manotorch.axislayer import AxisLayerFK

side = "left"
mano = ManoLayer(side=side, use_pca=False, flat_hand_mean=True)
axis = AxisLayerFK(side=side)
out = mano(torch.zeros(2, 48))
anatomical_transforms, anatomical_rotations, angles = axis(out.transforms_abs)
# Shapes: (2, 16, 4, 4), (2, 16, 3, 3), (2, 16, 3)
# angles[..., 0:3] = twist, spread, bend, in radians
```

Translation and `center_idx` change the positions in `anatomical_transforms`, not the extracted articulation angles.
For the left hand, the returned local rotation matrices and Euler angles follow the right-hand anatomical convention.
Mirrored poses give equal articulation angles. The global transforms still describe the actual left hand.

The anatomical utilities use the **16 MANO joints**, whose order differs from the 21 output joints:

| MANO slots | Joint chain |
| --- | --- |
| 0 | wrist |
| 1, 2, 3 | index MCP, PIP, DIP |
| 4, 5, 6 | middle MCP, PIP, DIP |
| 7, 8, 9 | little MCP, PIP, DIP |
| 10, 11, 12 | ring MCP, PIP, DIP |
| 13, 14, 15 | thumb CMC, MCP, final hinge |

The 21-joint `ManoLayer.joints` output orders wrist, thumb, index, middle, ring, little, with four slots per finger
including the tip. Do not index an angle tensor with those 21-joint slot numbers.

## Anatomy Loss

`AnatomyConstraintLossEE` penalizes Euler angles outside configured intervals. For lower and upper bounds `l, u`,
the penalty is `relu(l - angle) + relu(angle - u)`, with angles within `1e-6` radians of zero suppressed. The three
axis penalties are added per joint; the wrist is excluded. This is a pose prior, not a collision or contact loss.

```python
from manotorch.anatomy_loss import AnatomyConstraintLossEE

loss = AnatomyConstraintLossEE(reduction="mean")
loss.setup(finger_mcp=("+-:0", "+-:5", "+:90,-:0"))
value = loss(angles)
```

Configuration strings use **degrees**: `"+-:45"` allows `[-45, 45]`; `"+:45,-:15"` allows `[-15, 45]`.
Each joint configuration contains three entries in twist/spread/bend order. All tolerances must be finite and
nonnegative. The defaults are:

| Joint type | Twist | Spread | Bend |
| --- | --- | --- | --- |
| Thumb CMC | ±45° | −15° to +45° | 0° to +45° |
| Thumb MCP | 0° | ±10° | 0° to +90° |
| Thumb final hinge | 0° | 0° | 0° to +90° |
| Finger MCP | 0° | ±5° | 0° to +90° |
| Finger PIP / DIP | 0° | 0° | 0° to +90° |

`reduction="none"` returns `(B, 15)` in this order: index/middle/ring/little MCPs, those fingers' PIPs, their DIPs,
then the three thumb joints. `mean` averages these 15 joint penalties over the batch; `sum` sums them.

`setup()` copies the supplied configurations. Public configuration lists remain editable; an edit rebuilds the
limits on the next call. Configure the loss before compiling or capturing CUDA graphs. Limits and indices follow
`.to(device)`; for historical callers they also move to the angle device on the first call. Device/dtype/configuration
changes are initialization work, and steady-state calls have no host-device synchronization. Prefer warming up the
entire pipeline before performance measurements.

Euler extraction uses intrinsic `XYZ`: `R = Rx(twist) @ Ry(spread) @ Rz(bend)`. At gimbal lock the inverse cannot be
smooth or uniquely recover the original angles. The conversion sets the last angle to zero and chooses an equivalent
rotation. Avoid interpreting a derivative through the singularity as a physical anatomical gradient.

## Composing the Hand

`AxisLayerFK.compose(angles)` maps `(B, 16, 3)` anatomical Euler angles to `(B, 16, 3)` MANO axis-angles. It preserves
the input tensor, uses the same angle convention for both hands, and returns articulation with a zero global rotation.
The wrist angle slot does not carry a recoverable global orientation. Retain and apply that orientation separately.

```python
angles = torch.zeros(2, 16, 3)
angles[:, 1, 2] = torch.pi / 3  # index MCP bend
pose = axis.compose(angles).reshape(2, 48)
pose[:, :3] = global_axis_angle  # caller's (2, 3) wrist orientation
out = mano(pose, betas, transl)
```

This example requires `use_pca=False, flat_hand_mean=True`. A PCA layer expects PCA coefficients; a layer with
`flat_hand_mean=False` adds the MANO mean articulation to its input. Directly feeding the composed absolute pose
to either setting changes the pose interpretation. Rotation round-trips should compare matrices, since axis-angle
and Euler representatives are not unique.

Older left-hand `AxisLayerFK.forward()` returned different twist/spread signs. When consuming old stored angles,
convert those two signs or regenerate the angles from the original transforms. Preserve the dataset's shapedirs
choice independently: fitted official-model data usually needs the default `fix_left_shapedirs=False`, whereas
pipelines fitted with a corrected basis must keep that correction. Apply it exactly once.

## Anchor Interpolation

`AnchorLayer` returns 32 anchors from `(B, 778, 3)` vertices. The package supplies its triangle indices, barycentric
weights and region mapping, so construction does not depend on the working directory.

```python
from manotorch.anchorlayer import AnchorLayer

anchors = AnchorLayer()(out.verts)  # (B, 32, 3), same coordinate system and units as out.verts
```

For a triangle `(v0, v1, v2)` and the two stored weights, the anchor is
`v0 + w1 * (v1 - v0) + w2 * (v2 - v0)`. This is differentiable in the mesh vertices. Translate, center or rotate the
mesh before interpolating to obtain anchors in the corresponding frame. Move the anchor layer with the mesh device.
The vertex topology must remain the MANO topology; `joints_only=True` supplies no mesh and cannot be used here.

Pass `anchor_root="..."` to use custom definitions. Custom mapping pickles are trusted local resources, unlike the
restricted MANO model loader. Inspect `anchor_mapping` and `merged_vertex_assignment` for contact-region bookkeeping.

## Mesh subdivision

`UpSampleLayer` splits each triangle into four and appends the midpoint of each unique edge.
It preserves shared edges and face winding; it does not smooth the original mesh.

```python
from manotorch.upsamplelayer import UpSampleLayer

upsample = UpSampleLayer().prepare(mano.th_faces, vertex_count=778).to(out.verts.device)
dense_vertices, dense_faces = upsample(out.verts)
```

For changing topology, the existing `upsample(vertices, faces)` call remains supported. Reusing the same
faces tensor caches its topology; ordinary in-place edits invalidate that cache. Passing a freshly constructed
faces tensor rebuilds it. `prepare` takes a snapshot, so call it again when intentionally changing that snapshot,
including edits through `.data` or external storage which bypass version counters. Inference tensors without
version counters should use the snapshot path for caching.

Shared `(F,3)` / `(1,F,3)` faces broadcast across the vertex batch. Distinct `(B,F,3)` topologies need equal
unique-edge counts for a rectangular output. Cached integer buffers follow device conversions and are omitted
from checkpoints; prepare the topology again after constructing or loading a layer. Returned faces have independent
storage, so editing them does not corrupt the cache. `clear_cache()` releases the cached topology.

This layer is separate from MANO forward and fitting: its cache speeds repeated subdivision only.
