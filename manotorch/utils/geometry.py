"""Rotation conversions between axis-angles, quaternions, rotation matrices and Euler angles.

Conventions:
- quaternions are real-first, (w, x, y, z); the ones returned have w >= 0;
- axis-angles are (..., 3) vectors whose norm is the angle in radians;
- Euler angles follow an intrinsic convention string such as "XYZ": R = R_X(a) @ R_Y(b) @ R_Z(c).

Every function is batched over the leading dimensions, differentiable (including at the zero rotation) and free of
host-device synchronization.
"""

import torch

_AXES = {"X": 0, "Y": 1, "Z": 2}


def _skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) vectors to (..., 3, 3) cross-product matrices: _skew(v) @ u == cross(v, u)."""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).unflatten(-1, (3, 3))


def _canonical(quaternions: torch.Tensor) -> torch.Tensor:
    """The representative of +-q with a non-negative real part."""
    return torch.where(quaternions[..., :1] < 0, -quaternions, quaternions)


def _axis_indices(convention: str):
    if len(convention) != 3 or any(letter not in _AXES for letter in convention):
        raise ValueError(f"Invalid convention {convention!r}: expected three letters from X, Y, Z.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention!r}: consecutive axes must differ.")
    return tuple(_AXES[letter] for letter in convention)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angles to (..., 3, 3) rotation matrices.

    Rodrigues' formula with r = angle * axis: R = cos(t) I + sin(t)/t [r]x + (1 - cos(t))/t^2 r r^T, where both
    ratios are written with sinc so they stay exact and differentiable at t = 0.
    """
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)  # (..., 1)
    sin_ratio = torch.sinc(angle / torch.pi)  # sin(t) / t
    half_sinc = torch.sinc(angle / (2 * torch.pi))  # sin(t/2) / (t/2)
    cos_ratio = 0.5 * half_sinc * half_sinc  # (1 - cos(t)) / t^2
    # scale the (..., 3) vectors before forming the (..., 3, 3) terms: fewer large temporaries
    R = (cos_ratio * axis_angle).unsqueeze(-1) * axis_angle.unsqueeze(-2) + _skew(sin_ratio * axis_angle)
    R.diagonal(dim1=-2, dim2=-1).add_(torch.cos(angle))
    return R


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angles to (..., 4) unit quaternions (cos(t/2), sin(t/2) axis), real part first."""
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    # sin(t/2) * axis = sin(t/2)/t * r = 0.5 sinc(t / 2pi) * r
    return torch.cat([torch.cos(0.5 * angle), 0.5 * torch.sinc(angle / (2 * torch.pi)) * axis_angle], dim=-1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternions, real part first and not necessarily normalized, to (..., 3, 3) rotation matrices.

    For q = (w, v): R = I + 2 / |q|^2 (w [v]x + [v]x^2), with [v]x^2 = v v^T - |v|^2 I.
    """
    w, v = quaternions[..., :1], quaternions[..., 1:]
    scale = 2.0 / (quaternions * quaternions).sum(-1, keepdim=True)  # (..., 1)
    scaled_v = scale * v
    R = scaled_v.unsqueeze(-1) * v.unsqueeze(-2) + _skew(w * scaled_v)
    R.diagonal(dim1=-2, dim2=-1).add_(1.0 - (scaled_v * v).sum(-1, keepdim=True))
    return R


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternions, real part first and not necessarily normalized, to (..., 3) axis-angles.

    The angle is in [0, pi]. With |v| = |q| sin(t/2): r = t v / |v| = v / (|q| * 0.5 * sinc(t / 2pi)).
    """
    w, v = quaternions[..., :1], quaternions[..., 1:]
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    half_angle = torch.atan2(v_norm, w.abs())  # of the representative with w >= 0, in [0, pi/2]
    # the sign flips v for w < 0; |q| = hypot(|v|, w)
    scale = torch.copysign(torch.ones_like(w), w) / (0.5 * torch.hypot(v_norm, w) * torch.sinc(half_angle / torch.pi))
    return v * scale


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices to (..., 4) unit quaternions, real part first, with w >= 0.

    Shepperd's method: 4w^2, 4x^2, 4y^2 and 4z^2 follow from the diagonal; the largest of them (always >= 1) gives
    a well-conditioned component q_i, and the other three come from 4 q_i q_j, read off the off-diagonal entries.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {tuple(matrix.shape)}.")
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.flatten(-2).unbind(-1)
    four_sq = torch.stack(
        [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1
    )  # 4 * (w^2, x^2, y^2, z^2)
    # row i holds 4 q_i * (w, x, y, z)
    rows = torch.stack(
        [
            torch.stack([four_sq[..., 0], m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, four_sq[..., 1], m01 + m10, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m01 + m10, four_sq[..., 2], m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, four_sq[..., 3]], dim=-1),
        ],
        dim=-2,
    )
    best = four_sq.argmax(dim=-1, keepdim=True)  # (..., 1)
    row = torch.gather(rows, -2, best.unsqueeze(-1).expand(*best.shape, 4)).squeeze(-2)
    four_q_best = 2.0 * torch.sqrt(torch.gather(four_sq, -1, best))  # 4 q_i = 2 sqrt(4 q_i^2), with 4 q_i^2 >= 1
    return _canonical(row / four_q_best)


def rotation_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrices to (..., 3) axis-angles with angles in [0, pi]."""
    return quaternion_to_axis_angle(matrix_to_quaternion(rotation))


def _elementary_rotation(axis: int, angle: torch.Tensor) -> torch.Tensor:
    """(...,) angles to (..., 3, 3) rotations about the X (0), Y (1) or Z (2) axis."""
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    if axis == 0:
        entries = (one, zero, zero, zero, c, -s, zero, s, c)
    elif axis == 1:
        entries = (c, zero, s, zero, one, zero, -s, zero, c)
    else:
        entries = (c, -s, zero, s, c, zero, zero, zero, one)
    return torch.stack(entries, dim=-1).unflatten(-1, (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """(..., 3) intrinsic Euler angles (a, b, c) to (..., 3, 3) rotations R = R_i(a) @ R_j(b) @ R_k(c).

    `convention` names the axes i, j, k, e.g. "XYZ" (Tait-Bryan) or "ZYZ" (proper Euler).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError(f"Invalid Euler angles shape {tuple(euler_angles.shape)}.")
    i, j, k = _axis_indices(convention)
    a, b, c = euler_angles.unbind(-1)
    return _elementary_rotation(i, a) @ _elementary_rotation(j, b) @ _elementary_rotation(k, c)


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """(..., 3, 3) rotation matrices to (..., 3) intrinsic Euler angles, the inverse of euler_angles_to_matrix.

    For R = R_i(a) R_j(b) R_k(c) and s = +1 if (i, j) is a cyclic pair (XY, YZ, ZX), -1 otherwise:
    - Tait-Bryan (i != k): b = asin(s R_ik), a = atan2(-s R_jk, R_kk), c = atan2(-s R_ij, R_ii);
    - proper Euler (i == k), with l the remaining axis: b = acos(R_ii), a = atan2(R_ji, -s R_li),
      c = atan2(R_ij, s R_il).
    The middle angle is in [-pi/2, pi/2] (Tait-Bryan) or [0, pi] (proper Euler).
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {tuple(matrix.shape)}.")
    i, j, k = _axis_indices(convention)
    s = 1.0 if (j - i) % 3 == 1 else -1.0
    R = matrix
    if i != k:
        b = torch.asin((s * R[..., i, k]).clamp(-1.0, 1.0))
        a = torch.atan2(-s * R[..., j, k], R[..., k, k])
        c = torch.atan2(-s * R[..., i, j], R[..., i, i])
    else:
        l_axis = 3 - i - j
        b = torch.acos(R[..., i, i].clamp(-1.0, 1.0))
        a = torch.atan2(R[..., j, i], -s * R[..., l_axis, i])
        c = torch.atan2(R[..., i, j], s * R[..., i, l_axis])
    return torch.stack([a, b, c], dim=-1)
