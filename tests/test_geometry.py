import itertools
import math

import numpy as np
import pytest
import torch

from manotorch.utils import geometry as g

D = torch.float64
CONVENTIONS = ["".join(p) for p in itertools.product("XYZ", repeat=3) if p[1] != p[0] and p[1] != p[2]]


def random_axis_angles(n, generator, max_angle=math.pi):
    axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=generator, dtype=D), dim=-1)
    return axis * torch.rand(n, 1, generator=generator, dtype=D) * max_angle


def skew_reference(v):
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v.unbind(-1)
    return torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(*v.shape[:-1], 3, 3)


def test_axis_angle_to_matrix_matches_matrix_exponential():
    aa = random_axis_angles(500, torch.Generator().manual_seed(0))
    aa = torch.cat([aa, torch.zeros(1, 3, dtype=D), torch.tensor([[1e-9, 0, 0], [0, 0, math.pi]], dtype=D)])
    R = g.axis_angle_to_matrix(aa)
    # PyTorch 2.0's batched matrix_exp has ~4e-11 absolute error on this sample. Check the conversion itself
    # against independent numpy Rodrigues at 1e-12, then check matrix_exp at its observed accuracy.
    values = aa.numpy()
    theta = np.linalg.norm(values, axis=-1, keepdims=True)
    K = skew_reference(aa).numpy()
    reference = np.eye(3) + np.sinc(theta / np.pi)[..., None] * K
    reference += (0.5 * np.sinc(theta / (2 * np.pi))**2)[..., None] * (K @ K)
    torch.testing.assert_close(R, torch.from_numpy(reference), atol=1e-12, rtol=0)
    torch.testing.assert_close(R, torch.linalg.matrix_exp(skew_reference(aa)), atol=1e-10, rtol=0)
    torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(3, dtype=D).expand_as(R), atol=1e-12, rtol=0)


def test_quaternion_conversions():
    generator = torch.Generator().manual_seed(1)
    aa = random_axis_angles(500, generator)
    q = g.axis_angle_to_quaternion(aa)
    torch.testing.assert_close(g.quaternion_to_matrix(q), g.axis_angle_to_matrix(aa), atol=1e-12, rtol=0)
    torch.testing.assert_close(g.quaternion_to_axis_angle(q), aa, atol=1e-10, rtol=0)
    # non-unit quaternions of both signs describe the rotation of q / |q|
    raw = torch.randn(500, 4, generator=generator, dtype=D)
    R = g.quaternion_to_matrix(raw)
    torch.testing.assert_close(
        R, g.quaternion_to_matrix(torch.nn.functional.normalize(raw, dim=-1)), atol=1e-12, rtol=0
    )
    r = g.quaternion_to_axis_angle(raw)
    torch.testing.assert_close(g.axis_angle_to_matrix(r), R, atol=1e-12, rtol=0)
    assert (torch.linalg.vector_norm(r, dim=-1) <= math.pi + 1e-12).all()
    # matrix -> quaternion recovers +-q, with w >= 0
    unit = torch.nn.functional.normalize(raw, dim=-1)
    back = g.matrix_to_quaternion(R)
    assert (back[:, 0] >= 0).all()
    torch.testing.assert_close(back, torch.where(unit[:, :1] < 0, -unit, unit), atol=1e-12, rtol=0)
    torch.testing.assert_close(g.axis_angle_to_matrix(g.rotation_to_axis_angle(R)), R, atol=1e-12, rtol=0)


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_euler_round_trip(convention):
    generator = torch.Generator().manual_seed(2)
    angles = (torch.rand(500, 3, generator=generator, dtype=D) * 2 - 1) * math.pi
    if convention[0] == convention[2]:
        angles[:, 1] = angles[:, 1].abs()  # proper Euler: middle angle in [0, pi]
    else:
        angles[:, 1] *= 0.5  # Tait-Bryan: middle angle in [-pi/2, pi/2]
    R = g.euler_angles_to_matrix(angles, convention)
    i, j, k = ("XYZ".index(c) for c in convention)
    elementary = [
        g.axis_angle_to_matrix(torch.nn.functional.one_hot(torch.tensor(a), 3).to(D) * angles[:, n : n + 1])
        for n, a in enumerate((i, j, k))
    ]
    torch.testing.assert_close(R, elementary[0] @ elementary[1] @ elementary[2], atol=1e-12, rtol=0)
    torch.testing.assert_close(g.matrix_to_euler_angles(R, convention), angles, atol=1e-9, rtol=0)


def test_euler_gimbal_lock_is_finite():
    R = g.euler_angles_to_matrix(torch.tensor([[0.3, math.pi / 2, 0.2]], dtype=D), "XYZ") * (1 + 1e-12)
    angles = g.matrix_to_euler_angles(R, "XYZ")
    assert torch.isfinite(angles).all()
    torch.testing.assert_close(g.euler_angles_to_matrix(angles, "XYZ"), R, atol=1e-9, rtol=0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        g.euler_angles_to_matrix(torch.zeros(3), "XXY")
    with pytest.raises(ValueError):
        g.matrix_to_euler_angles(torch.eye(3), "XYW")
    with pytest.raises(ValueError):
        g.matrix_to_quaternion(torch.zeros(4, 4))


@pytest.mark.parametrize("scale", [0.0, 1e-9, 1e-7, 1.0])
def test_gradients_near_zero_rotation(scale):
    aa = (torch.randn(6, 3, dtype=D) * scale).requires_grad_(True)
    assert torch.autograd.gradcheck(g.axis_angle_to_matrix, (aa,))
    assert torch.autograd.gradcheck(g.axis_angle_to_quaternion, (aa,))
    assert torch.autograd.gradcheck(lambda a: g.quaternion_to_axis_angle(g.axis_angle_to_quaternion(a)), (aa,))
