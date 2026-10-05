"""Regressions found during the October review, including integration with torch.func."""

import math

import pytest
import torch
from conftest import requires_mano

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, _SkinApply
from manotorch.utils import geometry as g

CONVENTIONS = [a + b + c for a in "XYZ" for b in "XYZ" for c in "XYZ" if a != b and b != c]


@pytest.mark.parametrize("scale", [0.0, 1e-9, 1e-5, 0.009, 0.011, 0.01, 0.02])
def test_rotation_double_backward(scale):
    direction = torch.nn.functional.normalize(torch.tensor([[0.2, -0.4, 0.5]], dtype=torch.double), dim=-1)
    aa = (direction * scale).requires_grad_()
    for fn in (g.axis_angle_to_matrix, g.axis_angle_to_quaternion,
               lambda a: g.quaternion_to_axis_angle(g.axis_angle_to_quaternion(a))):
        assert torch.autograd.gradgradcheck(fn, (aa,))


@pytest.mark.parametrize("convention", CONVENTIONS)
@pytest.mark.parametrize("endpoint", [0, 1])
def test_exact_euler_lock_roundtrip(convention, endpoint):
    middle = ([0.0, math.pi] if convention[0] == convention[2] else [-math.pi / 2, math.pi / 2])[endpoint]
    angles = torch.tensor([[0.3, middle, 0.2]], dtype=torch.double)
    matrix = g.euler_angles_to_matrix(angles, convention)
    # Remove floating-point cos(pi/2), ensuring we test actual singular matrices rather than near-lock values.
    matrix = torch.where(matrix.abs() < 1e-15, 0.0, matrix).requires_grad_()
    recovered = g.matrix_to_euler_angles(matrix, convention)
    assert recovered[0, 2] == 0
    torch.testing.assert_close(g.euler_angles_to_matrix(recovered, convention), matrix, atol=1e-12, rtol=0)
    assert torch.isfinite(torch.autograd.grad(recovered.sum(), matrix)[0]).all()


def test_anatomy_configs_do_not_alias():
    first, second = AnatomyConstraintLossEE(), AnatomyConstraintLossEE()
    first.setup()
    second.setup()
    first.finger_mcp[2] = "+-:999"
    assert second.finger_mcp[2] == "+:90,-:0"
    supplied = ["+-:0", "+-:5", "+:90,-:0"]
    first.setup(finger_mcp=supplied)
    supplied[2] = "+-:999"
    assert first.finger_mcp[2] == "+:90,-:0"


def test_nonlocked_euler_inactive_branch_gradient():
    matrix = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]], dtype=torch.double,
                          requires_grad=True)
    angles = g.matrix_to_euler_angles(matrix, "XYZ")
    torch.testing.assert_close(g.euler_angles_to_matrix(angles, "XYZ"), matrix, atol=1e-12, rtol=0)
    assert torch.isfinite(torch.autograd.grad(angles.sum(), matrix)[0]).all()


@pytest.mark.parametrize("cfg", [["+-:0"], ["+-:0", "+-:nan", "+-:0"],
                                 ["+-:0", "+-:-1", "+-:0"], ["+-:0", "bad", "+-:0"]])
def test_anatomy_rejects_invalid_limits(cfg):
    with pytest.raises(ValueError):
        AnatomyConstraintLossEE().setup(finger_mcp=cfg)


def test_skin_torch_func():
    T = torch.randn(2, 3, 4, dtype=torch.double)
    points = torch.randn(2, 3, dtype=torch.double)

    def reference(t, p):
        return (t[..., :3] @ p.unsqueeze(-1)).squeeze(-1) + t[..., 3]

    tangents = (torch.randn_like(T), torch.randn_like(points))
    actual = torch.func.jvp(_SkinApply.apply, (T, points), tangents)
    expected = torch.func.jvp(reference, (T, points), tangents)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(torch.func.jacrev(_SkinApply.apply, argnums=(0, 1))(T, points),
                    torch.func.jacrev(reference, argnums=(0, 1))(T, points), strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.vmap(_SkinApply.apply)(T, points), reference(T, points))


@requires_mano
@pytest.mark.parametrize("joints_only", [False, True])
def test_mano_zero_double_backward(mano_root, joints_only):
    layer = ManoLayer(mano_assets_root=mano_root).double()
    pose = torch.zeros(1, 48, dtype=torch.double, requires_grad=True)
    betas = torch.zeros(1, 10, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradgradcheck(lambda p, b: layer(p, b, joints_only=joints_only).joints[:, ::5], (pose, betas))


@requires_mano
def test_fk_checkpoint_refreshes_basis(mano_root):
    fk = AxisLayerFK(mano_assets_root=mano_root).double()
    state = {k: v.clone() for k, v in fk.state_dict().items()}
    state["TMPL_R_p_a"][:, 1] @= g.euler_angles_to_matrix(torch.tensor([0.2, 0.0, 0.0], dtype=torch.double), "XYZ")
    wrapper = torch.nn.ModuleDict({"fk": fk})
    wrapper.load_state_dict({"fk." + k: v for k, v in state.items()})
    basis = fk.TMPL_R_p_a
    parents = torch.tensor(fk.transf_parent_mapping)
    expected = basis.index_select(1, parents).transpose(2, 3) @ basis
    torch.testing.assert_close(fk._Ra_par_tmplchd, expected)


@requires_mano
def test_closed_faces_checkpoint_refresh(mano_root):
    layer = ManoLayer(mano_assets_root=mano_root)
    state = {k: v.clone() for k, v in layer.state_dict().items()}
    state["th_faces"] = state["th_faces"].roll(1, dims=0)
    layer.load_state_dict(state)
    torch.testing.assert_close(layer.th_closed_faces[:1538], state["th_faces"])


@requires_mano
@pytest.mark.parametrize("joints_only", [False, True])
def test_mano_torch_func(mano_root, joints_only):
    layer = ManoLayer(mano_assets_root=mano_root).double()
    pose = torch.randn(2, 48, dtype=torch.double) * 0.2

    def fn(p):
        return layer(p, joints_only=joints_only).joints

    direction = torch.randn_like(pose)
    _, tangent = torch.func.jvp(fn, (pose,), (direction,))
    epsilon = 1e-6
    torch.testing.assert_close(tangent, (fn(pose + epsilon * direction) - fn(pose - epsilon * direction)) / (2 * epsilon),
                               atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(torch.vmap(lambda p: fn(p.unsqueeze(0)).squeeze(0))(pose), fn(pose))
    jac = torch.func.jacrev(fn, chunk_size=16)(pose)
    torch.testing.assert_close(torch.einsum("bjcnd,nd->bjc", jac, direction), tangent)


@requires_mano
@pytest.mark.parametrize("joints_only", [False, True])
def test_compiled_graph_and_gradients(mano_root, joints_only):
    layer = ManoLayer(mano_assets_root=mano_root).double()
    pose = (torch.randn(1, 48, dtype=torch.double) * 0.2).requires_grad_()
    betas = torch.randn(1, 10, dtype=torch.double, requires_grad=True)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    expected = layer(pose, betas, joints_only=joints_only).joints
    actual = compiled(pose, betas, joints_only=joints_only).joints
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), (pose, betas)),
                    torch.autograd.grad(expected.square().sum(), (pose, betas)), strict=True):
        torch.testing.assert_close(a, b)
