import pytest
import torch
from conftest import requires_mano

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer

pytestmark = requires_mano


@pytest.mark.parametrize("side", ["right", "left"])
def test_compose_inverts_forward(mano_root, side):
    """compose(euler angles of a pose) gives back the same joint rotations (the global rotation is not encoded).

    For the left hand, forward() returns the twist and spread angles with the opposite sign of the convention
    compose() expects (the right-hand one), so they are flipped before composing.
    """
    layer = ManoLayer(side=side, mano_assets_root=mano_root).double()
    fk = AxisLayerFK(side=side, mano_assets_root=mano_root).double()
    pose = torch.randn(16, 48, dtype=torch.float64) * 0.5
    pose[:, :3] = 0
    out = layer(pose)

    T_g_a, R, ee = fk(out.transforms_abs)
    assert T_g_a.shape == (16, 16, 4, 4) and R.shape == (16, 16, 3, 3) and ee.shape == (16, 16, 3)
    if side == "left":
        ee = ee * torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)
    composed = fk.compose(ee)
    torch.testing.assert_close(layer(composed.reshape(16, 48)).transforms_abs, out.transforms_abs, atol=1e-6, rtol=0)


def test_compose_mirrors_between_sides(mano_root):
    """The same anatomical angles compose into mirrored joint rotations on the left and right hands."""
    angles = torch.randn(8, 16, 3, dtype=torch.float64) * 0.3
    rots = {}
    for side in ("right", "left"):
        fk = AxisLayerFK(side=side, mano_assets_root=mano_root).double()
        pose = fk.compose(angles).reshape(8, 48)
        rots[side] = ManoLayer(side=side, mano_assets_root=mano_root).double()(pose).transforms_abs[..., :3, :3]
    M = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(rots["left"], M @ rots["right"] @ M, atol=1e-6, rtol=0)


def test_compose_does_not_modify_input(mano_root):
    fk = AxisLayerFK(side="left", mano_assets_root=mano_root)
    angles = torch.randn(4, 16, 3) * 0.3
    before = angles.clone()
    first = fk.compose(angles)
    torch.testing.assert_close(angles, before, atol=0, rtol=0)
    torch.testing.assert_close(fk.compose(angles), first, atol=0, rtol=0)


def test_state_dict(mano_root):
    fk = AxisLayerFK(mano_assets_root=mano_root)
    assert list(fk.state_dict()) == ["TMPL_T_p_a", "TMPL_R_p_a", "TMPL_T_g_a"]


def test_anatomy_loss_zero_on_flat_hand(mano_root):
    layer = ManoLayer(mano_assets_root=mano_root)
    fk = AxisLayerFK(mano_assets_root=mano_root)
    loss = AnatomyConstraintLossEE()
    loss.setup()
    _, _, ee = fk(layer(torch.zeros(2, 48)).transforms_abs)
    assert loss(ee).item() < 1e-6


def test_anchor_layer_uses_packaged_assets(mano_root):
    anchors = AnchorLayer()(ManoLayer(mano_assets_root=mano_root)(torch.zeros(3, 48)).verts)
    assert anchors.shape == (3, 32, 3)
