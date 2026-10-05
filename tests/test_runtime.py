"""Guards for properties checked once by hand: no host-device synchronization, no warnings, finite gradients."""

import warnings

import pytest
import torch
from conftest import requires_mano

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer

pytestmark = requires_mano


def build_layers(mano_root, device):
    loss = AnatomyConstraintLossEE()
    loss.setup()
    return {
        "pca": ManoLayer(mano_assets_root=mano_root, use_pca=True, flat_hand_mean=False, ncomps=45).to(device),
        "quat": ManoLayer(mano_assets_root=mano_root, rot_mode="quat").to(device),
        "fk": AxisLayerFK(side="left", mano_assets_root=mano_root).to(device),
        "left": ManoLayer(side="left", mano_assets_root=mano_root, center_idx=9).to(device),
        "anchors": AnchorLayer().to(device),
        "loss": loss,
    }


def run_everything(layers, device):
    """Forward and backward through every layer, in each pose mode and option."""
    pca, quat, fk, left, anchors, loss = (layers[k] for k in ("pca", "quat", "fk", "left", "anchors", "loss"))
    pose = (torch.randn(8, 48, device=device) * 0.5).requires_grad_(True)
    betas = torch.randn(8, 10, device=device).requires_grad_(True)
    transl = torch.randn(8, 3, device=device)
    q = torch.nn.functional.normalize(torch.randn(8, 16, 4, device=device), dim=-1).view(8, 64).requires_grad_(True)

    out = pca(pose, betas, transl)
    joints = pca(pose, betas[:1], joints_only=True).joints
    left_out = left(pose, betas)
    _, _, ee = fk(left_out.transforms_abs)
    composed = fk.compose(ee)
    total = (
        out.verts.sum()
        + joints.sum()
        + loss(ee)
        + anchors(out.verts).sum()
        + composed.sum()
        + quat(q).verts.sum()
        + left_out.transforms_abs.sum()
    )
    total.backward()
    return pose.grad, betas.grad, q.grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_no_host_device_synchronization(mano_root):
    layers = build_layers(mano_root, "cuda")  # copying the buffers to the device synchronizes
    run_everything(layers, "cuda")  # warm up: lazy CUDA initialization may synchronize too
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Synchronization debug mode is a prototype.*")
        torch.cuda.set_sync_debug_mode("error")
        try:
            run_everything(layers, "cuda")
        finally:
            torch.cuda.set_sync_debug_mode("default")


def test_no_warnings(mano_root):
    warn_always = torch.is_warn_always_enabled()
    torch.set_warn_always(True)  # report warnings that torch emits only once per process
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grads = run_everything(build_layers(mano_root, "cpu"), "cpu")
    finally:
        torch.set_warn_always(warn_always)
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("scale", [0.0, 1e-9, 1e-7])
@pytest.mark.parametrize("joints_only", [False, True])
def test_gradients_at_tiny_rotations(mano_root, scale, joints_only):
    layer = ManoLayer(mano_assets_root=mano_root).double()
    pose = torch.randn(2, 48, dtype=torch.float64) * scale
    pose[:, :3] = torch.randn(2, 3, dtype=torch.float64) * 0.5  # a regular global rotation
    pose.requires_grad_(True)
    betas = torch.randn(2, 10, dtype=torch.float64).requires_grad_(True)

    def f(p, b):
        out = layer(p, b, joints_only=joints_only)
        return out.joints if joints_only else out.verts[:, ::40]

    assert torch.autograd.gradcheck(f, (pose, betas))
