import pytest
import torch
from conftest import requires_mano

from manotorch.manolayer import JOINTS_REORDER, TIP_VERT_IDS, ManoLayer
from manotorch.utils.mano_io import find_mano_model, load_mano_model

pytestmark = requires_mano

STATE_DICT_KEYS = {
    "axisang": [
        "th_betas",
        "th_shapedirs",
        "th_posedirs",
        "th_v_template",
        "th_J_regressor",
        "th_weights",
        "th_faces",
        "th_hands_mean",
        "th_selected_comps",
    ],
    "quat": ["th_betas", "th_shapedirs", "th_posedirs", "th_v_template", "th_J_regressor", "th_weights", "th_faces"],
}


def rodrigues(aa):
    """(..., 3) axis-angles to (..., 3, 3) rotation matrices, R = I + sin(t) K + (1 - cos(t)) K^2."""
    theta = torch.linalg.vector_norm(aa, dim=-1, keepdim=True).unsqueeze(-1)
    k = aa / torch.linalg.vector_norm(aa, dim=-1, keepdim=True)
    zero = torch.zeros_like(k[..., 0])
    K = torch.stack([zero, -k[..., 2], k[..., 1], k[..., 2], zero, -k[..., 0], -k[..., 1], k[..., 0], zero], -1)
    K = K.view(*k.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def reference_mano(model, full_pose, betas):
    """Plain float64 MANO (Eq. 2-4 in MANO, Eq. 4 and 7 in SMPL), walking the model's own kinematic tree.

    The model parameters are rounded to float32 first, as ManoLayer stores them.
    """
    # Only floating model parameters enter the equations. PyTorch 2.0 does not accept uint32 kintree arrays.
    t = {k: torch.as_tensor(model[k]).float().double() for k in
         ("v_template", "shapedirs", "J_regressor", "posedirs", "weights")}
    v_shaped = t["v_template"] + torch.einsum("vck,bk->bvc", t["shapedirs"], betas)
    J = torch.einsum("jv,bvc->bjc", t["J_regressor"], v_shaped)
    R = rodrigues(full_pose.view(-1, 16, 3))
    pose_feature = (R[:, 1:] - torch.eye(3, dtype=torch.float64)).flatten(1)
    v_posed = v_shaped + torch.einsum("vck,bk->bvc", t["posedirs"], pose_feature)

    parents = model["kintree_table"][0]
    G = []
    for k in range(16):
        local = torch.zeros(len(full_pose), 4, 4, dtype=torch.float64)
        local[:, :3, :3] = R[:, k]
        local[:, :3, 3] = J[:, k] - (J[:, parents[k]] if k > 0 else 0)
        local[:, 3, 3] = 1
        G.append(local if k == 0 else G[parents[k]] @ local)
    G = torch.stack(G, 1)  # (B, 16, 4, 4)
    G_prime = G.clone()
    G_prime[..., :3, 3] -= (G[..., :3, :3] @ J.unsqueeze(-1)).squeeze(-1)
    T = torch.einsum("vk,bkij->bvij", t["weights"], G_prime)
    verts = (T[..., :3, :3] @ v_posed.unsqueeze(-1)).squeeze(-1) + T[..., :3, 3]
    return verts, G


def make_inputs(layer, batch_size, generator):
    dtype = torch.float64
    if layer.rot_mode == "quat":
        pose = torch.randn(batch_size, 16, 4, generator=generator, dtype=dtype)
        pose = torch.nn.functional.normalize(pose, dim=-1)  # unit quaternions, as quaternion_to_axis_angle expects
        return pose.view(batch_size, 64), torch.randn(batch_size, 10, generator=generator, dtype=dtype)
    ndof = layer.ncomps if layer.use_pca else 45
    pose = torch.randn(batch_size, 3 + ndof, generator=generator, dtype=dtype) * 0.6
    return pose, torch.randn(batch_size, 10, generator=generator, dtype=dtype)


CONFIGS = [
    {},
    {"use_pca": True, "flat_hand_mean": False, "ncomps": 45},
    {"use_pca": True, "flat_hand_mean": True, "ncomps": 6},
    {"flat_hand_mean": False, "center_idx": 9},
    {"center_idx": 0},
    {"rot_mode": "quat"},
]


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("fix_left_shapedirs", [False, True])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "default")
def test_matches_reference(mano_root, side, fix_left_shapedirs, cfg):
    layer = ManoLayer(side=side, mano_assets_root=mano_root, fix_left_shapedirs=fix_left_shapedirs, **cfg).double()
    model = load_mano_model(find_mano_model(mano_root, side))
    if side == "left" and fix_left_shapedirs:
        model["shapedirs"] = model["shapedirs"] * torch.tensor([-1.0, 1.0, 1.0]).view(1, 3, 1).numpy()
    pose, betas = make_inputs(layer, 8, torch.Generator().manual_seed(0))
    out = layer(pose, betas)

    verts, G = reference_mano(model, out.full_poses, betas)
    center = out.center_joint
    torch.testing.assert_close(out.verts, verts - center, atol=1e-10, rtol=0)
    torch.testing.assert_close(out.transforms_abs[..., :3, :3], G[..., :3, :3], atol=1e-10, rtol=0)
    torch.testing.assert_close(out.transforms_abs[..., :3, 3], G[..., :3, 3] - center, atol=1e-10, rtol=0)
    joints = torch.cat([G[..., :3, 3], verts[:, TIP_VERT_IDS]], 1)[:, JOINTS_REORDER]
    torch.testing.assert_close(out.joints, joints - center, atol=1e-10, rtol=0)
    if layer.center_idx is not None:
        assert out.joints[:, layer.center_idx].abs().max() < 1e-12


@pytest.mark.parametrize("rot_mode", ["axisang", "quat"])
def test_state_dict_and_attributes(mano_root, rot_mode):
    cfg = {"use_pca": True, "flat_hand_mean": False, "ncomps": 45} if rot_mode == "axisang" else {}
    layer = ManoLayer(rot_mode=rot_mode, mano_assets_root=mano_root, **cfg)
    assert list(layer.state_dict()) == STATE_DICT_KEYS[rot_mode]
    shapes = {k: tuple(v.shape) for k, v in layer.state_dict().items()}
    assert shapes["th_shapedirs"] == (778, 3, 10) and shapes["th_posedirs"] == (778, 3, 135)
    assert shapes["th_v_template"] == (1, 778, 3) and shapes["th_J_regressor"] == (16, 778)
    assert shapes["th_weights"] == (778, 16) and shapes["th_faces"] == (1538, 3)
    if rot_mode == "axisang":
        assert shapes["th_hands_mean"] == (1, 45) and shapes["th_selected_comps"] == (45, 45)
    assert len(layer.kintree_parents) == 16 and layer.side == "right"
    assert tuple(layer.get_mano_closed_faces().shape) == (1552, 3)


def test_unknown_kwargs_are_ignored(mano_root):
    # downstream code relies on **kargs swallowing unknown arguments
    ManoLayer(mano_assets_root=mano_root, some_unknown_argument=1)


def test_betas_none_uses_mean_shape(mano_root):
    layer = ManoLayer(mano_assets_root=mano_root)
    pose = torch.randn(4, 48) * 0.5
    torch.testing.assert_close(layer(pose).verts, layer(pose, torch.zeros(4, 10)).verts)


def test_outputs_do_not_alias_buffers(mano_root):
    layer = ManoLayer(mano_assets_root=mano_root)
    out = layer(torch.zeros(2, 48))
    out.verts += 1.0
    out.joints += 1.0
    out.transforms_abs += 1.0
    torch.testing.assert_close(layer(torch.zeros(2, 48)).verts, out.verts - 1.0)


def test_left_is_mirror_of_right(mano_root):
    right = ManoLayer(side="right", mano_assets_root=mano_root).double()
    pose, betas = torch.randn(8, 48, dtype=torch.float64) * 0.5, torch.randn(8, 10, dtype=torch.float64)
    mirror_pose = pose * torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64).repeat(16)
    flip = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    for fix, b, tol in [(False, torch.zeros_like(betas), 1e-9), (True, betas, 1e-5)]:
        left = ManoLayer(side="left", mano_assets_root=mano_root, fix_left_shapedirs=fix).double()
        torch.testing.assert_close(left(mirror_pose, b).verts, right(pose, b).verts * flip, atol=tol, rtol=0)
    # the official left model is not a mirror once betas are non-zero
    left = ManoLayer(side="left", mano_assets_root=mano_root).double()
    assert (left(mirror_pose, betas).verts - right(pose, betas).verts * flip).abs().max() > 1e-3


def test_gradients(mano_root):
    layer = ManoLayer(mano_assets_root=mano_root, use_pca=True, flat_hand_mean=False, ncomps=45).double()
    pose = (torch.randn(2, 48, dtype=torch.float64) * 0.5).requires_grad_(True)
    betas = torch.randn(2, 10, dtype=torch.float64).requires_grad_(True)
    torch.autograd.gradcheck(lambda p, b: layer(p, b).verts[:, ::50], (pose, betas))
    # zero pose: the axis-angle conversion must stay differentiable at the identity
    zero = torch.zeros(2, 48, dtype=torch.float64, requires_grad=True)
    ManoLayer(mano_assets_root=mano_root).double()(zero).verts.sum().backward()
    assert torch.isfinite(zero.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu(mano_root):
    layer = ManoLayer(mano_assets_root=mano_root, use_pca=True, flat_hand_mean=False, ncomps=45)
    pose, betas = torch.randn(32, 48) * 0.5, torch.randn(32, 10)
    cpu = layer(pose, betas)
    gpu = layer.cuda()(pose.cuda(), betas.cuda())
    torch.testing.assert_close(gpu.verts.cpu(), cpu.verts, atol=1e-5, rtol=0)
    torch.testing.assert_close(gpu.transforms_abs.cpu(), cpu.transforms_abs, atol=1e-5, rtol=0)


@pytest.mark.parametrize("side", ["right", "left"])
def test_fingertips_match_smplx(mano_root, side):
    """Fingertips are the MANO vertices smplx uses (smplx/vertex_ids.py), at the SNAP tip joints 4, 8, 12, 16, 20."""
    smplx_tips = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671}
    out = ManoLayer(side=side, mano_assets_root=mano_root)(torch.randn(4, 48) * 0.5)
    torch.testing.assert_close(out.joints[:, [4, 8, 12, 16, 20]], out.verts[:, list(smplx_tips.values())])


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "default")
def test_joints_only_matches_full_forward(mano_root, side, cfg):
    layer = ManoLayer(side=side, mano_assets_root=mano_root, **cfg).double()
    pose, betas = make_inputs(layer, 8, torch.Generator().manual_seed(1))
    for b in (betas, betas[:1], None):  # per-frame, shared and mean shape
        full, fast = layer(pose, b), layer(pose, b, joints_only=True)
        assert fast.verts is None
        torch.testing.assert_close(fast.joints, full.joints, atol=1e-12, rtol=0)
        torch.testing.assert_close(fast.transforms_abs, full.transforms_abs, atol=1e-12, rtol=0)
        torch.testing.assert_close(fast.center_joint, full.center_joint, atol=1e-12, rtol=0)


def test_joints_only_follows_edited_buffers(mano_root):
    # downstream code corrects the left shapedirs in place after construction
    layer = ManoLayer(side="left", mano_assets_root=mano_root).double()
    layer.th_shapedirs[:, 0, :] *= -1
    pose, betas = torch.randn(4, 48, dtype=torch.float64) * 0.5, torch.randn(4, 10, dtype=torch.float64)
    torch.testing.assert_close(
        layer(pose, betas, joints_only=True).joints, layer(pose, betas).joints, atol=1e-12, rtol=0
    )


def test_joints_only_gradients(mano_root):
    layer = ManoLayer(mano_assets_root=mano_root, use_pca=True, flat_hand_mean=False, ncomps=45).double()
    pose = (torch.randn(2, 48, dtype=torch.float64) * 0.5).requires_grad_(True)
    betas = torch.randn(2, 10, dtype=torch.float64).requires_grad_(True)
    torch.autograd.gradcheck(lambda p, b: layer(p, b, joints_only=True).joints, (pose, betas))


@pytest.mark.parametrize("center_idx", [None, 9])
@pytest.mark.parametrize("joints_only", [False, True])
def test_transl(mano_root, center_idx, joints_only):
    layer = ManoLayer(mano_assets_root=mano_root, center_idx=center_idx).double()
    pose, betas = torch.randn(4, 48, dtype=torch.float64) * 0.5, torch.randn(4, 10, dtype=torch.float64)
    base = layer(pose, betas, joints_only=joints_only)
    for transl in (torch.randn(4, 3, dtype=torch.float64), torch.randn(1, 3, dtype=torch.float64)):
        out = layer(pose, betas, transl, joints_only=joints_only)
        offset = transl.reshape(-1, 1, 3)
        torch.testing.assert_close(out.joints, base.joints + offset)
        if not joints_only:
            torch.testing.assert_close(out.verts, base.verts + offset)
        torch.testing.assert_close(out.transforms_abs[..., :3, :3], base.transforms_abs[..., :3, :3])
        torch.testing.assert_close(out.transforms_abs[..., :3, 3], base.transforms_abs[..., :3, 3] + offset)
        torch.testing.assert_close(out.transforms_abs[..., 3, :], base.transforms_abs[..., 3, :])
        torch.testing.assert_close(out.center_joint, base.center_joint)
        if center_idx is not None:  # the center joint lands at transl
            torch.testing.assert_close(out.joints[:, center_idx], transl.expand(4, 3))


def test_closed_faces(mano_root):
    for side in ("right", "left"):
        layer = ManoLayer(side=side, mano_assets_root=mano_root)
        faces = layer.get_mano_closed_faces()
        assert faces.device.type == "cpu" and torch.equal(faces, layer.th_closed_faces)
        torch.testing.assert_close(faces[:1538], layer.th_faces, atol=0, rtol=0)
        faces += 1  # a copy: the buffer is not modified
        assert not torch.equal(faces, layer.th_closed_faces)
        if torch.cuda.is_available():
            layer = layer.cuda()
            assert layer.th_closed_faces.is_cuda and layer.get_mano_closed_faces().device.type == "cpu"


def test_second_order_gradients(mano_root):
    # the skinning uses a hand-written backward, which must itself be differentiable
    layer = ManoLayer(mano_assets_root=mano_root).double()
    pose = (torch.randn(1, 48, dtype=torch.float64) * 0.5).requires_grad_(True)
    betas = torch.randn(1, 10, dtype=torch.float64).requires_grad_(True)
    assert torch.autograd.gradgradcheck(lambda p, b: layer(p, b).verts[:, ::97], (pose, betas))
