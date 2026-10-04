import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .utils.geometry import axis_angle_to_matrix, quaternion_to_axis_angle, quaternion_to_matrix
from .utils.mano_io import find_mano_model, load_mano_model

# Vertices appended as fingertips (thumb, index, middle, ring, little), aligned with smplx
TIP_VERT_IDS = [744, 320, 443, 554, 671]

# Order of the 21 output joints (SNAP definition), indexing the 16 MANO joints followed by the 5 fingertips
#   original MANO joint order (right hand)
#                16-15-14-13-\
#                             \
#          17 --3 --2 --1------0
#        18 --6 --5 --4-------/
#        19 -12 -11 --10-----/
#          20 --9 --8 --7---/
JOINTS_REORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

# Faces that close the open wrist of the right-hand mesh
# https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py
CLOSE_FACES = [
    [92, 38, 122],
    [234, 92, 122],
    [239, 234, 122],
    [279, 239, 122],
    [215, 279, 122],
    [215, 122, 118],
    [215, 118, 117],
    [215, 117, 119],
    [215, 119, 120],
    [215, 120, 108],
    [215, 108, 79],
    [215, 79, 78],
    [215, 78, 121],
    [214, 215, 121],
]


@dataclass
class MANOOutput:
    verts: Optional[torch.Tensor]  # None with forward(..., joints_only=True)
    joints: torch.Tensor
    center_idx: Optional[int] = None
    center_joint: Optional[torch.Tensor] = None
    full_poses: Optional[torch.Tensor] = None
    betas: Optional[torch.Tensor] = None
    transforms_abs: Optional[torch.Tensor] = None


def _apply_transforms(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """T[..., :3] @ points + T[..., 3] for (..., 3, 4) transforms and (..., 3) points, without (..., 3, 3) temporaries."""
    out = torch.addcmul(T[..., 3], T[..., 0], points[..., 0:1])
    out = torch.addcmul(out, T[..., 1], points[..., 1:2])
    return torch.addcmul(out, T[..., 2], points[..., 2:3])


class _SkinApply(torch.autograd.Function):
    """Linear blend skinning's last step, v = T[..., :3] @ p + T[..., 3], with a hand-written elementwise backward.

    Autograd through einsum ran millions of 3x3 batched GEMVs, and through broadcasting it kept (..., 3, 3)
    temporaries. This saves only T and p, as autograd did, and its backward is itself differentiable.
    """

    @staticmethod
    def forward(ctx, T, points):
        ctx.save_for_backward(T, points)
        return _apply_transforms(T, points)

    @staticmethod
    def backward(ctx, grad):
        T, points = ctx.saved_tensors
        grad_T = grad_points = None
        if ctx.needs_input_grad[0]:  # dv_i / dT_ij = p_j (j < 3), 1 (j = 3)
            grad_T = torch.cat([grad.unsqueeze(-1) * points.unsqueeze(-2), grad.unsqueeze(-1)], -1)
        if ctx.needs_input_grad[1]:  # dv_i / dp_j = T_ij
            grad_points = T[..., 0, :3] * grad[..., 0:1]
            grad_points = torch.addcmul(grad_points, T[..., 1, :3], grad[..., 1:2])
            grad_points = torch.addcmul(grad_points, T[..., 2, :3], grad[..., 2:3])
        return grad_T, grad_points


def th_with_zeros(tensor):
    """Append the homogeneous row [0, 0, 0, 1] to a batch of (3, 4) transforms."""
    padding = tensor.new_zeros(tensor.shape[0], 1, 4)
    padding[..., 3] = 1.0
    return torch.cat([tensor, padding], 1)


class ManoLayer(torch.nn.Module):
    def __init__(
        self,
        rot_mode: str = "axisang",
        side: str = "right",
        center_idx: Optional[int] = None,
        mano_assets_root: str = "assets/mano",
        use_pca: bool = False,
        flat_hand_mean: bool = True,  # Only used in pca mode
        ncomps: int = 15,  # Only used in pca mode
        fix_left_shapedirs: bool = False,
        **kargs,
    ):
        """Differentiable MANO layer.

        Args:
            rot_mode: "axisang" (pose given as axis-angles or PCA coefficients) or "quat" (16 quaternions).
            side: "right" or "left".
            center_idx: if set, outputs are expressed relative to this joint (of the 21 output joints).
            mano_assets_root: folder containing `models/MANO_{RIGHT,LEFT}.npz` or `.pkl`.
            use_pca: pose articulation given as `ncomps` PCA coefficients instead of 45 axis-angle values.
            flat_hand_mean: if False, the articulation is relative to the MANO mean hand pose.
            ncomps: number of PCA components used when `use_pca` is True.
            fix_left_shapedirs: the official left-hand model ships the right-hand shape blend shapes without
                mirroring their x component (https://github.com/vchoutas/smplx/issues/48), so a left hand with
                non-zero betas is not the mirror of the right hand with the same betas. When True, the x component
                is negated for the left hand. Off by default, to match the official model and the data fitted with
                it (e.g. with manopth or smplx); this changes the left-hand shape, not the joint rotations.
        """
        super().__init__()
        self.center_idx = center_idx
        self.rot_mode = rot_mode
        self.side = side
        self.use_pca = use_pca
        self.mano_assets_root = mano_assets_root
        self.flat_hand_mean = flat_hand_mean
        self.ncomps = ncomps if use_pca else -1
        self.fix_left_shapedirs = fix_left_shapedirs

        if rot_mode == "axisang":
            self.rot_dim = 3
        elif rot_mode == "quat":
            self.rot_dim = 4
            if use_pca or not flat_hand_mean:
                warnings.warn("Quat mode doesn't support PCA pose or non flat_hand_mean !")
        else:
            raise NotImplementedError(f"Unrecognized rotation mode, expect [pca|axisang|quat], got {rot_mode}")

        # load model according to side flag
        smpl_data = load_mano_model(find_mano_model(mano_assets_root, side))

        shapedirs = smpl_data["shapedirs"].copy()
        if side == "left" and fix_left_shapedirs:
            shapedirs[:, 0, :] *= -1

        self.register_buffer("th_betas", torch.zeros(1, shapedirs.shape[-1], dtype=torch.float32))
        self.register_buffer("th_shapedirs", torch.as_tensor(shapedirs, dtype=torch.float32))
        self.register_buffer("th_posedirs", torch.as_tensor(smpl_data["posedirs"], dtype=torch.float32))
        self.register_buffer(
            "th_v_template", torch.as_tensor(smpl_data["v_template"], dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer("th_J_regressor", torch.as_tensor(smpl_data["J_regressor"], dtype=torch.float32))
        self.register_buffer("th_weights", torch.as_tensor(smpl_data["weights"], dtype=torch.float32))
        self.register_buffer("th_faces", torch.from_numpy(smpl_data["f"].astype(np.int64)))

        self.kintree_parents = list(smpl_data["kintree_table"][0].tolist())
        hands_components = smpl_data["hands_components"]

        if rot_mode == "axisang":
            hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data["hands_mean"]
            self.register_buffer("th_hands_mean", torch.as_tensor(hands_mean, dtype=torch.float32).unsqueeze(0))

        if rot_mode == "axisang" or use_pca:
            self.register_buffer("th_selected_comps", torch.as_tensor(hands_components[:ncomps], dtype=torch.float32))

        # constants used in forward, kept on the layer's device (not part of the state dict)
        self.register_buffer("_tip_vert_ids", torch.tensor(TIP_VERT_IDS), persistent=False)
        self.register_buffer("_joints_reorder", torch.tensor(JOINTS_REORDER), persistent=False)
        self.register_buffer(
            "_homo_row", torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, 1, 1, 4), persistent=False
        )
        # faces closing the wrist; follows the layer's device (get_mano_closed_faces() returns a CPU copy)
        close_faces = torch.tensor(CLOSE_FACES)
        if side == "left":
            close_faces = close_faces[:, [2, 1, 0]]
        self.register_buffer("th_closed_faces", torch.cat([self.th_faces, close_faces]), persistent=False)

    def rotation_by_axisang(self, pose_coeffs):
        hand_pose_coeffs = pose_coeffs[:, self.rot_dim :]
        root_pose_coeffs = pose_coeffs[:, : self.rot_dim]
        full_hand_pose = hand_pose_coeffs.mm(self.th_selected_comps) if self.use_pca else hand_pose_coeffs

        # Concatenate back global rot
        full_poses = torch.cat([root_pose_coeffs, self.th_hands_mean + full_hand_pose], 1)  # (B, 48)
        full_rots = axis_angle_to_matrix(full_poses.reshape(-1, 16, 3))  # (B, 16, 3, 3)
        return {"full_rots": full_rots, "full_poses": full_poses}

    def rotation_by_quaternion(self, pose_coeffs):
        batch_size = pose_coeffs.shape[0]
        full_quat_poses = pose_coeffs.view((batch_size, 16, 4))  # [B. 16, 4]
        full_rots = quaternion_to_matrix(full_quat_poses)  # [B, 16, 3, 3]
        full_poses = quaternion_to_axis_angle(full_quat_poses).reshape(batch_size, -1)  # [B, 16 x 3]
        return {"full_rots": full_rots, "full_poses": full_poses}

    def _shaped_template(self, betas: torch.Tensor):
        """$ \\bar{T} + B_S $, Eq. 2 and 4 in MANO. Returns (?, 778, 3), ? = betas.shape[0]."""
        shapedirs = self.th_shapedirs  # (778, 3, 10)
        B_S = (betas @ shapedirs.view(-1, shapedirs.shape[-1]).T).view(-1, *shapedirs.shape[:2])
        return self.th_v_template + B_S

    @staticmethod
    def _forward_kinematics(rots: torch.Tensor, J: torch.Tensor):
        """Global rotation and translation of the 16 joints, Eq. 4 in SMPL.

        The 15 finger joints form 5 chains of 3 joints attached to the root: chain level l (1, 2, 3)
        holds the joints l, l + 3, ..., l + 12.

        Args:
            rots: (B, 16, 3, 3) local joint rotations.
            J: (B, 16, 3) rest-pose joint locations.

        Returns:
            (B, 16, 3, 3) global rotations and (B, 16, 3) global translations.
        """
        par_rot, par_tsl, par_J = rots[:, :1], J[:, :1], J[:, :1]
        lev_rots, lev_tsls = [], []
        for lev in range(1, 4):
            lev_J = J[:, lev::3]  # (B, 5, 3)
            rot = par_rot @ rots[:, lev::3]
            tsl = (par_rot @ (lev_J - par_J).unsqueeze(-1)).squeeze(-1) + par_tsl
            lev_rots.append(rot)
            lev_tsls.append(tsl)
            par_rot, par_tsl, par_J = rot, tsl, lev_J
        # interleave the levels back into the MANO joint order 1, 2, 3, 4, ...
        rot = torch.cat([rots[:, :1], torch.stack(lev_rots, 2).flatten(1, 2)], 1)
        tsl = torch.cat([J[:, :1], torch.stack(lev_tsls, 2).flatten(1, 2)], 1)
        return rot, tsl

    def skinning_layer(self, full_rots: torch.Tensor, betas: Optional[torch.Tensor], joints_only: bool = False):
        batch_size = full_rots.shape[0]
        _betas = self.th_betas if betas is None else betas
        eye = torch.eye(3, dtype=full_rots.dtype, device=full_rots.device)
        pose_feature = (full_rots[:, 1:] - eye).reshape(batch_size, -1)  # (B, 15 x 9)

        if joints_only:
            # Only the 16 joints and the 5 fingertip vertices are needed. J(beta) is affine in beta, so it is
            # computed from the joint-regressed template and shape blend shapes instead of all 778 vertices.
            # They are derived on every call: callers may edit the th_* buffers after construction.
            tip_ids = self._tip_vert_ids
            n_betas = _betas.shape[-1]
            J_regressor, shapedirs, v_template = self.th_J_regressor, self.th_shapedirs, self.th_v_template
            J_shapedirs = (J_regressor @ shapedirs.reshape(shapedirs.shape[0], -1)).view(-1, n_betas)  # (16 x 3, 10)
            J = ((J_regressor @ v_template).view(1, -1) + _betas @ J_shapedirs.T).view(-1, 16, 3)
            J = J.expand(batch_size, -1, -1)  # (B, 16, 3)
            tip_shapedirs = shapedirs.index_select(0, tip_ids).view(-1, n_betas)  # (5 x 3, 10)
            tip_posedirs = self.th_posedirs.index_select(0, tip_ids).view(-1, pose_feature.shape[-1])  # (5 x 3, 135)
            tip_shaped = v_template.index_select(1, tip_ids).view(1, -1) + _betas @ tip_shapedirs.T  # (?, 5 x 3)
            T_P = (tip_shaped + pose_feature @ tip_posedirs.T).view(batch_size, -1, 3)  # (B, 5, 3)
            weights = self.th_weights.index_select(0, tip_ids)  # (5, 16)
        else:
            # ============== Shape Blend Shape and joints >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            v_shaped = self._shaped_template(_betas)  # (?, 778, 3), ? = 1, or B
            # $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
            J = (self.th_J_regressor @ v_shaped).expand(batch_size, -1, -1)  # (B, 16, 3)

            # ============== Pose Blend Shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # $ B_P = \sum_{n=1}^{9K} (R_n (\arrow{\theta}) -  R_n (\arrow{\theta}^{*})) * \mathbf{P}_n $  #Eq.3 in MANO
            posedirs = self.th_posedirs  # (778, 3, 135)
            B_P = (pose_feature @ posedirs.view(-1, posedirs.shape[-1]).T).view(batch_size, *posedirs.shape[:2])
            # $ T_P =\bar{\mathbf{T}} + B_S + B_P $ # Eq.2 in MANO
            T_P = v_shaped + B_P  # (B, 778, 3)
            weights = self.th_weights  # (778, 16)

        # ============== Global transforms $ G_k $ and $ G^{\prime}_k = G_k [I, -J_k] $ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        rot, tsl = self._forward_kinematics(full_rots, J)  # (B, 16, 3, 3), (B, 16, 3)
        tsl_prime = tsl - (rot @ J.unsqueeze(-1)).squeeze(-1)

        # ============== Linear blend skinning, Eq. 7 in SMPL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        G_prime = torch.cat([rot, tsl_prime.unsqueeze(-1)], -1).view(batch_size, 16, 12)
        T = (weights @ G_prime).view(batch_size, -1, 3, 4)  # (B, V, 3, 4)
        skinned = _SkinApply.apply(T, T_P)  # (B, V, 3)

        # In addition to MANO reference joints we sample vertices on each finger to serve as finger tips,
        # then reorder joints to match SNAP definition
        tips = skinned if joints_only else skinned.index_select(1, self._tip_vert_ids)
        joints = torch.cat([tsl, tips], 1).index_select(1, self._joints_reorder)  # (B, 21, 3)

        if self.center_idx is not None:
            center_joint = joints[:, self.center_idx].unsqueeze(1)
        else:  # dummy center joint (B, 1, 3)
            center_joint = torch.zeros_like(joints[:, :1])

        # apply center shift on verts, joints and global transforms
        joints = joints - center_joint
        verts = None if joints_only else skinned - center_joint
        tsl = tsl - center_joint
        homo_row = self._homo_row.to(rot.dtype).expand(batch_size, 16, 1, 4)
        transforms_abs = torch.cat([torch.cat([rot, tsl.unsqueeze(-1)], -1), homo_row], -2)  # (B, 16, 4, 4)

        return {
            "verts": verts,
            "joints": joints,
            "center_joint": center_joint,
            "transforms_abs": transforms_abs,
            "betas": _betas,
        }

    def forward(
        self,
        pose_coeffs: torch.Tensor,
        betas: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        *,
        joints_only: bool = False,
        **kwargs,
    ) -> MANOOutput:
        """
        Args:
            pose_coeffs: (B, 3 + 45), (B, 3 + ncomps) with use_pca, or (B, 16 x 4) quaternions with rot_mode="quat".
            betas: (B, 10), or (1, 10) to share one shape across the batch; None for the mean shape.
            transl: (B, 3) or (1, 3) translation in meters, added to verts, joints and transforms_abs after the
                center_idx centering (so the center joint lands at transl). With center_idx=None this matches
                manopth's th_trans; manopth instead skips the centering when a non-zero th_trans is given.
            joints_only: skip the 778 vertices (verts is None) and skin only the 5 fingertip vertices. The joints
                and transforms equal those of the full forward up to float rounding.
        """
        if self.rot_mode == "axisang":
            rot_blob = self.rotation_by_axisang(pose_coeffs)
        else:
            rot_blob = self.rotation_by_quaternion(pose_coeffs)

        skinning_blob = self.skinning_layer(rot_blob["full_rots"], betas, joints_only=joints_only)
        verts, joints, transforms_abs = skinning_blob["verts"], skinning_blob["joints"], skinning_blob["transforms_abs"]
        if transl is not None:
            offset = transl.reshape(-1, 1, 3)
            verts = None if verts is None else verts + offset
            joints = joints + offset
            column = torch.cat([offset, offset.new_zeros(offset.shape[0], 1, 1)], -1).unsqueeze(-1)  # (?, 1, 4, 1)
            transforms_abs = torch.cat([transforms_abs[..., :3], transforms_abs[..., 3:] + column], -1)
        return MANOOutput(
            verts=verts,
            joints=joints,
            center_idx=self.center_idx,
            center_joint=skinning_blob["center_joint"],
            full_poses=rot_blob["full_poses"],
            betas=skinning_blob["betas"],
            transforms_abs=transforms_abs,
        )

    def get_rotation_center(self, betas: Optional[torch.Tensor] = None):
        """

        V = MANO(theta, beta)

        Then we apply a rotation R on the vertices V
        V_1 = R @ V

        or, we can apply a rotation R on the global components of theta: first 3 elements of the theta
        theta' = CONCAT( SO3.log(R @ SO3.exp(theta[:3])), theta[3:] )

        V_2 = MANO(theta', beta)

        No doubt that, V_1 != V_2
        we found V_1 = V_2 + t, the t is an unknown translation offset

        Directly apply R on V would rotate V w.r.t the rotation center at V's [0,0,0] coordinate.
        However, apply R on the theta[:3] would cause the vertices rotate w.r.t to a rotation center at
        a non-zero, soley beta-sepcified center, C

        In other word, apply any disturb on the theta[:3] would not change the C's coordinates.
        the following code describe how we acquire the rotation center C

        This function will be called at artiboost/utils/refineunit.py in our upcoming work ArtiBoost
        """

        if betas is None:
            betas = self.th_betas

        batch_size = betas.shape[0]
        if self.center_idx is not None:
            return betas.new_zeros((batch_size, 3))

        # root joint of $ \mathcal{J}(\bar{\mathbf{T}} + B_S)$ # Eq.10 in SMPL
        return self.th_J_regressor[0] @ self._shaped_template(betas)  # (B, 3)

    def get_mano_closed_faces(self):
        """
        The default MANO mesh is "open" at the wrist. By adding additional faces, the hand mesh is closed,
        which looks much better.
        https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py

        The added faces are at the end (indices 1538 to 1551); they match the wrist part of the hand, which is
        not an external surface of the human.

        Returns a new tensor on the CPU; `th_closed_faces` holds the same faces on the layer's device.
        """
        return self.th_closed_faces.to("cpu", copy=True)
