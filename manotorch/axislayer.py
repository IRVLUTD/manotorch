import torch
from torch.nn import Module

from manotorch.manolayer import ManoLayer
from manotorch.utils.geometry import euler_angles_to_matrix, matrix_to_euler_angles, rotation_to_axis_angle


def _homogeneous(rot: torch.Tensor, tsl: torch.Tensor) -> torch.Tensor:
    """Stack (..., 3, 3) rotations and (..., 3, 1) translations into (..., 4, 4) transforms."""
    top = torch.cat([rot, tsl], -1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom[..., 3] = 1.0
    return torch.cat([top, bottom], -2)


def _chain_rotations(R_par_chd: torch.Tensor) -> torch.Tensor:
    """Compose (B, 16, 3, 3) parent-to-child rotations along the MANO kinematic tree into global rotations.

    The 15 finger joints form 5 chains of 3 joints attached to the root: chain level l (1, 2, 3)
    holds the joints l, l + 3, ..., l + 12.
    """
    root = R_par_chd[:, :1]
    levels = [root]
    for lev in range(1, 4):
        levels.append(levels[-1] @ R_par_chd[:, lev::3])
    return torch.cat([root, torch.stack(levels[1:], 2).flatten(1, 2)], 1)


class AxisAdaptiveLayer(torch.nn.Module):
    def __init__(self, side: str = "right"):
        super(AxisAdaptiveLayer, self).__init__()
        self.joints_mapping = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
        self.parent_joints_mappings = [0, 5, 6, 0, 9, 10, 0, 17, 18, 0, 13, 14, 0, 1, 2]
        self.side = side
        thumb_up = [1.0, 1.0, 1.0] if side == "right" else [-1.0, 1.0, 1.0]
        up_axis_base = torch.tensor([[0.0, 1.0, 0.0]] * 13 + [thumb_up] * 3, dtype=torch.float32)
        self.register_buffer("up_axis_base", up_axis_base.unsqueeze(0))
        # the back axis of the root is +x for both sides
        self.register_buffer("_root_b_axis", torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32), persistent=False)
        self.register_buffer("_joints_mapping", torch.tensor(self.joints_mapping), persistent=False)
        self.register_buffer("_parent_joints_mapping", torch.tensor(self.parent_joints_mappings), persistent=False)

    def forward(self, hand_joints, transf):
        """Compute the back (twist), up (spread), and left (bend) axes direction of the hand
        Args:
            hand_joints (torch.Tensor): (B, 21, 3)
            transf (torch.Tensor): (B, 16, 4, 4)
        Returns:
            b_axis (torch.Tensor): (B, 16, 3)
            u_axis (torch.Tensor): (B, 16, 3)
            l_axis (torch.Tensor): (B, 16, 3)
        """
        bs = transf.shape[0]

        b_axis = hand_joints.index_select(1, self._parent_joints_mapping) - hand_joints.index_select(
            1, self._joints_mapping
        )
        b_axis = (transf[:, 1:, :3, :3].transpose(2, 3) @ b_axis.unsqueeze(-1)).squeeze(-1)
        b_axis = torch.cat((self._root_b_axis.expand(bs, 1, 3), b_axis), dim=1)  # (B, 16, 3)

        l_axis = torch.linalg.cross(b_axis, self.up_axis_base.expand(bs, 16, 3), dim=2)
        u_axis = torch.linalg.cross(l_axis, b_axis, dim=2)

        return (
            b_axis / torch.linalg.vector_norm(b_axis, dim=2, keepdim=True),
            u_axis / torch.linalg.vector_norm(u_axis, dim=2, keepdim=True),
            l_axis / torch.linalg.vector_norm(l_axis, dim=2, keepdim=True),
        )


class AxisLayerFK(Module):
    def __init__(self, side: str = "right", mano_assets_root: str = "assets/mano"):
        super(AxisLayerFK, self).__init__()
        self.transf_parent_mapping = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        self.side = side

        tmpl_mano = ManoLayer(side=side, mano_assets_root=mano_assets_root)(
            torch.zeros(1, 48, dtype=torch.float32), torch.zeros(1, 10, dtype=torch.float32)
        )
        tmpl_transf_abs = tmpl_mano.transforms_abs  # tmpl_T_g_p
        tmpl_b_axis, tmpl_u_axis, tmpl_l_axis = AxisAdaptiveLayer(side=side)(tmpl_mano.joints, tmpl_transf_abs)
        tmpl_R_p_a = torch.stack((tmpl_b_axis, tmpl_u_axis, tmpl_l_axis), dim=3)  # (1, 16, 3, 3)
        tmpl_T_p_a = _homogeneous(tmpl_R_p_a, tmpl_R_p_a.new_zeros(1, 16, 3, 1))  # (1, 16, 4, 4)
        tmpl_T_g_a = torch.matmul(tmpl_transf_abs, tmpl_T_p_a)  # (1, 16, 4, 4)
        self.register_buffer("TMPL_T_p_a", tmpl_T_p_a.float())
        self.register_buffer("TMPL_R_p_a", tmpl_R_p_a.float())
        self.register_buffer("TMPL_T_g_a", tmpl_T_g_a.float())

        parent = torch.tensor(self.transf_parent_mapping)
        self.register_buffer("_parent", parent, persistent=False)
        # rotation from each template parent's anatomy frame to its template child's anatomy frame
        Ra_par_tmplchd = self.TMPL_R_p_a[:, parent].transpose(2, 3) @ self.TMPL_R_p_a
        self.register_buffer("_Ra_par_tmplchd", Ra_par_tmplchd, persistent=False)
        # The left-hand anatomy frames are right-handed, so a motion mirrored from the right hand turns the opposite
        # way around the twist and spread axes. Angles are reported in the right-hand convention: R -> P R P with
        # P = diag(-1, -1, 1), which negates the twist and spread angles. Mirrored poses then give equal angles.
        sign = torch.tensor([-1.0, -1.0, 1.0] if side == "left" else [1.0, 1.0, 1.0], dtype=torch.float32)
        self.register_buffer("_angle_sign", sign, persistent=False)
        self.register_buffer("_rot_sign", sign[:, None] * sign[None, :], persistent=False)

    def forward(self, transf):
        """extract the anatomy aligned euler angles from the MANO global transformation
        #  transform order of right hand
        #         15-14-13-\
        #                   \
        #    3-- 2 -- 1 -----0
        #   6 -- 5 -- 4 ----/
        #   12 - 11 - 10 --/
        #    9-- 8 -- 7 --/

        Args:
            transf (torch.Tensor): [B, 16, 4, 4] MANO'joints global transformation
                defined in MANO origial frame

        Returns:
            T_g_a: torch.Tensor: [B, 16, 4, 4] MANO's joints global transformation defined in  anatomy aligned frame;
            Ra_tmplchd_chd: torch.Tensor: [B, 16, 3, 3], anatomy aligned rotation matrix;
            ee_a_tmplchd_chd: torch.Tensor: [B, 16, 3] anatomy aligned euler angle (twist, spread, bend);
            For the left hand, the rotations and angles follow the right-hand convention: mirrored poses of the two
            hands give the same angles, and compose() takes them back.
        """
        R_g_a = transf[:, :, :3, :3] @ self.TMPL_R_p_a  # (B, 16, 3, 3)
        T_g_a = _homogeneous(R_g_a, transf[:, :, :3, 3:])  # (B, 16, 4, 4)

        Ra_par_chd = R_g_a.index_select(1, self._parent).transpose(2, 3) @ R_g_a  # (B, 16, 3, 3)
        Ra_tmplchd_chd = (self._Ra_par_tmplchd.transpose(2, 3) @ Ra_par_chd) * self._rot_sign

        ee_a_tmplchd_chd = matrix_to_euler_angles(Ra_tmplchd_chd, convention="XYZ")  # (B, 16, 3)
        return T_g_a, Ra_tmplchd_chd, ee_a_tmplchd_chd

    def compose(self, angles):
        """Compose the MANO pose (\theta) from the anatomy aligned euler angles.
        #  transform order of right hand
        #         15-14-13-\
        #                   \
        #    3-- 2 -- 1 -----0
        #   6 -- 5 -- 4 ----/
        #   12 - 11 - 10 --/
        #    9-- 8 -- 7 --/

        Args:
            angles (torch.Tensor): [B, 16, 3] anatomy aligned euler angles, in the right-hand convention for both
                hands (as returned by forward); not modified.

        Returns:
            torch.Tensor: mano pose (\theta) in the MANO original frame
        """
        ee_tmplchd_chd = angles * self._angle_sign  # (B, 16, 3)

        Ra_tmplchd_chd = euler_angles_to_matrix(ee_tmplchd_chd, convention="XYZ")  # (B, 16, 3, 3)
        Ra_par_chd = self._Ra_par_tmplchd @ Ra_tmplchd_chd  # (B, 16, 3, 3)

        R_g_a = _chain_rotations(Ra_par_chd)  # (B, 16, 3, 3)
        R_g_p = R_g_a @ self.TMPL_R_p_a.transpose(2, 3)  # (B, 16, 3, 3)

        Rp_par_chd = R_g_p.index_select(1, self._parent).transpose(2, 3) @ R_g_p  # (B, 16, 3, 3)
        return rotation_to_axis_angle(Rp_par_chd)  # (B, 16, 3)
