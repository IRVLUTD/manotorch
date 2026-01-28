import math

import pyvista as pv
import torch
from trimesh import Trimesh

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def main(off_screen=False, use_pca=True):
    pl = pv.Plotter(off_screen=off_screen)
    pl.add_camera_orientation_widget()

    device = get_device()
    print(f"Using device: {device}")

    #  transform order of right hand
    #         15-14-13-\
    #                   \
    # *   3-- 2 -- 1 -----0   < NOTE: demo on this finger
    #   6 -- 5 -- 4 ----/
    #   12 - 11 - 10 --/
    #    9-- 8 -- 7 --/

    composed_ee = torch.zeros((1, 16, 3))
    composed_ee[:, 1] = torch.tensor([0, math.pi / 6, math.pi / 2]).unsqueeze(0)
    composed_ee[:, 2] = torch.tensor([0, 0, math.pi / 2]).unsqueeze(0)
    composed_ee[:, 3] = torch.tensor([0, 0, math.pi / 2]).unsqueeze(0)

    composed_ee = composed_ee.to(device)

    poses_aa = []

    for side in ["right", "left"]:
        mano_layer = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=0,
            mano_assets_root="assets/mano",
            flat_hand_mean=True,
            use_pca=use_pca,
            ncomps=45,
        )
        axisFK = AxisLayerFK(
            side=side,
            mano_assets_root="assets/mano",
        )

        mano_layer = mano_layer.to(device)
        axisFK = axisFK.to(device)

        hand_faces = mano_layer.get_mano_closed_faces()  # (NF, 3)

        if use_pca:
            hand_mean = mano_layer.th_hands_mean
            hand_comp = mano_layer.th_selected_comps

        # if side == "left": # --- flip the ee for left hand
        #     composed_ee[:, :, 0] *= -1.0
        #     composed_ee[:, :, 1] *= -1.0

        composed_aa = axisFK.compose(composed_ee).clone()  # (B=1, 16, 3)
        composed_aa = composed_aa.reshape(1, -1)  # (1, 16x3)

        if use_pca:
            composed_aa[:, 3:48] = torch.matmul(composed_aa[:, 3:48] - hand_mean, hand_comp.inverse())  # use PCA

        poses_aa.append(composed_aa)

        mano_output = mano_layer(composed_aa)

        T_g_p = mano_output.transforms_abs  # (B=1, 16, 4, 4)
        T_g_a, R, ee = axisFK(T_g_p)
        hand_verts = mano_output.verts.squeeze(0)  # (NV, 3)
        mesh = pv.wrap(Trimesh(hand_verts.cpu().numpy(), hand_faces.cpu().numpy()))

        pl.add_mesh(
            mesh,
            opacity=0.5,
            name=f"mesh_{side}",
            smooth_shading=True,
            color="orange" if side == "right" else "cyan",
        )

        bul_axes_loc = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 16, 1, 1).to(device)
        bul_axes_glb = torch.matmul(T_g_a[:, :, :3, :3], bul_axes_loc)  # (B, 16, 3, 3)
        b_axes_dir = bul_axes_glb[:, :, :, 0].cpu().numpy()  # back direction (B, 16, 3)
        u_axes_dir = bul_axes_glb[:, :, :, 1].cpu().numpy()  # up direction (B, 16, 3)
        l_axes_dir = bul_axes_glb[:, :, :, 2].cpu().numpy()  # left direction (B, 16, 3)
        axes_cen = T_g_a[:, :, :3, 3].cpu().numpy()  # center (B, 16, 3)
        pl.add_arrows(axes_cen, b_axes_dir, color="red", mag=0.02)
        pl.add_arrows(axes_cen, u_axes_dir, color="green", mag=0.02)
        pl.add_arrows(axes_cen, l_axes_dir, color="blue", mag=0.02)

    # check the composed poses
    is_close = torch.allclose(poses_aa[0], poses_aa[1], atol=1e-4)
    print(f"Are the composed axis-angle poses of right and left hands close? {is_close}")

    if not off_screen:
        pl.show()
    else:
        pl.window_size = (1024, 1024)
        view_up = [-1, 0, 0]
        path = pl.generate_orbital_path(factor=2.0, n_points=36, viewup=view_up, shift=0.1)
        pl.open_gif("simple_compose_new.gif")
        pl.orbit_on_path(path, write_frames=True, step=0.05, viewup=view_up)
        pl.close()


if __name__ == "__main__":
    main(off_screen=True, use_pca=True)
