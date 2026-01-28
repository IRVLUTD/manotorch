import argparse

import numpy as np
import pyvista as pv
import torch
from trimesh import Trimesh

from manotorch.anchorlayer import AnchorLayer
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


def main(args, off_screen=False, seed=42):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    BS = 1
    hand_pose = torch.randn(BS, 48, device=device)
    pl = pv.Plotter(off_screen=off_screen)
    pl.add_camera_orientation_widget()

    for side in [
        "right",
        "left",
    ]:
        mano_layer = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=0,
            mano_assets_root="assets/mano",
            flat_hand_mean=False,
            use_pca=True,
            ncomps=45,
        )
        axis_layer = AxisLayerFK(side=mano_layer.side, mano_assets_root="assets/mano")
        anchor_layer = AnchorLayer(anchor_root="assets/anchor")

        mano_layer = mano_layer.to(device)
        axis_layer = axis_layer.to(device)
        anchor_layer = anchor_layer.to(device)

        mano_results = mano_layer(hand_pose)
        verts = mano_results.verts  # (B, N, 3)
        faces = mano_layer.get_mano_closed_faces()

        T_g_p = mano_results.transforms_abs  # (B, 16, 4, 4)
        T_g_a, R, ee = axis_layer(T_g_p)  # ee (B, 16, 3)

        bul_axes_loc = torch.eye(3).reshape(1, 1, 3, 3).repeat(BS, 16, 1, 1).to(device)
        bul_axes_glb = torch.matmul(T_g_a[:, :, :3, :3], bul_axes_loc)  # (B, 16, 3, 3)

        b_axes_dir = bul_axes_glb[:, :, :, 0].cpu().numpy()  # back direction (B, 16, 3)
        u_axes_dir = bul_axes_glb[:, :, :, 1].cpu().numpy()  # up direction (B, 16, 3)
        l_axes_dir = bul_axes_glb[:, :, :, 2].cpu().numpy()  # left direction (B, 16, 3)
        axes_cen = T_g_a[:, :, :3, 3].cpu().numpy()  # center (B, 16, 3)

        F = faces.cpu().numpy()
        V = verts[0].cpu().numpy()
        tmesh = Trimesh(V, F, process=False)
        mesh = pv.wrap(tmesh)
        pl.add_mesh(
            mesh,
            opacity=0.3,
            name=f"{side}_mesh",
            smooth_shading=True,
            color="orange" if side == "right" else "cyan",
        )

        if args.mode == "axis":
            pl.add_arrows(axes_cen, b_axes_dir, color="red", mag=0.02)
            pl.add_arrows(axes_cen, u_axes_dir, color="green", mag=0.02)
            pl.add_arrows(axes_cen, l_axes_dir, color="blue", mag=0.02)
        elif args.mode == "anchor":
            anchors = anchor_layer(verts)[0].cpu().numpy()
            n_achors = anchors.shape[0]
            for i in range(n_achors):
                pl.add_mesh(
                    pv.Cube(center=anchors[i], x_length=3e-3, y_length=3e-3, z_length=3e-3),
                    color="yellow",
                    name=f"{side}_anchor{i}",
                )

    if not off_screen:
        pl.show(interactive=True)
    else:
        pl.window_size = (1024, 1024)
        view_up = [-1, 0, 0]
        path = pl.generate_orbital_path(factor=2.0, n_points=36, viewup=view_up, shift=0.1)
        pl.open_gif("simple_app_new.gif")
        pl.orbit_on_path(path, write_frames=True, step=0.05, viewup=view_up)
        pl.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["axis", "anchor"], default="axis", help="visualize axis or anchor")
    args = parser.parse_args()
    main(args, off_screen=True)
