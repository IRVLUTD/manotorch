"""Compose a right and a left hand from the same anatomy aligned Euler angles.

    uv run python scripts/simple_compose.py                                  # interactive window
    uv run python scripts/simple_compose.py --gif doc/simple_compose_new.gif # orbiting GIF, rendered off-screen

The index finger is bent: its MCP joint (1) by pi/6 around the spread axis and pi/2 around the bend axis, its PIP
and DIP joints (2, 3) by pi/2 around the bend axis. The angles follow one convention for both hands, so the two
composed hands are mirror images. Arrows: red = back (twist), green = up (spread), blue = left (bend).
"""

import argparse
import math

import pyvista as pv
import torch
from trimesh import Trimesh

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer

#  transform order of right hand
#         15-14-13-\
#                   \
# *   3-- 2 -- 1 -----0   < NOTE: demo on this finger
#   6 -- 5 -- 4 ----/
#   12 - 11 - 10 --/
#    9-- 8 -- 7 --/


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def show_or_record(pl: pv.Plotter, gif: str | None):
    if gif is None:
        pl.add_camera_orientation_widget()
        pl.show()
        return
    pl.window_size = (1024, 1024)
    view_up = [-1, 0, 0]
    path = pl.generate_orbital_path(factor=2.0, n_points=36, viewup=view_up, shift=0.1)
    pl.open_gif(gif)
    pl.orbit_on_path(path, write_frames=True, step=0.05, viewup=view_up)
    pl.close()
    print(f"saved {gif}")


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    composed_ee = torch.zeros((1, 16, 3), device=device)  # (twist, spread, bend) per joint
    composed_ee[:, 1] = torch.tensor([0, math.pi / 6, math.pi / 2])
    composed_ee[:, 2] = torch.tensor([0, 0, math.pi / 2])
    composed_ee[:, 3] = torch.tensor([0, 0, math.pi / 2])

    pl = pv.Plotter(off_screen=args.gif is not None)
    poses = {}
    for side in ["right", "left"]:
        mano_layer = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=0,
            mano_assets_root=args.mano_assets_root,
            flat_hand_mean=True,
            use_pca=args.use_pca,
            ncomps=45,
        ).to(device)
        axis_layer = AxisLayerFK(side=side, mano_assets_root=args.mano_assets_root).to(device)

        pose = axis_layer.compose(composed_ee).reshape(1, 48)  # axis-angles, (1, 16 x 3)
        if args.use_pca:  # express the articulation with the 45 PCA components
            pose[:, 3:] = (pose[:, 3:] - mano_layer.th_hands_mean) @ torch.linalg.inv(mano_layer.th_selected_comps)
        poses[side] = pose

        mano_output = mano_layer(pose)
        T_g_a, _, _ = axis_layer(mano_output.transforms_abs)
        mesh = pv.wrap(
            Trimesh(mano_output.verts[0].cpu().numpy(), mano_layer.get_mano_closed_faces().numpy(), process=False)
        )
        pl.add_mesh(mesh, opacity=0.5, smooth_shading=True, color="orange" if side == "right" else "cyan")

        centers = T_g_a[0, :, :3, 3].cpu().numpy()  # (16, 3)
        axes = T_g_a[0, :, :3, :3].cpu().numpy()  # (16, 3, 3), columns: back, up, left
        for k, color in enumerate(["red", "green", "blue"]):
            pl.add_arrows(centers, axes[:, :, k], color=color, mag=0.02)

    # mirrored poses share their PCA coefficients; as axis-angles, the y and z components change sign
    mirror = (
        torch.ones(48, device=device) if args.use_pca else torch.tensor([1.0, -1.0, -1.0], device=device).repeat(16)
    )
    is_mirror = torch.allclose(poses["left"], poses["right"] * mirror, atol=1e-4)
    print(f"Is the composed left hand the mirror of the right hand? {is_mirror}")

    show_or_record(pl, args.gif)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gif", default=None, help="write an orbiting GIF here instead of opening a window")
    parser.add_argument("--no-pca", dest="use_pca", action="store_false", help="feed axis-angles instead of PCA")
    parser.add_argument("--mano-assets-root", default="assets/mano")
    main(parser.parse_args())
