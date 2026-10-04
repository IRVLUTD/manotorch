"""Visualize the anatomy aligned axes (or the anchors) of a random right hand and its left counterpart.

    uv run python scripts/simple_app.py                              # interactive window
    uv run python scripts/simple_app.py --gif doc/axis_new.gif       # orbiting GIF, rendered off-screen
    uv run python scripts/simple_app.py --mode anchor --gif doc/anchor.gif

The random pose is drawn in the anatomy aligned angle space, so it stays natural: each finger curls by a random
amount with coupled flexion of its three joints, the MCP joints spread a little, and nothing twists. Both hands
are composed from the same angles, so the left hand is the mirror of the right one. Arrows: red = back (twist),
green = up (spread), blue = left (bend).
"""

import argparse

import pyvista as pv
import torch
from trimesh import Trimesh

from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def random_natural_angles(seed: int) -> torch.Tensor:
    """(1, 16, 3) anatomy aligned Euler angles (twist, spread, bend) of a natural random hand pose."""
    generator = torch.Generator().manual_seed(seed)

    def uniform(low, high):
        return low + (high - low) * torch.rand(1, generator=generator).item()

    ee = torch.zeros(1, 16, 3)
    for finger in range(4):  # index, middle, pinky, ring: MCP, PIP and DIP at joints 1 + 3k, 2 + 3k, 3 + 3k
        curl, mcp = uniform(0.0, 1.0), 1 + 3 * finger
        ee[0, mcp] = torch.tensor([0.0, uniform(-8, 8), 60 * curl])
        ee[0, mcp + 1, 2] = 80 * curl
        ee[0, mcp + 2, 2] = 55 * curl
    curl = uniform(0.0, 1.0)  # thumb: CMC, MCP and IP at joints 13, 14, 15
    ee[0, 13] = torch.tensor([0.0, uniform(10, 40), 30 * curl])
    ee[0, 14, 2] = 40 * curl
    ee[0, 15, 2] = 50 * curl
    return torch.deg2rad(ee)


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

    angles = random_natural_angles(args.seed).to(device)  # sampled on the CPU: the pose does not depend on the device

    pl = pv.Plotter(off_screen=args.gif is not None)
    for side in ["right", "left"]:
        mano_layer = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=0,
            mano_assets_root=args.mano_assets_root,
            flat_hand_mean=True,
            use_pca=False,
        ).to(device)
        axis_layer = AxisLayerFK(side=side, mano_assets_root=args.mano_assets_root).to(device)

        hand_pose = axis_layer.compose(angles).reshape(1, 48)  # no global rotation
        mano_results = mano_layer(hand_pose)
        verts = mano_results.verts  # (B, 778, 3)
        faces = mano_layer.get_mano_closed_faces()
        T_g_a, _, _ = axis_layer(mano_results.transforms_abs)  # (B, 16, 4, 4)

        mesh = pv.wrap(Trimesh(verts[0].cpu().numpy(), faces.numpy(), process=False))
        pl.add_mesh(mesh, opacity=0.3, smooth_shading=True, color="orange" if side == "right" else "cyan")

        if args.mode == "axis":
            centers = T_g_a[0, :, :3, 3].cpu().numpy()  # (16, 3)
            axes = T_g_a[0, :, :3, :3].cpu().numpy()  # (16, 3, 3), columns: back, up, left
            for k, color in enumerate(["red", "green", "blue"]):
                pl.add_arrows(centers, axes[:, :, k], color=color, mag=0.02)
        else:
            anchors = AnchorLayer().to(device)(verts)[0].cpu().numpy()  # (32, 3)
            for anchor in anchors:
                pl.add_mesh(pv.Cube(center=anchor, x_length=3e-3, y_length=3e-3, z_length=3e-3), color="yellow")

    show_or_record(pl, args.gif)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["axis", "anchor"], default="axis", help="visualize the axes or the anchors")
    parser.add_argument("--gif", default=None, help="write an orbiting GIF here instead of opening a window")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random hand pose")
    parser.add_argument("--mano-assets-root", default="assets/mano")
    main(parser.parse_args())
