"""Visualize the anatomy aligned axes (or the anchors) of a random right hand and its left counterpart.

    uv run python scripts/simple_app.py                              # interactive window
    uv run python scripts/simple_app.py --gif doc/axis_new.gif       # orbiting GIF, rendered off-screen
    uv run python scripts/simple_app.py --mode anchor --gif doc/anchor.gif

The random pose is drawn in the anatomy aligned angle space, so it stays natural: each finger curls by a random
amount with coupled flexion of its three joints, the MCP joints spread a little, and nothing twists. Both hands
are composed from the same angles, so the left hand is the mirror of the right one; they are drawn a few
centimeters apart. Arrows: red = back (twist), green = up (spread), blue = left (bend).
"""

import argparse

import pyvista as pv
import torch
from _common import (
    ANCHOR_COLOR,
    add_axes,
    add_hand,
    add_legend,
    get_device,
    hand_offset,
    hands_legend,
    new_plotter,
    show_or_record,
)

from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer


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


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    angles = random_natural_angles(args.seed).to(device)  # sampled on the CPU: the pose does not depend on the device

    pl = new_plotter(off_screen=args.gif is not None)
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
        T_g_a, _, _ = axis_layer(mano_results.transforms_abs)  # (B, 16, 4, 4)

        offset = hand_offset(side)
        add_hand(pl, verts[0].cpu().numpy() + offset, mano_layer.get_mano_closed_faces(), side)
        if args.mode == "axis":
            centers = T_g_a[0, :, :3, 3].cpu().numpy() + offset  # (16, 3)
            add_axes(pl, centers, T_g_a[0, :, :3, :3].cpu().numpy())  # columns: back, up, left
        else:
            anchors = AnchorLayer().to(device)(verts)[0].cpu().numpy() + offset  # (32, 3)
            spheres = pv.PolyData(anchors).glyph(geom=pv.Sphere(radius=1.8e-3), orient=False, scale=False)
            pl.add_mesh(spheres, color=ANCHOR_COLOR)

    lines = hands_legend()
    add_legend(pl, lines if args.mode == "axis" else lines[:2] + [("anchors", ANCHOR_COLOR)])
    show_or_record(pl, args.gif)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["axis", "anchor"], default="axis", help="visualize the axes or the anchors")
    parser.add_argument("--gif", default=None, help="write an orbiting GIF here instead of opening a window")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random hand pose")
    parser.add_argument("--mano-assets-root", default="assets/mano")
    main(parser.parse_args())
