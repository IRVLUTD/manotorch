"""Visualize the anatomy aligned axes (or the anchors) of a random right hand and its left counterpart.

    uv run python scripts/simple_app.py                              # interactive window
    uv run python scripts/simple_app.py --gif doc/axis_new.gif       # camera-sweep GIF, rendered off-screen
    uv run python scripts/simple_app.py --mode anchor --gif doc/anchor.gif

The random pose is drawn in the anatomy aligned angle space, so it stays natural: each finger curls by a random
amount with coupled flexion of its three joints, the MCP joints spread a little, and nothing twists. Both hands
are composed from the same angles, so the left hand is the mirror of the right one. Display rotations place both
hands upright without changing MANO coordinates. Arrows: red = twist, green = spread, blue = bend.
Anchor mode draws 32 purple anchors per hand and highlights anchor 0 and its source triangle.
"""

import argparse

import pyvista as pv
import torch
from _common import (
    ANCHOR_COLOR,
    add_axes,
    add_hand,
    add_legend,
    display_axes,
    display_points,
    get_device,
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

        add_hand(pl, display_points(side, verts[0].cpu().numpy()), mano_layer.get_mano_closed_faces(), side,
                 opacity=0.65 if args.mode == "axis" else 0.85)
        if args.mode == "axis":
            centers = display_points(side, T_g_a[0, :, :3, 3].cpu().numpy())  # (16, 3)
            add_axes(pl, centers, display_axes(side, T_g_a[0, :, :3, :3].cpu().numpy()), mag=0.014)
        else:
            anchor_layer = AnchorLayer().to(device)
            anchors = display_points(side, anchor_layer(verts)[0].cpu().numpy())  # (32, 3)
            spheres = pv.PolyData(anchors).glyph(geom=pv.Sphere(radius=2.2e-3), orient=False, scale=False)
            pl.add_mesh(spheres, color=ANCHOR_COLOR)
            triangle = display_points(side, verts[0, anchor_layer.face_vert_idx[0, 0]].cpu().numpy())
            pl.add_mesh(pv.lines_from_points(triangle, close=True), color="#B77900", line_width=4)
            pl.add_mesh(pv.Sphere(radius=3e-3, center=anchors[0]), color="#FFD166")

    lines = hands_legend()
    add_legend(pl, lines if args.mode == "axis" else lines[:2] + [
        ("32 anchors per hand (purple)", ANCHOR_COLOR), ("anchor 0 and its source triangle highlighted", "#8C5B00")])
    show_or_record(pl, args.gif)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["axis", "anchor"], default="axis", help="visualize the axes or the anchors")
    parser.add_argument("--gif", default=None, help="write a camera-sweep GIF here instead of opening a window")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random hand pose")
    parser.add_argument("--mano-assets-root", default="assets/mano")
    main(parser.parse_args())
