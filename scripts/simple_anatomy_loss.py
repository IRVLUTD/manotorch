"""Correct an anatomically implausible hand pose with the anatomy loss.

    uv run python scripts/simple_anatomy_loss.py                                # live window
    uv run python scripts/simple_anatomy_loss.py --gif doc/pose_correction.gif  # GIF, rendered off-screen

The index finger starts hyper-extended (negative bend at all three joints), twisted at its PIP joint and spread
at its DIP joint, all of which the anatomy limits forbid. Optimizing the finger axis-angles under
AnatomyConstraintLossEE brings every joint back into its range. The view looks along the bend axis; the arrows
are the anatomy aligned axes of the index finger: red = back (twist), green = up (spread), blue = left (bend).
"""

import argparse
from math import pi

import torch
import tqdm
from _common import AXIS_LEGEND, add_axes, add_hand, add_legend, compress_gif, get_device, new_plotter, open_gif

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer

#  transform order of right hand
#         15-14-13-\
#                   \
#    3-- 2 -- 1 -----0   < NOTE: demo on this finger
#   6 -- 5 -- 4 ----/
#   12 - 11 - 10 --/
#    9-- 8 -- 7 --/
INDEX_FINGER = [1, 2, 3]


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    mano_layer = ManoLayer(
        rot_mode="axisang",
        center_idx=9,
        mano_assets_root=args.mano_assets_root,
        use_pca=False,
        side="right",
        flat_hand_mean=True,
    ).to(device)
    axis_layer = AxisLayerFK(side="right", mano_assets_root=args.mano_assets_root).to(device)
    anatomy_loss = AnatomyConstraintLossEE()
    anatomy_loss.setup()
    faces = mano_layer.get_mano_closed_faces().numpy()

    # initial index finger, as (twist, spread, bend) per joint
    composed_ee = torch.zeros((1, 16, 3), device=device)
    composed_ee[:, 1] = torch.tensor([0, 0, -pi / 3])
    composed_ee[:, 2] = torch.tensor([pi / 3, 0, -pi / 3])
    composed_ee[:, 3] = torch.tensor([0, pi / 3, -pi / 3])
    articulation = axis_layer.compose(composed_ee)[:, 1:].clone().requires_grad_(True)  # (1, 15, 3)
    global_aa = torch.zeros((1, 1, 3), device=device)

    optimizer = torch.optim.Adam([articulation], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.iters // 5, gamma=0.5)

    def forward():
        out = mano_layer(torch.cat([global_aa, articulation], dim=1).reshape(1, 48))
        T_g_a, _, ee = axis_layer(out.transforms_abs)
        return out.verts[0], T_g_a[0], ee

    def draw(verts, T_g_a, it, loss):
        add_hand(pl, verts.detach().cpu().numpy(), faces, "right", opacity=0.7, name="hand")
        centers = T_g_a[INDEX_FINGER, :3, 3].detach().cpu().numpy()
        axes = T_g_a[INDEX_FINGER, :3, :3].detach().cpu().numpy()  # columns: back, up, left
        add_axes(pl, centers, axes, mag=0.025, name="axes")
        add_legend(pl, [(f"iteration {it:4d}   anatomy loss {loss:.5f}", "black"), AXIS_LEGEND])

    verts, T_g_a, ee_init = forward()
    ee = ee_init
    pl = new_plotter(off_screen=args.gif is not None, window_size=(768, 512))
    draw(verts, T_g_a, 0, anatomy_loss(ee_init).item())
    pl.camera_position = [(0.0, 0.0, 0.36), (0.0, 0.01, 0.0), (0.0, 1.0, 0.0)]
    if args.gif is not None:
        open_gif(pl, args.gif)
        pl.write_frame()
    else:
        pl.show(interactive_update=True)

    for it in (bar := tqdm.trange(1, args.iters + 1)):
        optimizer.zero_grad()
        verts, T_g_a, ee = forward()
        loss = anatomy_loss(ee)
        loss.backward()
        optimizer.step()
        scheduler.step()
        bar.set_description(f"anatomy loss: {loss.item():.5f}")
        if it % args.draw_every == 0 or it == args.iters:
            draw(verts, T_g_a, it, loss.item())
            pl.write_frame() if args.gif is not None else pl.update()

    deg = lambda e: torch.rad2deg(e[0, INDEX_FINGER]).detach().cpu().numpy().round(1)  # noqa: E731
    print(f"index finger (twist, spread, bend) in degrees, before:\n{deg(ee_init)}\nafter:\n{deg(ee)}")
    if args.gif is not None:
        pl.close()
        compress_gif(args.gif)
        print(f"saved {args.gif}")
    else:
        pl.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gif", default=None, help="write the optimization as a GIF instead of opening a window")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--draw-every", type=int, default=10, help="iterations between drawn frames")
    parser.add_argument("--mano-assets-root", default="assets/mano")
    main(parser.parse_args())
