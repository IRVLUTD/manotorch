"""Rendering helpers shared by the demo scripts: device choice, colors, hands, axes, legend and GIF recording."""

import numpy as np
import pyvista as pv
import torch
from PIL import Image
from trimesh import Trimesh

# A light warm / light cool pair (ColorBrewer RdBu), readable by color-blind viewers, so the axes stand out.
HAND_COLORS = {"right": "#F4A582", "left": "#86B9DF"}
LABEL_COLORS = {"right": "#C8643C", "left": "#3A7CA5"}
# Anatomy aligned axes in the usual x/y/z = red/green/blue order: back (twist), up (spread), left (bend).
AXIS_COLORS = ("#D62728", "#2CA02C", "#1F5FB4")
AXIS_LEGEND = ("axes: red = twist, green = spread, blue = bend", "dimgray")
ANCHOR_COLOR = "#7B3294"
# The two mirrored hands are moved apart by 2 x HAND_GAP along x, the mirror axis, so their wrists do not overlap.
HAND_GAP = 0.02
GIF_SIZE = 768
# Orbit radius of the GIFs, relative to the scene size; leaves a margin for the legend.
ORBIT_FACTOR = 2.6


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def new_plotter(off_screen: bool, window_size=(GIF_SIZE, GIF_SIZE)) -> pv.Plotter:
    pl = pv.Plotter(off_screen=off_screen, window_size=window_size)
    pl.enable_depth_peeling(number_of_peels=8)  # draw the translucent surfaces in depth order
    pl.enable_anti_aliasing("ssaa")
    return pl


def hand_offset(side: str) -> np.ndarray:
    """Translation that moves a hand away from the mirror plane, see HAND_GAP."""
    return np.array([-HAND_GAP if side == "right" else HAND_GAP, 0.0, 0.0])


def add_hand(pl: pv.Plotter, verts, faces, side: str, opacity: float = 0.6, name: str | None = None):
    """Translucent hand mesh (closed faces). Only the faces turned to the camera are drawn: through a translucent
    skin, the inner faces (back of the palm, other fingers, wrist cap) would otherwise show as dark patches."""
    mesh = pv.wrap(Trimesh(np.asarray(verts), np.asarray(faces), process=False))
    pl.add_mesh(mesh, color=HAND_COLORS[side], opacity=opacity, smooth_shading=True, culling="back", name=name)


def add_axes(pl: pv.Plotter, centers, axes, mag: float = 0.02, name: str | None = None):
    """Draw thin arrows along the three columns of `axes` (N, 3, 3), starting at `centers` (N, 3)."""
    arrow = pv.Arrow(tip_length=0.25, tip_radius=0.09, shaft_radius=0.035)
    for k, color in enumerate(AXIS_COLORS):
        points = pv.PolyData(np.asarray(centers, dtype=float))
        points["vectors"] = np.asarray(axes, dtype=float)[:, :, k]
        glyphs = points.glyph(orient="vectors", scale=False, factor=mag, geom=arrow)
        pl.add_mesh(glyphs, color=color, name=None if name is None else f"{name}_{k}")


def add_legend(pl: pv.Plotter, lines):
    """Write `(text, color)` lines in the upper left corner."""
    height = pl.window_size[1]
    for i, (text, color) in enumerate(lines):
        pl.add_text(text, position=(12, height - 28 - 20 * i), color=color, font_size=9, name=f"legend_{i}")


def hands_legend(sides=("right", "left")):
    return [(f"{side} hand", LABEL_COLORS[side]) for side in sides] + [AXIS_LEGEND]


def open_gif(pl: pv.Plotter, gif: str, fps: int = 10):
    pl.open_gif(gif, fps=fps)


def compress_gif(gif: str, colors: int = 96):
    """Re-encode `gif` with one palette shared by all frames and no dithering: about half the size."""
    with Image.open(gif) as im:
        duration = im.info.get("duration", 100)
        frames = []
        for k in range(im.n_frames):
            im.seek(k)
            frames.append(im.convert("RGB"))
    picks = frames[:: max(1, len(frames) // 8)]
    montage = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)))
    for i, frame in enumerate(picks):
        montage.paste(frame, (0, i * frame.height))
    palette = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)


def show_or_record(pl: pv.Plotter, gif: str | None):
    """Open an interactive window, or write an orbit around the scene to `gif`."""
    if gif is None:
        pl.add_camera_orientation_widget()
        pl.show()
        return
    view_up = (-1.0, 0.0, 0.0)
    path = pl.generate_orbital_path(factor=ORBIT_FACTOR, n_points=36, viewup=view_up, shift=0.1)
    focus = tuple(pl.center)
    open_gif(pl, gif)
    # set the camera explicitly: Plotter.orbit_on_path refits the view, so its orbit radius has no effect
    for point in path.points:
        pl.camera_position = [tuple(point), focus, view_up]
        pl.write_frame()
    pl.close()
    compress_gif(gif)
    print(f"saved {gif}")
