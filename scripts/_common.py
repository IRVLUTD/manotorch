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
# Displayed hands are moved apart by 2 x HAND_GAP along screen x so their wrists do not overlap.
HAND_GAP = 0.10
GIF_SIZE = 960


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def new_plotter(off_screen: bool, window_size=(GIF_SIZE, 640)) -> pv.Plotter:
    pl = pv.Plotter(off_screen=off_screen, window_size=window_size)
    pl.enable_depth_peeling(number_of_peels=8)  # draw the translucent surfaces in depth order
    pl.enable_anti_aliasing("ssaa")
    return pl


def hand_offset(side: str) -> np.ndarray:
    """Translation that separates the displayed hands, see HAND_GAP."""
    return np.array([-HAND_GAP if side == "right" else HAND_GAP, 0.0, 0.0])


def display_rotation(side: str) -> np.ndarray:
    """Rigid display rotations: both hands point up and remain mirrored across screen x.

    MANO's fingers run along opposite x directions. These proper rotations only
    affect the illustration, leaving model coordinates and anatomical angles unchanged.
    """
    sign = 1.0 if side == "right" else -1.0
    return np.array([[0.0, 0.0, sign], [-sign, 0.0, 0.0], [0.0, -1.0, 0.0]])


def display_points(side: str, points) -> np.ndarray:
    return np.asarray(points) @ display_rotation(side).T + hand_offset(side)


def display_axes(side: str, axes) -> np.ndarray:
    return display_rotation(side) @ np.asarray(axes)


def add_hand(pl: pv.Plotter, verts, faces, side: str, opacity: float = 0.6, name: str | None = None,
             highlight=None):
    """Draw a smooth closed hand, optionally highlighting selected vertices.

    Back-face culling avoids dark patches from inner faces when opacity is below one.
    """
    mesh = pv.wrap(Trimesh(np.asarray(verts), np.asarray(faces), process=False))
    kwargs = {"color": HAND_COLORS[side]}
    if highlight is not None:
        colors = np.tile(pv.Color(HAND_COLORS[side]).int_rgb, (mesh.n_points, 1))
        colors[np.asarray(highlight)] = pv.Color("#FFD166").int_rgb
        mesh["surface_colors"] = colors.astype(np.uint8)
        kwargs = {"scalars": "surface_colors", "rgb": True}
    pl.add_mesh(mesh, opacity=opacity, smooth_shading=True, culling="back", name=name, **kwargs)


def add_axes(pl: pv.Plotter, centers, axes, mag: float = 0.02, name: str | None = None):
    """Draw thin arrows along the three columns of `axes` (N, 3, 3), starting at `centers` (N, 3)."""
    arrow = pv.Arrow(tip_length=0.25, tip_radius=0.09, shaft_radius=0.035)
    for k, color in enumerate(AXIS_COLORS):
        points = pv.PolyData(np.asarray(centers, dtype=float))
        points["vectors"] = np.asarray(axes, dtype=float)[:, :, k]
        glyphs = points.glyph(orient="vectors", scale=False, factor=mag, geom=arrow)
        pl.add_mesh(glyphs, color=color, lighting=False, name=None if name is None else f"{name}_{k}")


def add_legend(pl: pv.Plotter, lines):
    """Write `(text, color)` lines in the upper left corner."""
    height = pl.window_size[1]
    for i, (text, color) in enumerate(lines):
        y = height - 36 - 28 * i
        if text == AXIS_LEGEND[0]:
            for k, (label, axis_color, x) in enumerate(zip(
                    ("twist (red)", "spread (green)", "bend (blue)"), AXIS_COLORS, (18, 220, 485), strict=True)):
                pl.add_text(label, position=(x, y), color=axis_color, font_size=14, name=f"legend_{i}_{k}")
        else:
            pl.add_text(text, position=(18, y), color=color, font_size=14, name=f"legend_{i}")


def hands_legend(sides=("right", "left")):
    return [(f"{side} hand", LABEL_COLORS[side]) for side in sides] + [AXIS_LEGEND]


def open_gif(pl: pv.Plotter, gif: str, fps: int = 10):
    pl.open_gif(gif, fps=fps)


def compress_gif(gif: str, colors: int = 128):
    """Re-encode with one shared palette and preserve thin annotation colors without dithering."""
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
    # Reserve annotation colors; quantizing mostly skin/background otherwise loses thin RGB arrows.
    reserved = [*AXIS_COLORS, *HAND_COLORS.values(), *LABEL_COLORS.values(), ANCHOR_COLOR,
                "#FFD166", "#8C5B00", "#000000", "#FFFFFF"]
    base_colors = colors - len(reserved)
    palette = montage.quantize(colors=base_colors, method=Image.Quantize.MEDIANCUT)
    entries = palette.getpalette()
    for i, color in enumerate(reserved):
        entries[(base_colors + i) * 3:(base_colors + i + 1) * 3] = pv.Color(color).int_rgb
    palette.putpalette(entries)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)


def show_or_record(pl: pv.Plotter, gif: str | None, oblique: float = 0.25, sweep: float = 25.0):
    """Fit a large orthographic view and gently sweep it, avoiding edge-on frames."""
    bounds = np.asarray(pl.bounds).reshape(3, 2)
    center = bounds.mean(1)
    center[1] += (bounds[1, 1] - bounds[1, 0]) * 0.12  # Reserve space above the meshes for the legend.
    corners = np.array(np.meshgrid(*bounds, indexing="ij")).reshape(3, -1).T - center
    angles = np.arctan(oblique) + np.deg2rad(sweep) * np.sin(np.linspace(0, 2 * np.pi, 48))
    directions = np.array([np.sin(angles), np.full_like(angles, 0.12), np.cos(angles)]).T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    aspect = pl.window_size[0] / pl.window_size[1]
    scale = 0.0
    for direction in directions:
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        vertical = np.cross(direction, right)
        scale = max(scale, np.abs(corners @ vertical).max(), np.abs(corners @ right).max() / aspect)
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = scale * 1.10

    def camera(direction):
        pl.camera_position = [tuple(center + direction), tuple(center), tuple(up)]
        pl.reset_camera_clipping_range()

    camera(directions[0])
    if gif is None:
        pl.add_camera_orientation_widget()
        pl.show()
        return
    open_gif(pl, gif)
    for direction in directions:
        camera(direction)
        pl.write_frame()
    pl.close()
    compress_gif(gif)
    print(f"saved {gif}")
