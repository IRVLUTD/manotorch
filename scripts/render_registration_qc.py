"""Render opaque registration targets, manotorch reconstruction and vertex errors with matched cameras.

This display removes wrist translation/global rotation only after scanner-frame accuracy is measured.
Requires the vis extra; raw licensed meshes and PNGs stay under ignored data/qc/MANO_Poses.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from benchmark_registrations import MANO16, digest, numpy_rotations

from manotorch.manolayer import ManoLayer
from manotorch.utils.mano_io import find_mano_model, load_mano_pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/MANO_Poses/mano_poses_v1_0'))
    parser.add_argument('--mano-assets-root', type=Path, default=Path('assets/mano'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/qc/MANO_Poses/right'))
    parser.add_argument('--accuracy-csv', type=Path,
                        default=Path('data/benchmarks/mano_registrations/2026-10-08/accuracy_cuda_float32.csv'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--samples', type=int, default=6)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    files = sorted((args.dataset / 'handsOnly_REGISTRATIONS_r_lm___POSES').glob('*.pkl'))
    if not files:
        parser.error('No registration PKLs found')
    recs = [load_mano_pickle(p) for p in files]
    # Extremes plus evenly spaced registrations; add the measured worst case when available.
    indices = [int(np.argmax([np.abs(r['betas']).max() for r in recs])),
               int(np.argmax([np.linalg.norm(r['pose'][3:]) for r in recs]))]
    if args.accuracy_csv.exists():
        import csv
        with args.accuracy_csv.open() as f:
            rows = list(csv.DictReader(f))
        worst = max(rows, key=lambda r: float(r['vertex_max_euclidean_mm']))['file']
        indices.insert(0, next(i for i, p in enumerate(files) if p.name == worst))
    indices = list(dict.fromkeys(indices + np.linspace(0, len(files) - 1, args.samples).astype(int).tolist()))[:args.samples]
    selected = [recs[i] for i in indices]
    device = torch.device(args.device)
    layer = ManoLayer(side='right', use_pca=False, flat_hand_mean=True, center_idx=None,
                      mano_assets_root=str(args.mano_assets_root)).to(device)
    p, b, t = [torch.tensor(np.stack([r[key] for r in selected]), dtype=torch.float32, device=device)
               for key in ('pose', 'betas', 'trans')]
    with torch.inference_mode():
        out = layer(p, b, t)
    vertices = out.verts.cpu().double().numpy()
    joints = out.joints[:, MANO16].cpu().double().numpy()
    errors = np.stack([np.linalg.norm(vertices[k] - r['v'], axis=-1) * 1000 for k, r in enumerate(selected)])
    scale = max(float(errors.max()) * 1.05, 1e-6)
    faces = layer.th_faces.cpu().numpy()
    pv_faces = np.c_[np.full(len(faces), 3), faces].ravel()
    plotter = pv.Plotter(shape=(len(indices), 3), off_screen=True, window_size=(1800, 430 * len(indices)),
                         border=True, border_color='#D5DCE5')
    plotter.set_background('white', all_renderers=True)
    annotations = []
    # Proper display rotation to make the canonical hand upright; it does not change measured errors.
    display = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    for row, index in enumerate(indices):
        rec = recs[index]
        wrist = rec['J_transformed'][0]
        global_rotation = numpy_rotations(rec['pose'][:3])[0]
        targets = (rec['v'] - wrist) @ global_rotation @ display.T
        predicted = (vertices[row] - wrist) @ global_rotation @ display.T
        center = (targets.max(0) + targets.min(0)) / 2
        direction = np.array([.35, .12, 1.])
        camera = [tuple(center + direction), tuple(center), (0., 1., 0.)]
        # One fitted scale reused across columns; reserve upper space for annotations.
        radius = np.linalg.norm(targets - center, axis=1).max()
        for col, points in enumerate((targets, predicted, predicted)):
            plotter.subplot(row, col)
            mesh = pv.PolyData(points, pv_faces)
            title = ('Dataset target', 'manotorch reconstruction', 'Vertex error (mm)')[col]
            plotter.add_text(title, position='upper_left', font_size=15, color='#172B4D')
            plotter.add_text(files[index].name, position=(12, 375), font_size=11, color='#42526E')
            if col == 2:
                mesh['error_mm'] = errors[row]
                plotter.add_mesh(mesh, scalars='error_mm', cmap='viridis', clim=(0, scale), opacity=1.,
                                 smooth_shading=True, show_scalar_bar=False)
                plotter.add_text(f"max {errors[row].max():.6f} mm\nRMSE {np.sqrt(np.mean(errors[row] ** 2)):.6f} mm",
                                 position='lower_left', font_size=11, color='#172B4D')
                if row == len(indices) - 1:
                    plotter.add_scalar_bar(title='Euclidean error (mm)', n_labels=3, fmt='%.6f',
                                           title_font_size=12, label_font_size=10, color='#172B4D')
            else:
                plotter.add_mesh(mesh, color='#EFA37D' if col == 0 else '#80B6DA', opacity=1., smooth_shading=True)
                plotter.add_text('display: wrist centered, global rotation removed', position='lower_left',
                                 font_size=9, color='#42526E')
            plotter.camera_position = camera
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = radius * 1.35
            plotter.reset_camera_clipping_range()
        annotations.append({'file': files[index].name, 'index': index,
                            'source_sha256': digest(files[index]),
                            'max_vertex_euclidean_mm': float(errors[row].max()),
                            'vertex_rmse_mm': float(np.sqrt(np.mean(errors[row] ** 2))),
                            'max_joint_euclidean_mm': float(np.linalg.norm(joints[row] - rec['J_transformed'], axis=-1).max()*1000),
                            'max_abs_beta': float(np.abs(rec['betas']).max())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'registration_qc.png'
    plotter.screenshot(str(path))
    plotter.close()
    manifest = {'image': str(path), 'device': str(device), 'torch': torch.__version__,
                'model_sha256': digest(find_mano_model(str(args.mano_assets_root), 'right')),
                'script_sha256': digest(Path(__file__)), 'error_color_max_mm': scale,
                'reference_helper_sha256': digest(Path(__file__).with_name('benchmark_registrations.py')),
                'implementation_sha256': {str(p): digest(p) for p in sorted(Path('manotorch').rglob('*.py'))},
                'accuracy_frame': 'original scanner frame including translation',
                'display_frame': 'wrist centered, inverse global rotation, proper upright display rotation',
                'rows': annotations}
    (args.output_dir / 'registration_qc.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
