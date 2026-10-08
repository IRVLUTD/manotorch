"""Render every MANO registration into an A3, six-hand/three-column QC PDF.

Run from the repository root with the vis extra and optional ReportLab dependency.
MANO inference and VTK/OpenGL rendering have independent devices: --require-gpu-render
rejects a software OpenGL context. Metrics stay in the original scanner frame.
The PDF uses one global error scale, native labels, subject bookmarks and a linked index.
"""

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyvista as pv
from _registration_geometry import digest, numpy_rotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-archive', type=Path, help='Cached poses/targets/reconstructions NPZ; no Torch or original dataset needed')
    parser.add_argument('--dataset', type=Path, default=Path('data/MANO_Poses/mano_poses_v1_0'))
    parser.add_argument('--mano-assets-root', type=Path, default=Path('assets/mano'))
    parser.add_argument('--output-dir', type=Path, help='Side-result directory; default data/qc/MANO_Poses/SIDE')
    parser.add_argument('--pdf-path', type=Path, help='PDF destination; indices, manifest and PNGs stay under output-dir')
    parser.add_argument('--accuracy-csv', type=Path,
                        default=Path('data/benchmarks/mano_registrations/2026-10-08/accuracy_cuda_float32.csv'))
    parser.add_argument('--device', default=None, help='Inference device for original PKLs; unused for cached archives')
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--require-gpu-render', action='store_true')
    parser.add_argument('--max-pages', type=int, help='Limit grid pages for a separate smoke output directory')
    parser.add_argument('--export-pages', type=int, nargs='*', default=None,
                        help='Export numbered grid pages as PNG; default first/middle/last and measured extremes')
    args = parser.parse_args()
    if min(args.chunk, args.threads) < 1 or (args.max_pages is not None and args.max_pages < 1):
        parser.error('chunk, threads and max-pages must be positive')
    from PIL import Image
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A3
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    started = time.perf_counter()
    source_script_hash = digest(Path(__file__))
    archive = None
    if args.pose_archive:
        with np.load(args.pose_archive, allow_pickle=False) as data:
            archive = {k: data[k] for k in data.files}
        metadata = json.loads(str(archive['metadata']))
        side = str(archive['model_side'])
        if side not in ('right', 'left'):
            raise ValueError('Archive model_side must be right or left')
        names = archive['names'].tolist()
        files = [Path(n) for n in names]  # Identifiers, not original PKL dependencies.
        if not names or names != sorted(set(names)):
            raise ValueError('Archive names must be nonempty, unique and sorted')
        recs = [{k: archive[k][i] for k in ('pose', 'betas', 'trans', 'v', 'J_transformed')}
                for i in range(len(names))]
        predicted, joint_predictions = archive['predicted_vertices'], archive['predicted_joints']
        targets = archive['v']
        expected_shapes = {'pose': (len(names), 48), 'betas': (len(names), 10),
                           'trans': (len(names), 3), 'v': (len(names), 778, 3),
                           'J_transformed': (len(names), 16, 3),
                           'predicted_vertices': (len(names), 778, 3),
                           'predicted_joints': (len(names), 16, 3),
                           'reference_vertices': (len(names), 778, 3),
                           'reference_joints': (len(names), 16, 3)}
        for key, shape in expected_shapes.items():
            if archive[key].shape != shape or not np.isfinite(archive[key]).all():
                raise ValueError(f'Invalid archive {key}: expected finite {shape}')
        ref_vmax = float(np.linalg.norm(predicted - archive['reference_vertices'], axis=-1).max() * 1000)
        ref_jmax = float(np.linalg.norm(joint_predictions - archive['reference_joints'], axis=-1).max() * 1000)
        if max(ref_vmax, ref_jmax) > .001:
            raise ValueError('Cached MANO vs independent NumPy reference exceeds 0.001 mm')
        device = metadata['device']
        torch_version = metadata['torch']
        inference_chunk = metadata['chunk']
        faces = archive['faces']
        source_hashes = archive['source_sha256'].tolist()
        model_hash = metadata['model_sha256']
        implementation_hashes = metadata['implementation_sha256']
    else:
        import torch
        from benchmark_registrations import MANO16

        from manotorch.manolayer import ManoLayer
        from manotorch.utils.mano_io import find_mano_model, load_mano_pickle

        torch.set_num_threads(args.threads)
        files = sorted((args.dataset / 'handsOnly_REGISTRATIONS_r_lm___POSES').glob('*.pkl'))
        if not files:
            parser.error('No registration PKLs found')
        recs = [load_mano_pickle(p) for p in files]
        names = [p.name for p in files]
        side = 'right'
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        layer = ManoLayer(side=side, use_pca=False, flat_hand_mean=True, center_idx=None,
                          mano_assets_root=str(args.mano_assets_root)).to(device)
        predicted, joint_predictions = [], []
        with torch.no_grad():
            for start in range(0, len(recs), args.chunk):
                batch = recs[start:start + args.chunk]
                inputs = [torch.tensor(np.stack([r[key] for r in batch]), dtype=torch.float32, device=device)
                          for key in ('pose', 'betas', 'trans')]
                out = layer(*inputs)
                if not torch.isfinite(out.verts).all() or not torch.isfinite(out.joints).all():
                    raise ValueError('Nonfinite MANO reconstruction')
                predicted.append(out.verts.cpu().double().numpy())
                joint_predictions.append(out.joints[:, MANO16].cpu().double().numpy())
        predicted = np.concatenate(predicted)
        joint_predictions = np.concatenate(joint_predictions)
        targets = np.stack([r['v'] for r in recs])
        faces = layer.th_faces.cpu().numpy()
        source_hashes = [digest(p) for p in files]
        torch_version = torch.__version__
        inference_chunk = args.chunk
        model_hash = digest(find_mano_model(str(args.mano_assets_root), side))
        implementation_hashes = {str(p): digest(p) for p in sorted(Path('manotorch').rglob('*.py'))}
        ref_vmax = ref_jmax = None
    total_pages = math.ceil(len(files) / 6)
    pages = min(args.max_pages or total_pages, total_pages)
    if args.export_pages is not None and any(p < 1 or p > pages for p in args.export_pages):
        parser.error('export-pages must refer to rendered grid pages')
    errors = np.linalg.norm(predicted - targets, axis=-1) * 1000
    joint_errors = np.linalg.norm(joint_predictions - np.stack([r['J_transformed'] for r in recs]), axis=-1) * 1000
    vmax, rmses = errors.max(1), np.sqrt((errors ** 2).mean(1))
    jmax = joint_errors.max(1)
    scale = max(float(vmax.max()) * 1.05, 1e-6)
    csv_delta = None
    if archive is None and args.accuracy_csv.exists():
        with args.accuracy_csv.open(newline='') as stream:
            reference = list(csv.DictReader(stream))
        if sorted(x['file'] for x in reference) != names:
            raise ValueError('Accuracy CSV must cover the same unique registrations')
        reference = {x['file']: x for x in reference}
        expected = np.array([[float(reference[n][k]) for k in
                             ('vertex_max_euclidean_mm', 'vertex_rmse_mm', 'joint_max_euclidean_mm')] for n in names])
        measured = np.stack((vmax, rmses, jmax), axis=1)
        csv_delta = float(np.abs(measured - expected).max())
        if csv_delta > 1e-7:
            raise ValueError(f'Atlas vs accuracy CSV metric discrepancy: {csv_delta} mm')
    target_threshold = .05 if side == 'left' else .001
    if max(float(vmax.max()), float(jmax.max())) > target_threshold:
        raise ValueError(f'Atlas target residual exceeds {target_threshold} mm')

    if args.output_dir is None:
        args.output_dir = Path('data/qc/MANO_Poses') / side
        if args.pdf_path is None:
            args.pdf_path = args.output_dir.parent / f'registration_atlas_{side}.pdf'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.pdf_path or args.output_dir / 'registration_atlas.pdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial.pdf')
    pdf = canvas.Canvas(str(partial), pagesize=A3, pageCompression=1)
    pdf.setTitle(f'MANO_Poses {side}-hand complete registration QC atlas')
    pdf.setAuthor('manotorch')
    width, height = A3
    margin = 28
    content_width = width - 2 * margin
    col_width = content_width / 3
    row_height = 162
    bottom = 88
    grid_height = 6 * row_height
    grid_top = bottom + grid_height
    subjects = Counter(n.split('_')[0] for n in names)
    first_indices = {s: next(i for i, n in enumerate(names) if n.startswith(s + '_')) for s in subjects}
    rendered_count = min(pages * 6, len(files))
    ink, muted = colors.HexColor('#17374F'), colors.HexColor('#526577')

    def text(x, y, value, size=10, bold=False, color=ink):
        pdf.setFillColor(color)
        pdf.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
        pdf.drawString(x, y, value)

    def footer(number):
        text(margin, 18, 'manotorch | MANO_Poses | local licensed-data QC', 9, color=muted)
        pdf.setFont('Helvetica', 9)
        pdf.drawRightString(width - margin, 18, f'PDF page {number}')

    pdf.bookmarkPage('cover')
    pdf.addOutlineEntry('About this atlas', 'cover', level=0)
    text(margin, height - 65, 'MANO_Poses', 30, bold=True)
    text(margin, height - 102, f'{side.capitalize()}-hand registration QC atlas', 23, bold=True)
    lines = [
        f'{len(files):,} registrations | {len(subjects)} subjects | {total_pages} six-hand grid pages',
        (f'{len(files):,} derived left-hand poses, mirrored from uniform MANO_RIGHT registrations'
         if side == 'left' else f'{sum("mirrored" not in n for n in names)} right; {sum("mirrored" in n for n in names)} left mirrored into MANO_RIGHT'),
        f'Rendered coverage: {rendered_count:,}/{len(files):,} hands ({pages}/{total_pages} grid pages)',
        ('Columns: mirrored right target / corrected MANO_LEFT / mirror residual' if side == 'left'
         else 'Columns: dataset target / manotorch reconstruction / Euclidean vertex error'),
        'A3 portrait, opaque surfaces, matched orthographic cameras and native PDF labels.',
        'Alphabetical registration order. Subject and worst-error index links jump to grid pages.',
        '',
        'Measurement: original scanner-frame pose, shape and translation; errors in millimetres.',
        'Display only: wrist centered, global rotation removed, side-specific upright rotation.',
        'The display transformation does not change the measured errors.',
        f'One global error scale on every page: 0 to {scale:.8f} mm.',
        f'Maximum vertex / 16-joint error: {vmax.max():.8f} / {jmax.max():.8f} mm.',
        (f'Independent NumPy reference max vertex / joint: {ref_vmax:.8f} / {ref_jmax:.8f} mm.'
         if side == 'left' else 'All registrations pass the 0.001 mm numerical reconstruction threshold.'),
        f'MANO inference: {device}, float32, chunk {inference_chunk}, PyTorch {torch_version}.',
        '',
        ('Targets are geometric mirrors, not independent left-hand scan registrations.' if side == 'left'
         else 'These are original MANO registration targets, not measured scan-fit quality.'),
        ('Left/right model assets are approximate mirrors; fix_left_shapedirs=True is enabled.' if side == 'left'
         else 'Anatomical plausibility and image-based fitting are separate evaluations.'),
        'Synthetic sequences are not included: they lack original registration mesh targets.',
        '',
        'Companion files: qc_summary.pdf, registration_atlas_index.csv, registration_atlas.json.',
    ]
    for i, line in enumerate(lines):
        text(margin, height - 151 - i * 24, line, 12)
    text(margin, 210, 'Pose archive:' if archive is not None else 'Dataset directory:', 11, bold=True)
    text(margin, 190, str(args.pose_archive or args.dataset), 10, color=muted)
    footer(1)
    pdf.showPage()

    pdf.bookmarkPage('index')
    pdf.addOutlineEntry('Subjects and worst-error index', 'index', level=0)
    text(margin, height - 55, 'Navigation index', 23, bold=True)
    text(margin, height - 80, 'Grid page 1 is PDF page 3. Click a row or use the PDF bookmarks.', 11)
    text(margin, height - 119, 'Subject / hands / first grid page', 14, bold=True)
    right_x = width / 2 + 10
    text(right_x, height - 119, 'Top 20 vertex errors / mm / grid page', 14, bold=True)
    for i, subject in enumerate(sorted(subjects)):
        grid = first_indices[subject] // 6 + 1
        y = height - 150 - 21 * i
        label = f'Subject {subject}     {subjects[subject]:3d} hands     grid {grid}'
        text(margin, y, label, 11)
        if grid <= pages:
            pdf.linkAbsolute('', f'subject_{subject}', (margin, y - 3, width / 2 - 15, y + 14))
            pdf.addOutlineEntry(f'Subject {subject} ({subjects[subject]} hands)', f'subject_{subject}', level=1)
    for rank, index in enumerate(np.argsort(vmax)[::-1][:20]):
        grid = int(index) // 6 + 1
        y = height - 150 - 21 * rank
        text(right_x, y, names[index], 10)
        text(right_x + 188, y, f'{vmax[index]:.7f}', 10)
        text(right_x + 295, y, str(grid), 10)
        if grid <= pages:
            pdf.linkAbsolute('', f'grid_{grid:03d}', (right_x, y - 3, width - margin, y + 14))
    text(right_x, height - 625, 'The complete filename/page/row mapping is in', 11)
    text(right_x, height - 648, 'registration_atlas_index.csv.', 11)
    footer(2)
    pdf.showPage()

    pv_faces = np.c_[np.full(len(faces), 3), faces].ravel()
    plotter = pv.Plotter(shape=(6, 3), off_screen=True, border=False,
                         window_size=(1800, round(1800 * grid_height / content_width)))
    display = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    if side == 'left':
        # Reflection in model x becomes a vertical reflection after the right-hand display basis.
        # A proper 180-degree rotation about display z puts fingers up and the thumb opposite the right.
        display = np.diag([-1., -1., 1.]) @ display
    if not np.allclose(display @ display.T, np.eye(3)) or not np.isclose(np.linalg.det(display), 1.):
        raise ValueError('Display transformation must be a proper rotation preserving handedness')
    opengl = {}
    minimum_clearance, maximum_bbox_delta = 10**9, 0
    export_default = {1, (pages + 1) // 2, pages,
                      int(vmax.argmax()) // 6 + 1,
                      int(np.argmax([np.abs(r['betas']).max() for r in recs])) // 6 + 1,
                      int(np.argmax([np.linalg.norm(r['pose'][3:]) for r in recs])) // 6 + 1}
    export = sorted({p for p in (export_default if args.export_pages is None else args.export_pages) if p <= pages})
    try:
        for grid in range(1, pages + 1):
            plotter.clear()
            plotter.enable_lightkit()  # clear() removes lights as well as actors.
            plotter.set_background('white', all_renderers=True)
            indices = range((grid - 1) * 6, min(grid * 6, len(files)))
            for row, index in enumerate(indices):
                rec = recs[index]
                rotation = numpy_rotations(rec['pose'][:3])[0]
                target = (rec['v'] - rec['J_transformed'][0]) @ rotation @ display.T
                pred = (predicted[index] - rec['J_transformed'][0]) @ rotation @ display.T
                center = (target.min(0) + target.max(0)) / 2
                radius = np.linalg.norm(target - center, axis=1).max()
                camera = [tuple(center + [.35, .12, 1.]), tuple(center), (0., 1., 0.)]
                for col, points in enumerate((target, pred, pred)):
                    plotter.subplot(row, col)
                    mesh = pv.PolyData(points, pv_faces)
                    if col == 2:
                        mesh['error_mm'] = errors[index]
                        plotter.add_mesh(mesh, scalars='error_mm', clim=(0, scale), cmap='viridis',
                                         smooth_shading=True, opacity=1., show_scalar_bar=False, render=False)
                    else:
                        plotter.add_mesh(mesh, color='#EFA37D' if col == 0 else '#80B6DA',
                                         smooth_shading=True, opacity=1., render=False)
                    plotter.camera_position = camera
                    plotter.enable_parallel_projection()
                    plotter.camera.parallel_scale = radius * 1.35
                    plotter.reset_camera_clipping_range()
            # screenshot() renders automatically only once; subsequent pages need an explicit render
            # after ALL camera updates, including the last viewport, rather than the last add_mesh().
            plotter.render()
            pixels = plotter.screenshot(return_img=True)
            panel_height = pixels.shape[0] // 6
            for row in range(len(indices)):
                boxes = []
                for col in range(3):
                    panel = pixels[row*panel_height:(row+1)*panel_height, col*600:(col+1)*600, :3]
                    mask = (np.ptp(panel, axis=-1) > 20) & (panel.min(-1) < 230)
                    if mask.sum() < 3000:
                        raise RuntimeError(f'Empty model panel at grid {grid}, row {row+1}, column {col+1}')
                    yy, xx = np.where(mask)
                    box = np.array([xx.min(), yy.min(), xx.max(), yy.max()])
                    clearance = min(yy.min(), panel_height-1-yy.max(), xx.min(), 599-xx.max())
                    if clearance < 20:
                        raise RuntimeError(f'Clipped model panel at grid {grid}, row {row+1}, column {col+1}')
                    minimum_clearance = min(minimum_clearance, int(clearance))
                    boxes.append(box)
                delta = int(np.abs(np.stack(boxes) - boxes[0]).max())
                if delta > 3:
                    raise RuntimeError(f'Unmatched camera/image bounds at grid {grid}, row {row+1}: {delta} pixels')
                maximum_bbox_delta = max(maximum_bbox_delta, delta)
            if grid == 1:
                capabilities = plotter.ren_win.ReportCapabilities()
                opengl = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                          for line in capabilities.splitlines() if line.startswith('OpenGL ') and 'string:' in line}
                renderer = opengl.get('OpenGL renderer string', '')
                if args.require_gpu_render and (not renderer or any(x in renderer.lower() for x in
                                                                    ('llvmpipe', 'softpipe', 'software', 'swiftshader'))):
                    raise RuntimeError(f'GPU OpenGL renderer required; got {renderer!r}')
                print('OpenGL:', json.dumps(opengl), flush=True)
            jpeg = io.BytesIO()
            Image.fromarray(pixels).save(jpeg, format='JPEG', quality=94, subsampling=0)
            jpeg.seek(0)
            pdf.bookmarkPage(f'grid_{grid:03d}')
            text(margin, height - 45, f'MANO_Poses | {side.capitalize()}-hand QC', 21, bold=True)
            text(margin, height - 70, f'Grid {grid}/{total_pages} | registrations {(grid-1)*6+1}-{min(grid*6,len(files))}', 11)
            titles = (('Mirrored right target', 'MANO_LEFT (corrected)', 'Mirror residual (mm)') if side == 'left'
                      else ('Dataset target', 'manotorch reconstruction', 'Vertex error (mm)'))
            for col, title in enumerate(titles):
                pdf.setFillColor(ink)
                pdf.rect(margin + col * col_width, grid_top + 15, col_width - 1, 25, fill=1, stroke=0)
                text(margin + col * col_width + 8, grid_top + 23, title, 12, bold=True, color=colors.white)
            pdf.drawImage(ImageReader(jpeg), margin, bottom, width=content_width, height=grid_height)
            for row, index in enumerate(indices):
                top = grid_top - row * row_height
                for col in range(3):
                    text(margin + col * col_width + 8, top - 14, names[index], 10, bold=True)
                text(margin + 2 * col_width + 8, top - row_height + 25, f'max {vmax[index]:.8f} mm', 9)
                text(margin + 2 * col_width + 8, top - row_height + 12, f'RMSE {rmses[index]:.8f} mm', 9)
                subject = names[index].split('_')[0]
                if first_indices[subject] == index:
                    pdf.bookmarkHorizontalAbsolute(f'subject_{subject}', top)
            pdf.setStrokeColor(colors.HexColor('#CDD7E0'))
            pdf.setLineWidth(.5)
            for row in range(7):
                pdf.line(margin, bottom + row * row_height, width - margin, bottom + row * row_height)
            for col in range(4):
                pdf.line(margin + col * col_width, bottom, margin + col * col_width, grid_top)
            # Vector colour legend and text stay legible independently of raster resolution.
            from matplotlib import colormaps
            cmap = colormaps['viridis']
            legend_x, legend_y, legend_w = margin, 50, 260
            for i in range(100):
                pdf.setFillColorRGB(*cmap(i / 99)[:3])
                pdf.rect(legend_x + i * legend_w / 100, legend_y, legend_w / 100 + .2, 9, fill=1, stroke=0)
            for i, value in enumerate((0., scale / 2, scale)):
                text(legend_x + i * legend_w / 2, legend_y + 14, f'{value:.8f}', 9)
            legend = 'Euclidean mirror residual (mm)' if side == 'left' else 'Euclidean vertex error (mm)'
            text(legend_x + legend_w + 55, legend_y + 2, legend + ', global scale', 10)
            text(margin, 32, 'Display only: wrist centered, global rotation removed; errors measured in original scanner frame.', 9)
            footer(grid + 2)
            pdf.showPage()
            if grid == 1 or grid % 10 == 0 or grid == pages:
                progress = {'completed_grid_pages': grid, 'total_grid_pages': pages,
                            'elapsed_seconds': time.perf_counter() - started}
                (args.output_dir / 'registration_atlas_progress.json').write_text(json.dumps(progress, indent=2) + '\n')
                print('progress', progress, flush=True)
        pdf.save()
    finally:
        plotter.close()
    partial.replace(path)
    ranks = {int(index): rank + 1 for rank, index in enumerate(np.argsort(vmax)[::-1])}
    index_rows = [{'file': names[i], 'subject': names[i].split('_')[0], 'grid_page': i // 6 + 1,
                   'pdf_page': i // 6 + 3, 'row': i % 6 + 1, 'error_rank': ranks[i],
                   'vertex_max_euclidean_mm': float(vmax[i]), 'vertex_rmse_mm': float(rmses[i]),
                   'joint_max_euclidean_mm': float(jmax[i]), 'source_sha256': source_hashes[i]}
                  for i in range(rendered_count)]
    index_path = args.output_dir / 'registration_atlas_index.csv'
    with index_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'path_base': 'manifest_directory',
                'pdf': os.path.relpath(path.resolve(), args.output_dir.resolve()), 'pdf_sha256': digest(path),
                'complete': rendered_count == len(files), 'dataset_samples': len(files), 'rendered_samples': rendered_count,
                'grid_pages': pages, 'cover_index_pages': 2, 'pdf_pages': pages + 2, 'hands_per_page': 6,
                'index_csv': index_path.name, 'index_sha256': digest(index_path), 'error_color_max_mm': scale,
                'vertex_max_euclidean_mm': float(vmax.max()), 'joint_max_euclidean_mm': float(jmax.max()),
                'accuracy_csv_max_metric_delta_mm': csv_delta, 'device': str(device), 'torch': torch_version, 'model_side': side, 'inference_chunk': inference_chunk,
                'reference_vertex_max_euclidean_mm': ref_vmax, 'reference_joint_max_euclidean_mm': ref_jmax,
                'pose_archive': os.path.relpath(args.pose_archive.resolve(), args.output_dir.resolve())
                if args.pose_archive else None,
                'pose_archive_sha256': digest(args.pose_archive) if args.pose_archive else None,
                'opengl': opengl, 'script_sha256': source_script_hash,
                'helper_sha256': digest(Path(__file__).with_name('_registration_geometry.py')),
                'model_sha256': model_hash,
                'implementation_sha256': implementation_hashes,
                'accuracy_frame': 'original scanner frame including translation',
                'renderer_checks': {'nonempty_unclipped_panels': 3*rendered_count,
                                    'minimum_boundary_clearance_pixels': minimum_clearance,
                                    'maximum_matched_bbox_delta_pixels': maximum_bbox_delta},
                'display_frame': 'wrist centered, inverse global rotation, side-specific proper upright rotation',
                'display_convention': 'upright_left_v2' if side == 'left' else 'upright_right_v1',
                'display_rotation_matrix': display.tolist(),
                'rows': index_rows, 'elapsed_seconds': time.perf_counter() - started}
    (args.output_dir / 'registration_atlas.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if export:
        if not shutil.which('pdftoppm'):
            raise RuntimeError('PDF complete; install Poppler pdftoppm for selected-page PNG exports')
        folder = args.output_dir / 'atlas_pages'
        folder.mkdir(exist_ok=True)
        for grid in export:
            subprocess.run(['pdftoppm', '-f', str(grid + 2), '-l', str(grid + 2), '-singlefile', '-scale-to', '1800',
                            '-png', str(path), str(folder / f'grid_{grid:03d}')], check=True)
    print(f'Wrote {path}: {pages+2} PDF pages, {rendered_count} hands; exports {export}', flush=True)


if __name__ == '__main__':
    main()
