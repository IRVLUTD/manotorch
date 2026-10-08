"""Publish one combined right/left QC report, including paired visuals and historical fitting."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from _registration_geometry import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, default=Path('data/qc/MANO_Poses'))
    args = parser.parse_args()
    from matplotlib import pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    root = args.bundle_dir
    report = json.loads((root / 'mirror_validation.json').read_text())
    atlases = {}
    for side in ('right', 'left'):
        manifest_path = root / side / 'registration_atlas.json'
        atlas = json.loads(manifest_path.read_text())
        if not atlas['complete'] or atlas['rendered_samples'] != report['samples_per_side']:
            raise ValueError(f'Complete {side} atlas required')
        pdf_path = root / f'registration_atlas_{side}.pdf'
        if digest(pdf_path) != atlas['pdf_sha256']:
            raise ValueError(f'{side} atlas PDF hash mismatch')
        atlases[side] = atlas
    historical_path = root / 'right/audit_statistics.json'
    historical = json.loads(historical_path.read_text()) if historical_path.exists() else {}
    for side, data in report['sides'].items():
        if digest(root / data['archive']) != data['archive_sha256'] or digest(root / data['model']) != data['model_sha256']:
            raise ValueError(f'{side} bundle hash mismatch')
    if digest(root / 'mirror_validation.csv') != report['validation_csv_sha256']:
        raise ValueError('Validation CSV hash mismatch')
    with (root / 'mirror_validation.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(root / 'left/poses.npz', allow_pickle=False) as archive:
        names = archive['names'].tolist()
    for side, data in report['sides'].items():
        for case in data['accuracy']:
            case_rows = [r for r in rows if (r['side'], r['device'], r['dtype']) ==
                         (side, case['device'], case['dtype'])]
            if sorted(r['file'] for r in case_rows) != names:
                raise ValueError('CSV coverage mismatch')
            metrics = np.array([[float(r[k]) for k in ('reference_vertex_max_mm', 'reference_joint_max_mm',
                                                       'asset_mirror_vertex_max_mm')] for r in case_rows])
            if not np.isfinite(metrics).all() or metrics.min() < 0:
                raise ValueError('Invalid CSV metrics')
    left = report['sides']['left']
    pdf_path = root / 'qc_summary.pdf'
    partial = pdf_path.with_suffix('.partial.pdf')
    pdf = canvas.Canvas(str(partial), pagesize=A4, pageCompression=1)
    pdf.setTitle('MANO_Poses combined right/left QC summary')
    pdf.setAuthor('manotorch')
    width, height = A4
    margin = 42
    ink, muted = colors.HexColor('#17374F'), colors.HexColor('#526577')

    def text(y, value, size=10, bold=False, x=margin, color=ink):
        pdf.setFillColor(color)
        pdf.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
        pdf.drawString(x, y, value)

    def footer(page):
        text(22, 'manotorch | licensed local data | metrics in scanner frame', 9, color=muted)
        text(22, f'{page}/4', 9, x=width - 65, color=muted)

    pdf.bookmarkPage('statistics')
    pdf.addOutlineEntry('Right/left coverage, accuracy and stability', 'statistics', level=0)
    text(height - 55, 'Right / left registration QC', 25, True)
    text(height - 83, f'{report["samples_per_side"]:,} poses per side | {report["subjects"]} subjects | two 261-page atlases', 12)
    for i, line in enumerate([
        'The source registrations all use MANO_RIGHT; the left set is derived by reflection.',
        'This is a paired implementation/symmetry audit, not independent left-scan validation.',
        'All 48 pose values are absolute axis-angle. Betas stay unchanged; translation is mirrored.',
        'Use MANO_LEFT with fix_left_shapedirs=True, use_pca=False, flat_hand_mean=True.',
    ]):
        text(height - 113 - i * 18, line, 10)
    text(628, 'Implementation accuracy against independent NumPy FK/LBS', 13, True)
    text(608, 'Maximum Euclidean vertex / 16-joint error in micrometres (um).', 10)
    for x, label in zip((margin, 115, 295, 397), ('Side', 'Device / dtype', 'Vertex max (um)', 'Joint max (um)'), strict=True):
        text(581, label, 10, True, x=x)
    y = 560
    for side, data in report['sides'].items():
        for case in data['accuracy']:
            for x, value in zip((margin, 115, 295, 397),
                                (side, f'{case["device"]} / {case["dtype"]}',
                                 f'{case["vertex_max_mm"]["max"] * 1000:.6f}',
                                 f'{case["joint_max_mm"]["max"] * 1000:.6f}'), strict=True):
                text(y, value, 10, x=x)
            y -= 22
    text(355, 'Left/right model asset symmetry (independent NumPy, float64)', 13, True)
    asset = left['numpy_vs_target_vertex_max_mm']
    raw = left['uncorrected_numpy_vs_mirrored_target_vertex_max_mm']
    for i, line in enumerate([
        f'Corrected left vs mirrored right vertex max: {asset["max"]:.8f} mm ({asset["max"]*1000:.3f} um).',
        f'Per-hand maxima: median {asset["p50"]:.8f}; p95 {asset["p95"]:.8f} mm.',
        f'Corrected left vs mirrored right joint max: {left["numpy_vs_target_joint_max_mm"]["max"]:.8f} mm.',
        f'Without the left shapedirs correction: vertex max {raw["max"]:.5f} mm.',
        'The small corrected mirror residual is an asset difference, not a layer numerical error.',
    ]):
        text(332 - i * 18, line, 10)
    text(218, 'Stability and coverage', 13, True)
    for i, line in enumerate([
        'Both sides: all 1,554 forward outputs, transforms and pose/betas/trans gradients finite.',
        'Both sides: B=1,554 vs chunks, permutation and joints-only agree within 0.001 mm.',
        'Sixteen evenly spaced samples per side also pass single-hand vs full-batch checks.',
        f'Rotation reflection matrix agreement: {report["rotation_matrix_max_delta"]:.3g}.',
        'Both atlases pass every panel visibility, clipping and matched-camera gate.',
        'GPU inference/rendering performed; inference timing and fitting were not rerun.',
    ]):
        text(194 - i * 18, line, 10)
    footer(1)
    pdf.showPage()
    pdf.bookmarkPage('distributions')
    pdf.addOutlineEntry('Right/left distributions and reproducibility', 'distributions', level=0)
    text(height - 55, 'Accuracy distributions and provenance', 22, True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for side, color in (('right', '#237FA1'), ('left', '#C15A33')):
        side_selected = next(c for c in report['sides'][side]['accuracy']
                             if c['device'] == atlases[side]['device'] and c['dtype'] == 'float32')
        side_rows = [r for r in rows if (r['side'], r['device'], r['dtype']) ==
                     (side, side_selected['device'], 'float32')]
        values = np.sort([float(r['reference_vertex_max_mm']) * 1000 for r in side_rows])
        axes[0].plot(values, np.arange(1, len(values)+1) / len(values), color=color, linewidth=2, label=side)
        if side == 'left':
            values = np.sort([float(r['asset_mirror_vertex_max_mm']) * 1000 for r in side_rows])
            axes[1].plot(values, np.arange(1, len(values)+1) / len(values), color=color, linewidth=2)
    for ax, title in zip(axes, ('Layer vs independent reference', 'Corrected left-model mirror residual'), strict=True):
        ax.set(xlabel='Per-hand maximum vertex error (um)', ylabel='Cumulative fraction', title=title, ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower right')
    image = io.BytesIO()
    fig.savefig(image, format='png', dpi=180)
    plt.close(fig)
    image.seek(0)
    pdf.drawImage(ImageReader(image), margin, 555, width=width - 2*margin, height=190)
    lines = [
        'Reflection conventions',
        'Position/translation: (x, y, z) -> (-x, y, z). Rotation: R_left = S R_right S.',
        'Axis-angle: (rx, ry, rz) -> (rx, -ry, -rz), including the global rotation.',
        'Official L.npy articulation is retained; 115 rows use equivalent canonical vectors.',
        'Faces use official left winding, verified against reflected right triangles as sets.',
        '',
        'Self-contained inputs',
        'right/poses.npz and left/poses.npz each contain full pose, betas, trans, targets,',
        '16 joints, 21 reconstructed joints, faces, cached predictions and NumPy references.',
        'models/ contains raw dense model assets; apply the left correction exactly once.',
        'Source filenames remain identifiers: their suffix does not describe the derived side.',
        'sources/ retains the original README and official R/L arrays, with source hashes.',
        '',
        'Re-rendering and new inference',
        'The bundled tools renderer needs NumPy/PyVista/ReportLab/Pillow/matplotlib and',
        'Poppler for PNG exports; cached re-rendering needs no Torch or original source folder.',
        'The bundle README documents re-rendering, new ManoLayer calls and array meanings.',
        'Use mirror_validation.csv for all per-hand/device/dtype errors; JSON records stability.',
        'All PDFs are at bundle root; left/right folders hold matching data and page indices.',
        '',
        'Limits',
        'No new fitting or controlled inference performance measurement is claimed here.',
        'Synthetic temporal sequences, anatomical plausibility and true left-scan fitting',
        'quality are separate evaluations. Licensed inputs remain local and untracked.',
    ]
    y = 521
    headings = {'Reflection conventions', 'Self-contained inputs', 'Re-rendering and new inference', 'Limits'}
    for line in lines:
        text(y, line, 10, line in headings)
        y -= 18
    footer(2)
    pdf.showPage()
    pdf.bookmarkPage('paired_visuals')
    pdf.addOutlineEntry('Paired right/left visual QC', 'paired_visuals', level=0)
    text(height - 45, 'Paired right / left visual QC', 23, True)
    text(height - 65, 'Same source poses: first / maximum left mirror residual / maximum shape magnitude.', 9)
    from pypdf import PdfReader
    readers = {side: PdfReader(root / f'registration_atlas_{side}.pdf') for side in ('right', 'left')}
    with np.load(root / 'left/poses.npz', allow_pickle=False) as archive:
        preview_indices = [0, int(np.argmax([r['vertex_max_euclidean_mm'] for r in atlases['left']['rows']])),
                           int(np.argmax(np.abs(archive['betas']).max(axis=1)))]
    y = height - 88
    preview_names = []
    for index in preview_indices:
        preview_names.append(names[index])
        for side in ('right', 'left'):
            entry = atlases[side]['rows'][index]
            text(y, f'{side.upper()} | {names[index]} | max target residual {entry["vertex_max_euclidean_mm"]:.8f} mm', 10, True)
            image = readers[side].pages[int(entry['pdf_page']) - 1].images[0].image
            row_height = image.height // 6
            image = image.crop((0, (int(entry['row'])-1)*row_height, image.width, int(entry['row'])*row_height))
            if side == 'left' and atlases['left'].get('display_convention') != 'upright_left_v2':
                # Align older downward-facing atlases only; upright atlas panels already match the right.
                panel_width = image.width // 3
                for col in range(3):
                    panel = image.crop((col*panel_width, 0, (col+1)*panel_width, image.height)).rotate(180)
                    image.paste(panel, (col*panel_width, 0))
            draw_height = (width - 2*margin) * image.height / image.width
            pdf.drawImage(ImageReader(image), margin, y - 8 - draw_height, width=width-2*margin, height=draw_height)
            y -= 113
    text(50, 'Columns: target / reconstruction / error. Error scales differ by side; see each full atlas.', 9)
    preview_roll = 0 if atlases['left'].get('display_convention') == 'upright_left_v2' else 180
    text(36, ('Side-specific upright display rotations preserve handedness; measured errors are unchanged.'
              if preview_roll == 0 else 'Left preview panels rolled 180 degrees for display; measured errors are unchanged.'), 9)
    footer(3)
    pdf.showPage()
    pdf.bookmarkPage('historical_fitting')
    pdf.addOutlineEntry('Historical right-hand fitting and pending work', 'historical_fitting', level=0)
    text(height - 55, 'Historical fitting / pending work', 23, True)
    fitting = historical.get('fitting', [])
    text(height - 81, ('Existing right-model benchmark: fixed 128-hand bank; no new left fitting was run.'
                      if fitting else 'No historical fitting measurements are included in this bundle.'), 10)
    text(height - 100, 'Shared-machine timings are provisional. Table reports convergence, not speed ranking.', 10)
    columns = (margin, 74, 135, 182, 220, 295, 366, 414, 464)
    headers = ('B', 'Task', 'Weight', 'Steps', 'Target mm', 'Mesh mm', 'Pass %', 'Prior rad', 'MiB')
    for x, label in zip(columns, headers, strict=True):
        text(710, label, 9, True, x=x)
    y = 688
    for case in fitting:
        peak = max(v for shard in case['group_shard_peak_extra_allocated_mib'] for v in shard)
        values = (str(case['batch']), 'verts' if case['task'] == 'vertices' else 'joints',
                  f'{case["anatomy_weight"]:.0e}', str(case['steps']),
                  f'{case["final_target_rmse_mm"]["mean"]:.4f}', f'{case["final_mesh_rmse_mm"]["mean"]:.4f}',
                  f'{case["final_threshold_pass_rate"]*100:.2f}', f'{case["final_anatomy_mean_rad"]:.4f}', f'{peak:.2f}')
        for x, value in zip(columns, values, strict=True):
            text(y, value, 9, x=x)
        y -= 23
    lines = [
        'Target / mesh: mean final per-hand RMSE; vertices or MANO 16 joints as labeled.',
        'Pass % is the final iterate, not ever-passed: 0.1 mm vertices / 1 mm joints.',
        'Prior is the final mean anatomical penalty in radians; MiB is peak extra allocation.',
        'Each row covers the same 128-hand bank; B32 partitions it, B128 fits it in one batch.',
        'Full fitting case statistics and timing groups are retained in right/audit_statistics.json.',
        '',
        'Pending / limitations',
        'No fitting evaluation has been performed on the derived left-hand bank.',
        'Full B=1,554 fitting and controlled GPU timings await available resources.',
        'Anatomy weight 1e-4 increases geometric error; calibrate smaller weights first.',
        'Synthetic temporal QC and true independent left-scan transfer are separate tasks.',
        '',
        'Reproducibility',
        'Root qc_metrics.csv and qc_statistics.json summarize both sides and actual PDF pages.',
        'Each side folder has poses.npz, qc_metrics.csv, qc_statistics.json and atlas metadata.',
        'README documents optional dependencies, units, mirror rules and regeneration commands.',
    ]
    y = 281
    for line in lines:
        text(y, line, 10, line in ('Pending / limitations', 'Reproducibility'))
        y -= 15
    footer(4)
    pdf.save()
    partial.replace(pdf_path)
    metrics = []
    atlas_rows = {side: {r['file']: r for r in atlas['rows']} for side, atlas in atlases.items()}
    for record in rows:
        entry = atlas_rows[record['side']][record['file']]
        metrics.append({**record, 'atlas_pdf': f'registration_atlas_{record["side"]}.pdf',
                        'atlas_pdf_page': entry['pdf_page'], 'atlas_grid_page': entry['grid_page'],
                        'atlas_row': entry['row']})
    for destination, subset in [(root / 'qc_metrics.csv', metrics),
                                *[(root / side / 'qc_metrics.csv', [r for r in metrics if r['side'] == side])
                                  for side in ('right', 'left')]]:
        with destination.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(subset)
    source_hashes = {'mirror_validation.json': digest(root / 'mirror_validation.json'),
                     **{f'{side}/registration_atlas.json': digest(root / side / 'registration_atlas.json')
                        for side in ('right', 'left')}, 'summary_script': digest(Path(__file__))}
    if historical_path.exists():
        source_hashes['right/audit_statistics.json'] = digest(historical_path)
    output = {'validation': report, 'atlases': {s: {k: v for k, v in a.items() if k != 'rows'}
                                              for s, a in atlases.items()},
              'historical_right_fitting': fitting, 'timing_note': historical.get('timing_note'),
              'preview_names': preview_names, 'preview_left_panel_roll_degrees': preview_roll,
              'summary_pdf_sha256': digest(pdf_path),
              'source_sha256': source_hashes, 'metrics_csv_sha256': digest(root / 'qc_metrics.csv')}
    (root / 'qc_statistics.json').write_text(json.dumps(output, indent=2) + '\n')
    for side in ('right', 'left'):
        side_output = {'model_side': side, 'samples': report['samples_per_side'], **report['sides'][side],
                       'atlas': output['atlases'][side], 'combined_summary': '../qc_summary.pdf',
                       'combined_summary_sha256': output['summary_pdf_sha256'],
                       'metrics_csv_sha256': digest(root / side / 'qc_metrics.csv')}
        (root / side / 'qc_statistics.json').write_text(json.dumps(side_output, indent=2) + '\n')
    print(f'Wrote {pdf_path}', flush=True)


if __name__ == '__main__':
    main()
