"""Build a right-only registration audit PDF, CSV index and JSON summary.

Use summarize_registration_mirror.py for the combined right/left bundle report.

Publishing dependencies are optional: use `uv run --with reportlab --with matplotlib python ...`.
Metrics come from the complete accuracy run, not from the six-hand preview.
Atlas page numbers in the CSV describe a proposed alphabetical six-hand atlas;
this script does not render that atlas. Licensed assets remain under data/.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'p50': float(np.quantile(a, .5)),
            'p95': float(np.quantile(a, .95)), 'p99': float(np.quantile(a, .99)),
            'max': float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path('data/benchmarks/mano_registrations/2026-10-08'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/benchmarks/mano_registrations/right_qc_audit'))
    parser.add_argument('--preview', type=Path, default=Path('data/qc/MANO_Poses/right/registration_qc.png'))
    parser.add_argument('--fitting-dir', type=Path, action='append', default=[],
                        help='Completed fitting results directory; repeat to include multiple runs')
    parser.add_argument('--full-batch-validation-dir', type=Path,
                        help='Optional accuracy/stability run with chunk equal to the complete dataset size')
    parser.add_argument('--timing-note', default='Provisional shared-machine measurements; repeat performance runs '
                        'when idle before making optimization decisions.', help='Performance qualification printed in the PDF')
    parser.add_argument('--atlas-manifest', type=Path, help='Completed registration_atlas.json to link actual PDF pages')
    args = parser.parse_args()
    # Keep --help usable without optional publishing dependencies.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    report = json.loads((args.results_dir / 'report.json').read_text())
    manifest = json.loads((args.results_dir / 'manifest.json').read_text())
    count = manifest['registrations']
    names = sorted(entry['name'] for entry in manifest['files'])
    if len(set(names)) != count or len(names) != count:
        raise ValueError('Manifest must contain unique registration names')
    cases = {}
    for row in report['accuracy']:
        key = f"{row['device'].replace(':', '_')}_{row['dtype']}"
        path = args.results_dir / f'accuracy_{key}.csv'
        with path.open(newline='') as stream:
            records = list(csv.DictReader(stream))
        if len(records) != count or sorted(x['file'] for x in records) != names:
            raise ValueError(f'Incomplete or duplicate sample coverage: {path}')
        cases[key] = {x['file']: {k: float(v) for k, v in x.items() if k != 'file'} for x in records}
        if not all(math.isfinite(v) and v >= 0 for values in cases[key].values() for v in values.values()):
            raise ValueError(f'Nonfinite/negative metrics: {path}')
    preferred = 'cuda_float32' if 'cuda_float32' in cases else next(iter(cases))
    atlas = None
    atlas_rows = {}
    if args.atlas_manifest:
        atlas = json.loads(args.atlas_manifest.read_text())
        atlas_rows = {r['file']: r for r in atlas['rows']}
        if not atlas['complete'] or atlas['rendered_samples'] != count or sorted(atlas_rows) != names:
            raise ValueError('Atlas must cover every unique registration')
        atlas_pdf = Path(atlas['pdf'])
        if atlas.get('path_base') == 'manifest_directory':
            atlas_pdf = args.atlas_manifest.parent / atlas_pdf
        if digest(atlas_pdf) != atlas['pdf_sha256']:
            raise ValueError('Atlas PDF hash mismatch')
    ranks = {name: i + 1 for i, name in enumerate(sorted(
        names, key=lambda n: cases[preferred][n]['vertex_max_euclidean_mm'], reverse=True))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / 'qc_metrics.csv'
    rows = []
    for i, name in enumerate(names):
        row = {'file': name, 'subject': name.split('_')[0],
               'source_side': 'left_mirrored_to_right' if 'mirrored' in name else 'right',
               'proposed_atlas_grid_page': i // 6 + 1, 'proposed_atlas_row': i % 6 + 1,
               'error_rank': ranks[name]}
        for key, samples in cases.items():
            row.update({f'{key}_{metric}': value for metric, value in samples[name].items()})
        if atlas:
            row.update({'atlas_pdf_page': atlas_rows[name]['pdf_page'], 'atlas_grid_page': atlas_rows[name]['grid_page'],
                        'atlas_row': atlas_rows[name]['row']})
        rows.append(row)
    with metrics_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    threshold = report['environment']['arguments']['accuracy_mm']
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'registrations': count,
        'subjects': len({n.split('_')[0] for n in names}),
        'right': sum('mirrored' not in n for n in names),
        'left_mirrored_to_right': sum('mirrored' in n for n in names),
        'accuracy_threshold_mm': threshold, 'metric': 'Euclidean distance in original scanner frame',
        'unit': 'mm', 'accuracy': [], 'stability': report['stability'],
        'timing_note': args.timing_note,
        'environment': report['environment'],
        'source_hashes': {str(p): digest(p) for p in [args.results_dir / 'report.json',
                         args.results_dir / 'manifest.json', Path(__file__),
                         *sorted(args.results_dir.glob('accuracy_*.csv'))]},
        'proposed_atlas': {'grid_pages': math.ceil(count / 6), 'hands_per_page': 6,
                           'columns': ['Dataset target', 'manotorch reconstruction', 'Vertex error'],
                           'ordering': 'registration filename ascending', 'rendered': False,
                           'index_note': 'Grid page numbers exclude any cover/statistics pages'},
    }
    if atlas:
        summary['atlas'] = {k: v for k, v in atlas.items() if k != 'rows'}
        summary['proposed_atlas']['rendered'] = True
        summary['source_hashes'][str(args.atlas_manifest)] = digest(args.atlas_manifest)
    fitting_rows = []
    fitting_environments = []
    for fit_dir in args.fitting_dir:
        fit_report_path = fit_dir / 'report.json'
        fit_report = json.loads(fit_report_path.read_text())
        fit_arguments = fit_report['environment']['arguments']
        run_rows = fit_report['fitting']
        expected = len(fit_arguments['fit_batches']) * len(fit_arguments['fit_steps']) * len(
            fit_arguments['anatomy_weights']) * 2
        if len(run_rows) != expected:
            raise ValueError('Fitting report is incomplete')
        fitting_rows.extend(run_rows)
        fitting_environments.append(fit_report['environment'])
        summary['source_hashes'][str(fit_report_path)] = digest(fit_report_path)
    if fitting_rows:
        summary['fitting'] = [{**{k: v for k, v in row.items() if k not in ('per_hand', 'selected_indices')},
                              'final_anatomy_mean_rad': float(np.mean([x['anatomy_rad'] for x in row['per_hand']])),
                              'final_threshold_pass_rate': sum(x['target_rmse_mm'] <= row['threshold_mm']
                                                               for x in row['per_hand']) / len(row['per_hand'])}
                             for row in fitting_rows]
        summary['fitting_environments'] = fitting_environments
    if args.full_batch_validation_dir:
        source = args.full_batch_validation_dir / 'report.json'
        full = json.loads(source.read_text())
        if full['registrations'] != count or full['environment']['arguments']['chunk'] != count or not all(
                x['pass'] for stage in ('accuracy', 'stability') for x in full[stage]):
            raise ValueError('Full-batch validation must cover all registrations in one chunk and pass')
        summary['full_batch_validation'] = {'accuracy': full['accuracy'], 'stability': full['stability']}
        summary['source_hashes'][str(source)] = digest(source)
    for key, samples in cases.items():
        worst = max(names, key=lambda n: samples[n]['vertex_max_euclidean_mm'])
        summary['accuracy'].append({
            'case': key, 'samples': count, 'worst_file': worst,
            'violations': sum(max(x['vertex_max_euclidean_mm'], x['joint_max_euclidean_mm']) > threshold
                              for x in samples.values()),
            **{k: summarize([x[k] for x in samples.values()]) for k in next(iter(samples.values()))},
        })
    (args.output_dir / 'qc_statistics.json').write_text(json.dumps(summary, indent=2) + '\n')

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyQC', fontSize=10, leading=14, spaceAfter=8,
                              textColor=colors.HexColor('#26364A')))
    styles.add(ParagraphStyle(name='SmallQC', fontSize=8, leading=11, spaceAfter=6,
                              textColor=colors.HexColor('#475569')))
    styles['Title'].textColor = colors.HexColor('#16364F')
    styles['Heading2'].textColor = colors.HexColor('#16364F')
    story = []

    def paragraph(text, style='BodyQC'):
        story.append(Paragraph(text, styles[style]))

    def table(data, widths):
        item = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        item.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16364F')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8), ('LEADING', (0, 0), (-1, -1), 11),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F0F4F7'), colors.white]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#CED8E0')),
        ]))
        story.extend([item, Spacer(1, 10)])

    paragraph('MANO_Poses | Registration QC', 'Title')
    paragraph(f"Full-dataset statistics | {count:,} registrations | {summary['subjects']} subjects", 'Heading2')
    paragraph(f"Coverage: {summary['right']} right hands and {summary['left_mirrored_to_right']} "
              'left hands mirrored into the right-hand model. All stored meshes are tested with MANO_RIGHT. '
              'The synthetic sequences are outside this registration audit.')
    env = report['environment']
    paragraph(f"Environment: PyTorch {escape(env['torch'])}, NumPy {env['numpy']}, "
              f"{escape(env['gpu'] or 'CPU')}. Validation chunk size: {env['arguments']['chunk']}.", 'SmallQC')
    paragraph('Reconstruction accuracy', 'Heading2')
    paragraph('Errors are measured before display normalization, against the original PKL vertices and '
              '16 MANO joints, including scanner translation. The table uses micrometres (1 um = 0.001 mm). '
              'Each vertex statistic summarizes the maximum Euclidean error per hand.')
    data = [['Device / dtype', 'Vertex mean', 'Vertex P95', 'Vertex max', 'Joint max', 'Failures']]
    for row in summary['accuracy']:
        v = row['vertex_max_euclidean_mm']
        data.append([row['case'].replace('_', ' / '), f"{v['mean']*1000:.4f}", f"{v['p95']*1000:.4f}",
                     f"{v['max']*1000:.4f}", f"{row['joint_max_euclidean_mm']['max']*1000:.4f}",
                     str(row['violations'])])
    table(data, [113, 78, 75, 75, 75, 59])
    paragraph(f"Acceptance threshold: {threshold:g} mm. All {len(cases)} device/dtype cases cover all "
              f'{count:,} registrations. Float64 still uses model buffers rounded to float32 at construction; '
              'it is not a full double-precision model-asset path.', 'SmallQC')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout='constrained')
    for key in cases:
        if key.endswith('float32'):
            values = np.sort([x['vertex_max_euclidean_mm']*1000 for x in cases[key].values()])
            ax.plot(values, np.arange(1, count + 1) / count * 100, label=key.replace('_', ' / '), linewidth=2)
    ax.set(xlabel='Maximum vertex error per hand (um)', ylabel='Hands at or below error (%)',
           ylim=(0, 101), title='Full registration coverage: cumulative error distribution')
    ax.grid(alpha=.25)
    ax.legend(loc='lower right')
    chart = io.BytesIO()
    fig.savefig(chart, format='png', dpi=180)
    plt.close(fig)
    chart.seek(0)
    story.append(Image(chart, width=475, height=185))
    paragraph('This measures numerical reconstruction agreement with MANO registrations. It does not '
              'measure scan-fit quality, anatomical plausibility, or generalization to real images.', 'SmallQC')

    story.append(PageBreak())
    paragraph('Worst samples and numerical stability', 'Title')
    paragraph(f'Top 10 registrations by {preferred.replace("_", " / ")} vertex error', 'Heading2')
    data = [['Registration', 'Vertex max (mm)', 'Vertex RMSE (mm)', 'Joint max (mm)']]
    for name in sorted(names, key=ranks.get)[:10]:
        values = cases[preferred][name]
        data.append([name, *[f'{values[k]:.6g}' for k in
                             ('vertex_max_euclidean_mm', 'vertex_rmse_mm', 'joint_max_euclidean_mm')]])
    table(data, [163, 104, 119, 89])
    paragraph('Stability and independent reference checks', 'Heading2')
    for row in report['stability']:
        if 'device' in row:
            paragraph(f"<b>{row['device'].upper()}:</b> all {row['finite_forward_backward_samples']:,} "
                      f"forward/backward samples finite; {row['representative_samples']} representative/extreme "
                      f"hands checked for batching, permutation, joints-only and shared shape. "
                      f"Largest coordinate delta: {max(row['deltas'].values()):.6g} mm; "
                      f"directional finite-difference discrepancy: {row['directional_fd_max_abs']:.3g}. "
                      'Second derivatives finite at zero, near pi and rotation formula boundaries.', 'SmallQC')
    independent = next(x for x in report['stability'] if x.get('audit', '').startswith('independent float64'))
    paragraph(f"Independent NumPy float64 reconstruction vs original PKLs: maximum coordinate error "
              f"{independent['max_coordinate_mm']:.3g} mm. Left-pose matrix correspondence is also checked.", 'SmallQC')
    if 'full_batch_validation' in summary:
        largest = max(x['vertex_max_mm_per_hand']['max'] for x in summary['full_batch_validation']['accuracy'])
        paragraph(f"Single-batch confirmation: all {count:,} registrations processed together for accuracy "
                  f"and finite forward/backward checks; maximum vertex error {largest:.6g} mm. "
                  'Chunking in the main audit is a reporting choice, not a model batch-size limit.', 'SmallQC')
    left = [x for x in report['stability'] if x.get('audit') == 'left independent reference']
    paragraph('Left-model results are a separate derived-model check, not the stored right-model mesh comparison. '
              f"With shape-direction correction, mirror residual is {next(x for x in left if x['fix_left_shapedirs'])['mirror_max_coordinate_mm']:.6g} mm; "
              f"without correction it is {next(x for x in left if not x['fix_left_shapedirs'])['mirror_max_coordinate_mm']:.6g} mm. "
              'The corrected left layer agrees with its independent model reference; the remaining mirror '
              'residual reflects differences between left/right assets.', 'SmallQC')
    paragraph('Files and traceability', 'Heading2')
    paragraph('qc_metrics.csv contains every registration, every device/dtype error, error rank, and an '
              'alphabetical atlas page/row index. When an atlas is supplied, actual PDF page numbers are included. '
              'qc_statistics.json preserves distributions, environment and '
              'SHA-256 references. Original report.json, manifest.json and accuracy CSVs are in the source results directory.',
              'SmallQC')
    paragraph(f'Source: {escape(str(args.results_dir))}<br/>Report SHA-256: '
              f"{digest(args.results_dir / 'report.json')}", 'SmallQC')

    story.append(PageBreak())
    paragraph('Visual QC and full atlas', 'Title')
    status = 'Completed full atlas' if atlas else 'Proposed full atlas'
    paragraph(f'{status}: {math.ceil(count/6)} grid pages, with six hands per page and three '
              'columns (target / reconstruction / error). Use A3 portrait pages with large models and native '
              'PDF labels of at least 9 pt. Sort by filename, add subject bookmarks, use one '
              'global error colour scale, and link an error-ranked index to each grid page. Export selected '
              'PNG pages for sharing. Keep synthetic sequences in a separate atlas.', 'SmallQC')
    paragraph((f"The complete atlas is {atlas['pdf_pages']} pages including cover/index: {escape(atlas['pdf'])}. "
               'Subject bookmarks and index links jump to the corresponding hand pages. '
               if atlas else 'The full atlas has not been rendered. ') +
              'The preview below contains six selected hands, including '
              'the worst CUDA float32 case and extreme shapes/poses. Its display removes wrist translation '
              'and global rotation after errors are measured. Small-batch rounding can differ slightly from '
              'the full validation chunk; full-dataset statistics come from the accuracy CSVs.', 'SmallQC')
    if args.preview.exists():
        from PIL import Image as PILImage
        with PILImage.open(args.preview) as preview:
            w, h = preview.size
        height = min(575 if atlas else 620, 475 * h / w)
        story.append(Image(str(args.preview), width=height*w/h, height=height))
    else:
        paragraph('No QC preview image was supplied.')

    for sample_count in sorted({x['samples'] for x in fitting_rows}):
        story.append(PageBreak())
        bank_rows = [x for x in fitting_rows if x['samples'] == sample_count]
        paragraph(f'Batched MANO fitting | {sample_count:,} hands', 'Title')
        paragraph('The bank is subject-stratified and includes extreme shape/articulation; a full bank '
                  'contains every registration. Fits perturb stored pose '
                  'by 0.05 rad, shape by 0.1 and translation by 0.005 m. Adam optimizes all three. '
                  'This is a near-target convergence test, not image-based initialization or pose estimation.')
        devices = ', '.join(sorted({x['device'] for x in bank_rows}))
        groups = ', '.join(str(x) for x in sorted({len(r['group_shard_total_ms'][0]) for r in bank_rows}))
        paragraph(f'Eager float32 on {escape(devices)}; per-hand coordinate MSE plus optional mean anatomy penalty, '
                  'summed across hands. Learning rates: pose 0.01, shape 0.02, translation 0.001. '
                  f'Time is the sum of median shard durations over {groups} reset runs; loading, construction, '
                  'warmup and diagnostics are excluded. RMSE columns are mean per-hand Euclidean RMSE in mm.', 'SmallQC')
        data = [['Batch / N', 'Target', 'Prior wt.', 'Steps', 'Time (s)', 'Target RMSE', 'Mesh RMSE', 'Reached (%)']]
        for row in bank_rows:
            data.append([f"{row['batch']}/{row['samples']}", row['task'], f"{row['anatomy_weight']:g}", str(row['steps']),
                         f"{row['median_total_ms']/1000:.3f}", f"{row['final_target_rmse_mm']['mean']:.4f}",
                         f"{row['final_mesh_rmse_mm']['mean']:.4f}", f"{row['success_rate']*100:.1f}"])
        table(data, [65, 48, 45, 35, 50, 79, 73, 80])
        paragraph('Reached means at least one recorded update meets 0.1 mm vertex RMSE or 1 mm joint RMSE; '
                  'it is not the final pass rate. Threshold timestamps come from a separate diagnostic replay '
                  'that includes metric/host overhead. Joint fitting can match 16 joints while leaving mesh or '
                  'latent parameters ambiguous.', 'SmallQC')
        paragraph('The default prior weight is an experimental choice, not a calibrated recommendation. '
                  'Compare geometry and anatomy penalties together; a stronger prior can reduce its penalty '
                  'while moving away from the stored target. Timing cases run sequentially on a shared machine, '
                  'so the raw group/shard spread must be inspected before comparing batch sizes or prior costs.', 'SmallQC')
        penalties = []
        for task in ('vertices', 'joints'):
            matching = [r for r in summary['fitting'] if r['samples'] == sample_count and r['task'] == task]
            if matching:
                largest_batch = max(r['batch'] for r in matching)
                longest = max(r['steps'] for r in matching if r['batch'] == largest_batch)
                values = [r for r in matching if r['batch'] == largest_batch and r['steps'] == longest]
                penalties.extend(f"{task}, weight {r['anatomy_weight']:g}: {r['final_anatomy_mean_rad']:.4f} rad"
                                 for r in values)
        if penalties:
            paragraph('Final mean anatomy penalty (largest batch, longest fit): ' + '; '.join(penalties) + '.', 'SmallQC')
        paragraph(f'Timing status: {escape(args.timing_note)}', 'SmallQC')
        if not any(x['samples'] == count and x['batch'] == count for x in fitting_rows):
            paragraph('Full-dataset single-batch fitting is pending. Full-batch accuracy and finite-gradient '
                      'checks are recorded separately when supplied.', 'SmallQC')
        memory = [v for row in bank_rows for shard in row.get('group_shard_peak_extra_allocated_mib', []) for v in shard]
        if memory:
            paragraph(f'Largest measured extra fitting allocation: {max(memory):.2f} MiB. Model/input/optimizer '
                      'buffers already live before timing are excluded; this is not total GPU usage.', 'SmallQC')
        paragraph(f"Sources: {escape(', '.join(str(p) for p in args.fitting_dir))}. Complete curves, per-hand metrics, actual threshold "
                  'timestamps and memory measurements are preserved in fitting.json and report.json.', 'SmallQC')

    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#64748B'))
        canvas.drawString(40, 23, 'manotorch | MANO_Poses QC | licensed data, local artifact')
        canvas.drawRightString(A4[0] - 40, 23, str(doc.page))

    output = args.output_dir / 'qc_summary.pdf'
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=40, leftMargin=40,
                      topMargin=35, bottomMargin=40, title='MANO_Poses registration QC statistics',
                      author='manotorch').build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Wrote {output}, {metrics_path} and qc_statistics.json ({count} registrations)')


if __name__ == '__main__':
    main()
