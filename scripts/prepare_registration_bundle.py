"""Build a self-contained right/left MANO_Poses QC bundle with independent references.

Left poses use S R S with S=diag(-1,1,1), including the global rotation.
Targets are geometric mirrors of original right-model registrations, not new left scans.
No timing/fitting benchmark is run. Licensed models and data remain under ignored data/.
"""

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from benchmark_registrations import FIELDS, MANO16, digest, numpy_mano, numpy_rotations, stats

from manotorch.manolayer import ManoLayer
from manotorch.utils.mano_io import find_mano_model, load_mano_model, load_mano_pickle


def geometry_error(pred, target):
    distance = np.linalg.norm(pred - target, axis=-1) * 1000
    return distance.max(1), np.sqrt((distance ** 2).mean(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/MANO_Poses/mano_poses_v1_0'))
    parser.add_argument('--mano-assets-root', type=Path, default=Path('assets/mano'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/qc/MANO_Poses'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.chunk, args.threads) < 1:
        parser.error('chunk and threads must be positive')
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((args.dataset / 'handsOnly_REGISTRATIONS_r_lm___POSES').glob('*.pkl'))
    if not files:
        raise FileNotFoundError('No original registrations found')
    records = [load_mano_pickle(p) for p in files]
    if any(r['model_name'] != 'MANO_RIGHT.pkl' or r['ncomps'] != 0 for r in records):
        raise ValueError('Expected absolute axis-angle MANO_RIGHT registrations')
    right = {k: np.stack([r[k] for r in records]).astype(np.float64) for k in FIELDS}
    for k, shape in FIELDS.items():
        if right[k].shape != (len(files), *shape) or not np.isfinite(right[k]).all():
            raise ValueError(f'Invalid original {k}')
    aggregate_paths = {s: args.dataset / f'handsOnly_REGISTRATIONS_r_lm___POSES___{s}.npy' for s in ('R', 'L')}
    r_pose, l_pose = [np.load(aggregate_paths[s], allow_pickle=False) for s in ('R', 'L')]
    if not np.array_equal(r_pose, right['pose'][:, 3:]) or l_pose.shape != r_pose.shape:
        raise ValueError('Invalid official R/L pose arrays or ordering')
    mirror = np.array([-1., 1., 1.])
    axial = -mirror
    left = {k: v.copy() for k, v in right.items()}
    left['pose'] = (right['pose'].reshape(-1, 16, 3) * axial).reshape(-1, 48)
    # Retain the dataset's canonical axis-angle representation, validating rotations rather than vectors.
    left['pose'][:, 3:] = l_pose
    left['trans'] *= mirror
    left['v'] *= mirror
    left['J_transformed'] *= mirror
    reflected_rotations = numpy_rotations(right['pose']) * mirror[None, :, None] * mirror[None, None, :]
    rotation_delta = float(np.abs(numpy_rotations(left['pose']) - reflected_rotations).max())
    if rotation_delta > 1e-12:
        raise ValueError(f'Pose reflection is inconsistent: {rotation_delta}')
    source_hashes = np.array([digest(p) for p in files])
    names = np.array([p.name for p in files])
    models_dir = args.output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    implementation_hashes = {str(p): digest(p) for p in sorted(Path('manotorch').rglob('*.py'))}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'samples_per_side': len(files),
              'subjects': len({p.name.split('_')[0] for p in files}),
              'mirror_matrix': np.diag(mirror).tolist(), 'rotation_matrix_max_delta': rotation_delta,
              'official_L_nonidentical_axis_angle_rows': int(np.any(np.abs(l_pose - (right['pose'][:, 3:].reshape(-1, 15, 3)
                                                                        * axial).reshape(-1, 45)) > 1e-8, axis=1).sum()),
              'aggregate_sha256': {s: digest(p) for s, p in aggregate_paths.items()},
              'implementation_sha256': implementation_hashes, 'script_sha256': digest(Path(__file__)),
              'reference_helper_sha256': digest(Path(__file__).with_name('benchmark_registrations.py')), 'sides': {},
              'units': 'metres in arrays; millimetres in metrics; radians in absolute axis-angle poses',
              'target_semantics': 'right: original registration; left: geometric mirror of right target',
              'timing_and_fitting': 'not run; GPU timing remains deferred while resources are shared'}
    rows = []
    selected_device = torch.device(args.device)
    devices = ['cpu'] + ([str(selected_device)] if selected_device.type != 'cpu' else [])
    for side, data in (('right', right), ('left', left)):
        model_path = find_mano_model(str(args.mano_assets_root), side)
        model = load_mano_model(model_path)
        # Store raw model arrays; correction is applied by ManoLayer, never twice in the asset.
        model_destination = models_dir / f'MANO_{side.upper()}.npz'
        np.savez_compressed(model_destination, **model)
        reference_model = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in model.items()}
        if side == 'left':
            reference_model['shapedirs'][:, 0, :] *= -1
        refs_v, refs_j = [], []
        uncorrected_v = []
        for start in range(0, len(files), args.chunk):
            ids = slice(start, start + args.chunk)
            v, j = numpy_mano(reference_model, data['pose'][ids], data['betas'][ids], data['trans'][ids])
            refs_v.append(v)
            refs_j.append(j)
            if side == 'left':
                v_raw, _ = numpy_mano(model, data['pose'][ids], data['betas'][ids], data['trans'][ids])
                uncorrected_v.append(v_raw)
        refs_v, refs_j = np.concatenate(refs_v), np.concatenate(refs_j)
        asset_vmax, asset_rmse = geometry_error(refs_v, data['v'])
        asset_jmax, _ = geometry_error(refs_j, data['J_transformed'])
        side_report = {'fix_left_shapedirs': side == 'left',
                       'numpy_vs_target_vertex_max_mm': stats(asset_vmax),
                       'numpy_vs_target_vertex_rmse_mm': stats(asset_rmse),
                       'numpy_vs_target_joint_max_mm': stats(asset_jmax), 'accuracy': []}
        if side == 'left':
            raw_vmax, _ = geometry_error(np.concatenate(uncorrected_v), data['v'])
            side_report['uncorrected_numpy_vs_mirrored_target_vertex_max_mm'] = stats(raw_vmax)
        saved_v = saved_j = saved_j21 = faces = None
        for dev in devices:
            for dtype_name in ('float32', 'float64'):
                device = torch.device(dev)
                layer = ManoLayer(side=side, use_pca=False, flat_hand_mean=True, center_idx=None,
                                  fix_left_shapedirs=side == 'left', mano_assets_root=str(models_dir)).to(
                                      device=device, dtype=getattr(torch, dtype_name))
                out_v, out_j, out_j21 = [], [], []
                with torch.no_grad():
                    for start in range(0, len(files), args.chunk):
                        inputs = [torch.as_tensor(data[k][start:start + args.chunk], device=device,
                                                  dtype=getattr(torch, dtype_name)) for k in ('pose', 'betas', 'trans')]
                        out = layer(*inputs)
                        if not all(torch.isfinite(t).all() for t in (out.verts, out.joints, out.transforms_abs)):
                            raise RuntimeError('Nonfinite output/transform')
                        out_v.append(out.verts.cpu().double().numpy())
                        out_j.append(out.joints[:, MANO16].cpu().double().numpy())
                        out_j21.append(out.joints.cpu().double().numpy())
                out_v, out_j, out_j21 = map(np.concatenate, (out_v, out_j, out_j21))
                vmax, rmse = geometry_error(out_v, refs_v)
                jmax, _ = geometry_error(out_j, refs_j)
                if max(vmax.max(), jmax.max()) > .001:
                    raise RuntimeError(f'{side}/{dev}/{dtype_name}: reference error exceeds 0.001 mm')
                case = {'device': dev, 'dtype': dtype_name, 'vertex_max_mm': stats(vmax),
                        'vertex_rmse_mm': stats(rmse), 'joint_max_mm': stats(jmax)}
                side_report['accuracy'].append(case)
                for i, name in enumerate(names):
                    rows.append({'file': str(name), 'side': side, 'device': dev, 'dtype': dtype_name,
                                 'reference_vertex_max_mm': vmax[i], 'reference_vertex_rmse_mm': rmse[i],
                                 'reference_joint_max_mm': jmax[i], 'asset_mirror_vertex_max_mm': asset_vmax[i],
                                 'asset_mirror_vertex_rmse_mm': asset_rmse[i], 'asset_mirror_joint_max_mm': asset_jmax[i]})
                if device == selected_device and dtype_name == 'float32':
                    saved_v, saved_j, saved_j21 = out_v, out_j, out_j21
                    faces = layer.th_faces.cpu().numpy()
                print(side, dev, dtype_name, 'reference max mm', vmax.max(), jmax.max(), flush=True)
        layer = ManoLayer(side=side, use_pca=False, flat_hand_mean=True, center_idx=None,
                          fix_left_shapedirs=side == 'left', mano_assets_root=str(models_dir)).to(selected_device)
        # Check every sample's first derivatives in small batches, avoiding benchmark/timing workloads.
        for start in range(0, len(files), args.chunk):
            inputs = [torch.tensor(data[k][start:start + args.chunk], device=selected_device,
                                   dtype=torch.float32, requires_grad=True) for k in ('pose', 'betas', 'trans')]
            out = layer(*inputs)
            loss = out.verts.square().sum() + out.joints.square().sum()
            gradients = torch.autograd.grad(loss, inputs)
            if not all(torch.isfinite(t).all() for t in gradients):
                raise RuntimeError(f'Nonfinite {side} gradient')
        with torch.no_grad():
            inputs = [torch.tensor(data[k], device=selected_device, dtype=torch.float32) for k in ('pose', 'betas', 'trans')]
            full = layer(*inputs)
            full_delta = float(np.linalg.norm(full.verts.cpu().double().numpy() - saved_v, axis=-1).max() * 1000)
            ids = np.linspace(0, len(files) - 1, 16, dtype=int)
            singles = torch.cat([layer(*(x[i:i+1] for x in inputs)).verts for i in ids])
            single_delta = float(torch.linalg.vector_norm(singles - full.verts[ids], dim=-1).max().item() * 1000)
            perm = np.random.default_rng(0).permutation(len(files))
            shuffled = layer(*(x[perm] for x in inputs)).verts.cpu().double().numpy()
            permutation_delta = float(np.linalg.norm(shuffled - full.verts.cpu().double().numpy()[perm], axis=-1)
                                      .max() * 1000)
            joints_only = layer(*inputs, joints_only=True).joints
            joints_delta = float(torch.linalg.vector_norm(joints_only - full.joints, dim=-1).max().item() * 1000)
        if max(full_delta, single_delta, permutation_delta, joints_delta) > .001:
            raise RuntimeError('Batch, permutation or joints-only mismatch exceeds 0.001 mm')
        side_report['stability'] = {'finite_output_samples': len(files), 'finite_gradient_samples': len(files),
                                    'full_batch_samples': len(files), 'single_batch_checked_samples': len(ids),
                                    'full_vs_chunk_vertex_max_mm': full_delta,
                                    'single_vs_full_vertex_max_mm': single_delta,
                                    'permutation_vertex_max_mm': permutation_delta,
                                    'joints_only_max_mm': joints_delta}
        metadata = {'device': str(selected_device), 'dtype': 'float32', 'torch': torch.__version__,
                    'chunk': args.chunk, 'model_sha256': digest(model_destination),
                    'implementation_sha256': implementation_hashes, 'fix_left_shapedirs': side == 'left',
                    'use_pca': False, 'flat_hand_mean': True, 'center_idx': None,
                    'target': 'original right-model registration' if side == 'right' else 'mirrored original right target',
                    'units': 'metres; radians for pose', 'J_transformed_order': 'MANO 16',
                    'predicted_joints21_order': 'manotorch 21; MANO16 indices stored explicitly'}
        side_dir = args.output_dir / side
        side_dir.mkdir(exist_ok=True)
        archive_path = side_dir / 'poses.npz'
        np.savez_compressed(archive_path, **data, names=names, source_sha256=source_hashes,
                            source_side=np.array(['left_mirrored' if 'mirrored' in p.name else 'right' for p in files]),
                            model_side=np.array(side), faces=faces, predicted_vertices=saved_v,
                            predicted_joints=saved_j, predicted_joints21=saved_j21, mano16_indices=np.array(MANO16),
                            reference_vertices=refs_v, reference_joints=refs_j, metadata=np.array(json.dumps(metadata)))
        side_report['archive'] = str(archive_path.relative_to(args.output_dir))
        side_report['archive_sha256'] = digest(archive_path)
        side_report['model'] = str(model_destination.relative_to(args.output_dir))
        side_report['model_sha256'] = digest(model_destination)
        report['sides'][side] = side_report
        print(side, 'bundle complete; asset residual max mm', asset_vmax.max(), flush=True)
    with np.load(args.output_dir / 'right/poses.npz', allow_pickle=False) as archive:
        right_faces = archive['faces']
    with np.load(args.output_dir / 'left/poses.npz', allow_pickle=False) as archive:
        left_faces = archive['faces']

    def oriented_faces(faces):
        return sorted(tuple(np.roll(face, -int(face.argmin()))) for face in faces)

    # Official left faces use different row/cyclic ordering; compare oriented triangles as sets.
    if oriented_faces(left_faces) != oriented_faces(right_faces[:, [2, 1, 0]]):
        raise ValueError('Left winding must reverse right winding after reflection')
    csv_path = args.output_dir / 'mirror_validation.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report['validation_csv_sha256'] = digest(csv_path)
    (args.output_dir / 'mirror_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    sources = args.output_dir / 'sources'
    sources.mkdir(exist_ok=True)
    shutil.copy2(args.dataset.parent / 'README.md', sources / 'MANO_Poses_README.md')
    for s, path in aggregate_paths.items():
        shutil.copy2(path, sources / f'official_{s}.npy')
    tools_dir = args.output_dir / 'tools'
    tools_dir.mkdir(exist_ok=True)
    for name in ('render_registration_atlas.py', '_registration_geometry.py',
                 'reinfer_registration_bundle.py', 'summarize_registration_mirror.py', 'verify_registration_bundle.py'):
        shutil.copy2(Path(__file__).with_name(name), tools_dir / name)
    for name in ('prepare_registration_bundle.py', 'benchmark_registrations.py'):
        shutil.copy2(Path(__file__).with_name(name), sources / f'{Path(name).stem}_executed.py')
    for path in Path('manotorch').rglob('*.py'):
        target = args.output_dir / 'code' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2('LICENSE', args.output_dir / 'code/LICENSE')
    shutil.copy2(Path(__file__).with_name('registration_bundle_README.md'), args.output_dir / 'README.md')
    print('Wrote portable archives, raw dense models, source snapshot, documentation and tools', flush=True)
    print('Next: render both atlases, publish the combined summary, then write/verify the bundle inventory.', flush=True)


if __name__ == '__main__':
    main()
