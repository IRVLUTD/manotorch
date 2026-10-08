"""Recompute a bundle side using its stored poses/models and bundled manotorch source.

This portable entry point is copied to the QC folder's tools/. Originals are preserved;
the recomputed NPZ can be passed to the cached atlas renderer. No timing benchmark is run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from _registration_geometry import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, default=Path('data/qc/MANO_Poses'))
    parser.add_argument('--side', choices=('right', 'left'), required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, help='New NPZ; defaults to SIDE/poses_recomputed.npz in bundle')
    args = parser.parse_args()
    if min(args.chunk, args.threads) < 1:
        parser.error('chunk and threads must be positive')
    root = args.bundle_dir.resolve()
    source = root / args.side / 'poses.npz'
    destination = args.output or root / args.side / 'poses_recomputed.npz'
    if destination.resolve() == source:
        parser.error('output must preserve the original archive')
    if not (root / 'code/manotorch/manolayer.py').exists():
        parser.error('bundle must contain the manotorch source snapshot under code/')
    sys.path.insert(0, str(root / 'code'))
    import torch

    from manotorch.manolayer import ManoLayer

    torch.set_num_threads(args.threads)
    with np.load(source, allow_pickle=False) as data:
        archive = {k: data[k] for k in data.files}
    if str(archive['model_side']) != args.side:
        raise ValueError('Archive side mismatch')
    device = torch.device(args.device)
    layer = ManoLayer(side=args.side, mano_assets_root=str(root / 'models'), use_pca=False,
                      flat_hand_mean=True, center_idx=None, fix_left_shapedirs=args.side == 'left').to(device)
    vertices, joints = [], []
    with torch.no_grad():
        for start in range(0, len(archive['pose']), args.chunk):
            inputs = [torch.tensor(archive[k][start:start+args.chunk], device=device, dtype=torch.float32)
                      for k in ('pose', 'betas', 'trans')]
            out = layer(*inputs)
            if not all(torch.isfinite(t).all() for t in (out.verts, out.joints, out.transforms_abs)):
                raise ValueError('Nonfinite reconstruction')
            vertices.append(out.verts.cpu().double().numpy())
            joints.append(out.joints.cpu().double().numpy())
    archive['predicted_vertices'] = np.concatenate(vertices)
    archive['predicted_joints21'] = np.concatenate(joints)
    archive['predicted_joints'] = archive['predicted_joints21'][:, archive['mano16_indices']]
    v_error = float(np.linalg.norm(archive['predicted_vertices'] - archive['reference_vertices'], axis=-1).max()*1000)
    j_error = float(np.linalg.norm(archive['predicted_joints'] - archive['reference_joints'], axis=-1).max()*1000)
    if max(v_error, j_error) > .001:
        raise ValueError(f'Independent reference mismatch >0.001 mm: {v_error}, {j_error}')
    metadata = json.loads(str(archive['metadata']))
    metadata.update(device=str(device), torch=torch.__version__, chunk=args.chunk,
                    model_sha256=digest(root / f'models/MANO_{args.side.upper()}.npz'),
                    implementation_sha256={str(p.relative_to(root / 'code')): digest(p)
                                           for p in sorted((root / 'code/manotorch').rglob('*.py'))},
                    recomputed_from_archive_sha256=digest(source), inference_script_sha256=digest(Path(__file__)))
    archive['metadata'] = np.array(json.dumps(metadata))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial.npz')
    np.savez_compressed(partial, **archive)
    partial.replace(destination)
    print(f'Wrote {destination}; reference vertex/joint max: {v_error:.8f}/{j_error:.8f} mm')


if __name__ == '__main__':
    main()
