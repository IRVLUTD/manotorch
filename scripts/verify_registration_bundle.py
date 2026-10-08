"""Verify a portable QC inventory, or explicitly write one after a completed build.

Uses only the Python standard library. Hashes use relative paths so a folder can move.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_dir', type=Path)
    parser.add_argument('--write-manifest', action='store_true', help='Explicitly replace the inventory for this run')
    args = parser.parse_args()
    root = args.bundle_dir.resolve()
    manifest_path = root / 'bundle_manifest.json'
    if args.write_manifest:
        report = json.loads((root / 'mirror_validation.json').read_text())
        for side in ('right', 'left'):
            atlas = json.loads((root / side / 'registration_atlas.json').read_text())
            if not atlas['complete'] or atlas['rendered_samples'] != report['samples_per_side']:
                raise ValueError(f'A complete {side} atlas is required')
            if digest(root / f'registration_atlas_{side}.pdf') != atlas['pdf_sha256']:
                raise ValueError(f'{side} atlas PDF hash mismatch')
            if digest(root / side / 'poses.npz') != report['sides'][side]['archive_sha256']:
                raise ValueError(f'{side} pose archive hash mismatch')
        summary = json.loads((root / 'qc_statistics.json').read_text())
        if digest(root / 'qc_summary.pdf') != summary['summary_pdf_sha256']:
            raise ValueError('Combined summary PDF hash mismatch')
        files = [p for p in sorted(root.rglob('*')) if p.is_file() and p != manifest_path
                 and '__pycache__' not in p.parts and not p.name.endswith('.partial.pdf')]
        manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'relative_paths': True,
                    'samples_per_side': report['samples_per_side'], 'self_contained_inputs': True,
                    'external_runtime_dependencies': 'Python packages, GPU/OpenGL driver; Poppler for PNG exports',
                    'files': {str(p.relative_to(root)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in files}}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid or missing bundle file: {name}')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Bundle checksum mismatch: {name}')
    print(f'Verified {len(manifest["files"])} files; {manifest["samples_per_side"]} poses per side; relative paths portable')


if __name__ == '__main__':
    main()
