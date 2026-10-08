"""Run with: uv run pytest scripts/test_benchmark_scripts.py (no MANO assets needed)."""

from types import SimpleNamespace

import numpy as np
import torch
from benchmark_fitting import run_case, synchronize
from benchmark_registrations import MANO16, numpy_rotations


class TranslationModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pose, betas, transl, **kwargs):
        points = (transl[:, None] + 0 * pose[:, None, :3] + 0 * betas[:, None, :3]).expand(-1, 21, -1)
        return SimpleNamespace(verts=points, joints=points)


def test_final_update_is_in_curve_and_threshold(monkeypatch):
    markers = []
    compiler = getattr(torch, 'compiler', None)
    if compiler is not None and hasattr(compiler, 'cudagraph_mark_step_begin'):
        monkeypatch.setattr(compiler, 'cudagraph_mark_step_begin', lambda: markers.append(True))
    arrays = {'right_initial_pose': np.zeros((1, 48), dtype=np.float32),
              'right_initial_betas': np.zeros((1, 10), dtype=np.float32),
              'right_initial_transl': np.full((1, 3), .001, dtype=np.float32),
              'right_joints': np.zeros((1, 21, 3), dtype=np.float32)}
    args = SimpleNamespace(mano_assets_root='unused', source='mano', repeats=1, steps=1, threshold_mm=.1)
    row = run_case(args, TranslationModel, arrays, torch.device('cpu'), 'right', 1, 'joints', 'eager')
    assert row['curve_rmse_mm'][0] > .1
    assert row['final_rmse_mm'] < .1
    assert row['curve_rmse_mm'][-1] == row['final_rmse_mm']
    assert len(row['curve_rmse_mm']) == args.steps + 1
    assert row['steps_to_threshold'] == 1
    assert markers == []  # Eager CPU fitting must not initialize Inductor/CUDA-graph machinery.


def test_indexed_cuda_device_synchronizes(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(device))
    synchronize(torch.device('cuda:0'))
    synchronize(torch.device('cpu'))
    assert calls == [torch.device('cuda:0')]


def test_joint_order_and_independent_rodrigues():
    # Fixed MANO indices in the current 21-joint output; this must never become joints[:, :16].
    from manotorch.manolayer import JOINTS_REORDER

    assert [JOINTS_REORDER[index] for index in MANO16] == list(range(16))
    angles = np.array([[0., 0., 0.], [0., 0., np.pi / 2], [np.pi, 0., 0.]])
    rotation = numpy_rotations(angles)
    np.testing.assert_allclose(rotation[0], np.eye(3), atol=0)
    np.testing.assert_allclose(rotation[1], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=3e-16)
    np.testing.assert_allclose(rotation[2], np.diag([1, -1, -1]), atol=3e-16)
    np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rotation.shape), atol=1e-15)
