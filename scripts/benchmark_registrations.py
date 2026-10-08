"""Validate MANO against official registration meshes, then measure inference and batched fitting.

Run from the repository root. All 1,554 PKLs, including l_mirrored, use MANO_RIGHT.
Targets come from the dataset, never from the layer under test. See scripts/README.md.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer
from manotorch.utils.mano_io import find_mano_model, load_mano_model, load_mano_pickle

MANO16 = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
FIELDS = {"pose": (48,), "betas": (10,), "trans": (3,), "v": (778, 3), "J_transformed": (16, 3)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mark_step(enabled=True):
    if not enabled:
        return
    fn = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()


def finite(*values):
    if not all(torch.isfinite(x).all().item() for x in values):
        raise RuntimeError("Non-finite output or derivative")


def stats(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def load_data(args):
    folder = args.dataset / "handsOnly_REGISTRATIONS_r_lm___POSES"
    files = sorted(folder.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No registration PKLs under {folder}")
    arrays = {key: [] for key in FIELDS}
    hashes = []
    for path in files:
        rec = load_mano_pickle(path)
        if rec["model_name"] != "MANO_RIGHT.pkl" or rec["ncomps"] != 0:
            raise ValueError(f"Unexpected pose/model convention: {path}")
        for key, shape in FIELDS.items():
            value = np.asarray(rec[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} in {path}")
            arrays[key].append(value)
        hashes.append({"name": path.name, "sha256": digest(path)})
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    aggregate = {}
    for side in ("R", "L"):
        path = args.dataset / f"handsOnly_REGISTRATIONS_r_lm___POSES___{side}.npy"
        aggregate[side] = np.load(path, allow_pickle=False)
        if aggregate[side].shape != (len(files), 45) or not np.isfinite(aggregate[side]).all():
            raise ValueError(f"Invalid aggregate poses: {path}")
    if not np.array_equal(arrays["pose"][:, 3:], aggregate["R"]):
        raise ValueError("R.npy does not follow sorted PKL order")
    arrays["left_articulation"] = aggregate["L"]
    manifest = {"registrations": len(files), "files": hashes,
                "subjects": len({p.name.split('_')[0] for p in files}),
                "max_abs_beta": float(np.abs(arrays["betas"]).max()),
                "model_convention": "all right, absolute 48-axis-angle, no centering, metres",
                "aggregate_sha256": {s: digest(args.dataset / f"handsOnly_REGISTRATIONS_r_lm___POSES___{s}.npy")
                                     for s in aggregate},
                "models": {s: {"path": find_mano_model(str(args.mano_assets_root), s),
                               "sha256": digest(find_mano_model(str(args.mano_assets_root), s))}
                           for s in ("right", "left")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return arrays, [p.name for p in files], manifest


def numpy_rotations(pose):
    """Independent float64 Rodrigues reference with analytic limits at zero."""
    a = np.asarray(pose).reshape(-1, 3)
    theta = np.linalg.norm(a, axis=-1)
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -a[:, 2], a[:, 1]
    K[:, 1, 0], K[:, 1, 2] = a[:, 2], -a[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -a[:, 1], a[:, 0]
    return np.eye(3) + np.sinc(theta / np.pi)[:, None, None] * K + (
        .5 * np.sinc(theta / (2 * np.pi)) ** 2)[:, None, None] * (K @ K)


def numpy_mano(model, pose, betas, translation):
    """Independent sequential FK + homogeneous LBS; no manotorch numerical helpers."""
    shaped = model['v_template'] + np.einsum('vdk,bk->bvd', model['shapedirs'], betas)
    joints = np.einsum('jv,bvd->bjd', model['J_regressor'], shaped)
    rotations = numpy_rotations(pose).reshape(-1, 16, 3, 3)
    feature = (rotations[:, 1:] - np.eye(3)).reshape(len(pose), -1)
    posed = shaped + np.einsum('vdp,bp->bvd', model['posedirs'], feature)
    parents = model['kintree_table'][0].astype(int)
    transforms = np.zeros((len(pose), 16, 4, 4))
    for joint in range(16):
        local = np.zeros((len(pose), 4, 4))
        local[:, :3, :3] = rotations[:, joint]
        local[:, :3, 3] = joints[:, joint] if joint == 0 else joints[:, joint] - joints[:, parents[joint]]
        local[:, 3, 3] = 1
        transforms[:, joint] = local if joint == 0 else transforms[:, parents[joint]] @ local
    posed_joints = transforms[:, :, :3, 3].copy()
    transforms[:, :, :3, 3] -= np.einsum('bjik,bjk->bji', transforms[:, :, :3, :3], joints)
    skin = np.einsum('vj,bjik->bvik', model['weights'], transforms)
    vertices = np.einsum('bvik,bvk->bvi', skin[:, :, :3, :3], posed) + skin[:, :, :3, 3]
    return vertices + translation[:, None], posed_joints + translation[:, None]


def tensor_inputs(data, indices, device, dtype):
    return [torch.as_tensor(data[key][indices], dtype=dtype, device=device) for key in ('pose', 'betas', 'trans')]


def layer_for(args, device, dtype=torch.float32, **kwargs):
    return ManoLayer(mano_assets_root=str(args.mano_assets_root), side='right', use_pca=False,
                     flat_hand_mean=True, center_idx=None, **kwargs).to(device=device, dtype=dtype)


def accuracy(args, data, names):
    result = []
    for dev in args.devices:
        device = torch.device(dev)
        for dtype_name in ('float32', 'float64'):
            dtype = getattr(torch, dtype_name)
            layer = layer_for(args, device, dtype)
            vertex_errors, joint_errors, rmses = [], [], []
            with torch.no_grad():
                for start in range(0, len(names), args.chunk):
                    ids = np.arange(start, min(start + args.chunk, len(names)))
                    out = layer(*tensor_inputs(data, ids, device, dtype))
                    finite(out.verts, out.joints, out.transforms_abs)
                    v = out.verts.cpu().double().numpy() - data['v'][ids]
                    j = out.joints[:, MANO16].cpu().double().numpy() - data['J_transformed'][ids]
                    vertex_errors.extend(np.linalg.norm(v, axis=-1).max(-1) * 1000)
                    joint_errors.extend(np.linalg.norm(j, axis=-1).max(-1) * 1000)
                    rmses.extend(np.sqrt(np.mean(np.sum(v * v, axis=-1), axis=-1)) * 1000)
            tag = f'{dev.replace(":", "_")}_{dtype_name}'
            with (args.output_dir / f'accuracy_{tag}.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['file', 'vertex_max_euclidean_mm', 'joint_max_euclidean_mm', 'vertex_rmse_mm'])
                writer.writerows(zip(names, vertex_errors, joint_errors, rmses, strict=True))
            worst = int(np.argmax(vertex_errors))
            with torch.no_grad():
                out = layer(*tensor_inputs(data, [worst], device, dtype))
            np.savez_compressed(args.output_dir / f'worst_{tag}.npz', registration_file=names[worst],
                                target=data['v'][worst], predicted=out.verts[0].cpu().numpy(),
                                faces=layer.th_faces.cpu().numpy())
            row = {'device': dev, 'dtype': dtype_name, 'samples': len(names),
                   'vertex_max_mm_per_hand': stats(vertex_errors), 'joint_max_mm_per_hand': stats(joint_errors),
                   'vertex_rmse_mm_per_hand': stats(rmses), 'worst_file': names[worst],
                   'pass': bool(max(max(vertex_errors), max(joint_errors)) <= args.accuracy_mm)}
            result.append(row)
            print('accuracy', row, flush=True)
    return result


def stability(args, data, names):
    result = []
    # Deterministic representative set plus the most extreme shape/rotation, independent of timing batches.
    ids = np.unique(np.r_[np.linspace(0, len(names) - 1, min(32, len(names))).astype(int),
                         np.abs(data['betas']).max(1).argmax(), np.linalg.norm(data['pose'][:, 3:], axis=1).argmax()])
    cpu_output = None
    for dev in args.devices:
        device = torch.device(dev)
        layer = layer_for(args, device)
        for start in range(0, len(names), args.chunk):
            ix = np.arange(start, min(start + args.chunk, len(names)))
            p, b, t = [x.requires_grad_() for x in tensor_inputs(data, ix, device, torch.float32)]
            out = layer(p, b, t)
            grads = torch.autograd.grad(out.verts.square().sum() + out.joints.square().sum(), (p, b, t))
            finite(out.verts, out.joints, *grads)
        inputs = tensor_inputs(data, ids, device, torch.float32)
        with torch.no_grad():
            batch = layer(*inputs)
            singles = torch.cat([layer(*(x[i:i+1] for x in inputs)).verts for i in range(len(ids))])
            reversed_out = layer(*(x.flip(0) for x in inputs)).verts.flip(0)
            only = layer(*inputs, joints_only=True)
            shared = layer(inputs[0], inputs[1][:1], inputs[2])
            expanded = layer(inputs[0], inputs[1][:1].expand(len(ids), -1), inputs[2])
            deltas = {'batch_single_mm': (batch.verts - singles).abs().max().item() * 1000,
                      'permutation_mm': (batch.verts - reversed_out).abs().max().item() * 1000,
                      'joints_only_mm': (batch.joints - only.joints).abs().max().item() * 1000,
                      'shared_betas_mm': (shared.verts - expanded.verts).abs().max().item() * 1000}
            if cpu_output is not None:
                deltas['cross_device_mm'] = (batch.verts.cpu() - cpu_output).abs().max().item() * 1000
            if device.type == 'cpu':
                cpu_output = batch.verts.cpu()
        # Compare full/joints-only input gradients on the same joint objective.
        p, b, t = [x.detach().requires_grad_() for x in inputs]
        grad_full = torch.autograd.grad(layer(p, b, t).joints.square().sum(), (p, b, t))
        grad_only = torch.autograd.grad(layer(p, b, t, joints_only=True).joints.square().sum(), (p, b, t))
        for a, b_grad in zip(grad_full, grad_only, strict=True):
            torch.testing.assert_close(a, b_grad, atol=2e-5, rtol=2e-4)
        double_layer = layer_for(args, device, torch.float64)
        rng = torch.Generator(device=device).manual_seed(args.seed)
        directional = []
        for index in ids[:args.derivative_samples]:
            p, b, t = [x.requires_grad_() for x in tensor_inputs(data, [index], device, torch.float64)]
            direction = torch.randn(p.shape, generator=rng, device=device, dtype=p.dtype)
            direction /= direction.norm()
            objective = double_layer(p, b, t).verts.square().sum()
            gradient = torch.autograd.grad(objective, p, create_graph=True)[0]
            hessian_vector = torch.autograd.grad((gradient * direction).sum(), p)[0]
            eps = 1e-6
            with torch.no_grad():
                fd = (double_layer(p + eps * direction, b, t).verts.square().sum()
                      - double_layer(p - eps * direction, b, t).verts.square().sum()) / (2 * eps)
            analytic = (gradient * direction).sum()
            torch.testing.assert_close(fd, analytic, atol=1e-8, rtol=1e-5)
            finite(gradient, hessian_vector)
            directional.append(abs((fd - analytic).item()))
        # Exercise model second derivatives, not just a standalone rotation conversion.
        for angle in (0., 1e-8, .00999, .01001, .01999, .02001, math.pi - 1e-6, math.pi + 1e-6):
            p = torch.zeros(1, 48, device=device, dtype=torch.float64)
            p[0, ::3] = angle
            p.requires_grad_()
            out = double_layer(p)
            grad = torch.autograd.grad(out.verts.square().sum(), p, create_graph=True)[0]
            second = torch.autograd.grad(grad.sum(), p)[0]
            finite(out.verts, grad, second)
        row = {'device': dev, 'finite_forward_backward_samples': len(names), 'representative_samples': len(ids),
               'deltas': deltas, 'directional_fd_max_abs': max(directional), 'boundary_second_derivatives': 'finite',
               'pass': max(deltas.values()) <= args.accuracy_mm}
        result.append(row)
        print('stability', row, flush=True)
    # Matrix comparison accommodates equivalent >pi representatives in L.npy.
    right = data['pose'].copy()
    right[:, :3] = 0
    left = np.c_[np.zeros((len(names), 3)), data['left_articulation']]
    mirrored = right.reshape(-1, 3) * np.array([1, -1, -1])
    matrix_delta = float(np.abs(numpy_rotations(mirrored) - numpy_rotations(left)).max())
    if matrix_delta > 1e-10:
        raise RuntimeError(f'L.npy mirror rotation discrepancy: {matrix_delta}')
    model_r = load_mano_model(find_mano_model(str(args.mano_assets_root), 'right'))
    model_l = load_mano_model(find_mano_model(str(args.mano_assets_root), 'left'))
    raw_reference_max = 0.
    for start in range(0, len(names), args.chunk):
        sl = slice(start, start + args.chunk)
        v, j = numpy_mano(model_r, data['pose'][sl], data['betas'][sl], data['trans'][sl])
        raw_reference_max = max(raw_reference_max, float(np.abs(v - data['v'][sl]).max()),
                                float(np.abs(j - data['J_transformed'][sl]).max()))
    if raw_reference_max * 1000 > args.accuracy_mm:
        raise RuntimeError('Model arrays do not reproduce the original registration targets')
    dev = torch.device(args.devices[-1])
    r_layer = layer_for(args, dev)
    for fixed in (False, True):
        model = {k: np.asarray(v).copy() if isinstance(v, np.ndarray) else v for k, v in model_l.items()}
        if fixed:
            model['shapedirs'][:, 0, :] *= -1
        layer = ManoLayer(side='left', use_pca=False, flat_hand_mean=True,
                          fix_left_shapedirs=fixed, mano_assets_root=str(args.mano_assets_root)).to(dev)
        max_reference, max_mirror = 0., 0.
        with torch.no_grad():
            for start in range(0, len(names), args.chunk):
                sl = slice(start, start + args.chunk)
                beta = data['betas'][sl]
                v, _ = numpy_mano(model, left[sl], beta, np.zeros((len(beta), 3)))
                p = torch.as_tensor(left[sl], device=dev, dtype=torch.float32)
                b = torch.as_tensor(beta, device=dev, dtype=torch.float32)
                output = layer(p, b).verts
                finite(output)
                reference = torch.as_tensor(v, device=dev, dtype=torch.float32)
                max_reference = max(max_reference, (output - reference).abs().max().item() * 1000)
                rv = r_layer(torch.as_tensor(right[sl], device=dev, dtype=torch.float32), b).verts
                max_mirror = max(max_mirror, (output - rv * rv.new_tensor([-1, 1, 1])).abs().max().item() * 1000)
        result.append({'audit': 'left independent reference', 'fix_left_shapedirs': fixed,
                       'samples': len(names), 'reference_max_coordinate_mm': max_reference,
                       'mirror_max_coordinate_mm': max_mirror, 'pass': max_reference <= args.accuracy_mm})
    result.append({'audit': 'independent float64 NumPy vs original PKLs', 'max_coordinate_mm': raw_reference_max * 1000,
                   'left_mirror_matrix_max_abs': matrix_delta, 'pass': True})
    return result


def block_time(fn, device, repeats, compiled=False):
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        mark_step(compiled and device.type == 'cuda')
        fn()
    sync(device)
    return (time.perf_counter() - start) * 1000 / repeats


def runtime(args, data):
    result = []
    for dev in args.runtime_devices:
        device = torch.device(dev)
        for batch in args.batches:
            if batch > len(data['pose']):
                raise ValueError('Runtime batches must not exceed the number of independent registrations')
            inputs = tensor_inputs(data, np.arange(batch), device, torch.float32)
            for joints_only in (False, True):
                layer = layer_for(args, device)
                native = partial(layer, joints_only=joints_only)
                with torch.inference_mode():
                    expected = native(*inputs)
                    expected_j = expected.joints.clone()
                    expected_v = None if joints_only else expected.verts.clone()
                methods, startup = {'eager': native}, {'eager': 0.}
                if 'compile' in args.modes:
                    torch._dynamo.reset()
                    compiled = torch.compile(native, fullgraph=True, mode='reduce-overhead')
                    sync(device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        mark_step(device.type == 'cuda')
                        output = compiled(*inputs)
                        sync(device)
                        startup['compile'] = time.perf_counter() - start
                        torch.testing.assert_close(output.joints, expected_j, atol=1e-6, rtol=1e-5)
                        if not joints_only:
                            torch.testing.assert_close(output.verts, expected_v, atol=1e-6, rtol=1e-5)
                    methods['compile'] = compiled
                samples = {name: [] for name in methods}
                gpu_samples = {name: [] for name in methods}
                memory = {}
                with torch.inference_mode():
                    for name, fn in methods.items():
                        for _ in range(10):
                            mark_step(name == 'compile' and device.type == 'cuda')
                            fn(*inputs)
                    names = list(methods)
                    for group in range(args.groups):
                        for name in names[group % len(names):] + names[:group % len(names)]:
                            call = partial(methods[name], *inputs)
                            samples[name].append(block_time(call, device, args.repeats, compiled=name == 'compile'))
                            if device.type == 'cuda':
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record(torch.cuda.current_stream(device))
                                for _ in range(args.repeats):
                                    mark_step(name == 'compile')
                                    call()
                                end.record(torch.cuda.current_stream(device))
                                end.synchronize()
                                gpu_samples[name].append(start.elapsed_time(end) / args.repeats)
                                torch.cuda.reset_peak_memory_stats(device)
                                allocated = torch.cuda.memory_allocated(device)
                                mark_step(name == 'compile')
                                call()
                                sync(device)
                                memory[name] = {'peak_extra_allocated_mib':
                                                (torch.cuda.max_memory_allocated(device) - allocated) / 2**20,
                                                'total_reserved_mib': torch.cuda.memory_reserved(device) / 2**20}
                for name in methods:
                    median = statistics.median(samples[name])
                    row = {'device': dev, 'batch': batch, 'independent_samples': batch,
                           'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                           'joints_only': joints_only, 'mode': name, 'group_wall_ms': samples[name],
                           'median_wall_ms': median, 'hands_per_second': batch * 1000 / median,
                           'group_cuda_event_ms': gpu_samples[name], 'startup_seconds': startup[name],
                           'memory': memory.get(name), 'grad_mode': 'inference_mode'}
                    result.append(row)
                    (args.output_dir / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
                    print('runtime', row, flush=True)
    return result


def fitting(args, data, names):
    device = torch.device(args.fit_device)
    # One fixed, subject-stratified bank for all batch sizes; include the largest shapes/rotations.
    rng = np.random.default_rng(args.seed)
    selected = []
    for subject in sorted({name.split('_')[0] for name in names}):
        candidates = [i for i, name in enumerate(names) if name.split('_')[0] == subject]
        selected.append(int(rng.choice(candidates)))
    extremes = [int(np.abs(data['betas']).max(1).argmax()), int(np.linalg.norm(data['pose'][:, 3:], axis=1).argmax())]
    selected = list(dict.fromkeys(extremes + selected))[:args.fit_samples]
    remainder = [int(i) for i in rng.permutation(len(names)) if i not in selected]
    selected += remainder[:max(0, args.fit_samples - len(selected))]
    bank = {key: data[key][selected] for key in FIELDS}
    initial = [bank[k] + rng.normal(0, sigma, bank[k].shape) for k, sigma in
               (('pose', .05), ('betas', .1), ('trans', .005))]
    result = []
    for batch in args.fit_batches:
        for task in ('vertices', 'joints'):
            for weight in args.anatomy_weights:
                for steps in args.fit_steps:
                    per_hand, totals, crossings, memories = [], [], [], []
                    for start in range(0, len(selected), batch):
                        ix = slice(start, min(start + batch, len(selected)))
                        layer = layer_for(args, device)
                        axis = AxisLayerFK(mano_assets_root=str(args.mano_assets_root)).to(device)
                        prior = AnatomyConstraintLossEE(reduction='none').to(device)
                        prior.setup()
                        p, b, t = [torch.tensor(a[ix], device=device, dtype=torch.float32, requires_grad=True)
                                   for a in initial]
                        originals = [x.detach().clone() for x in (p, b, t)]
                        target = torch.as_tensor(bank['v' if task == 'vertices' else 'J_transformed'][ix],
                                                 device=device, dtype=torch.float32)
                        optimizer = torch.optim.Adam([{'params': [p], 'lr': .01}, {'params': [b], 'lr': .02},
                                                      {'params': [t], 'lr': .001}], foreach=False)

                        def predict(layer=layer, p=p, b=b, t=t, task=task):
                            out = layer(p, b, t)
                            points = out.verts if task == 'vertices' else out.joints[:, MANO16]
                            return out, points

                        def step(optimizer=optimizer, predict=predict, target=target, weight=weight, prior=prior, axis=axis):
                            optimizer.zero_grad(set_to_none=True)
                            out, points = predict()
                            # Sum independent per-hand losses: batching does not rescale Adam gradients/epsilon.
                            objective = (points - target).square().mean((1, 2))
                            if weight:
                                objective = objective + weight * prior(axis(out.transforms_abs)[2]).mean(-1)
                            objective.sum().backward()
                            optimizer.step()

                        for _ in range(5):
                            step()
                        with torch.no_grad():
                            for value, original in zip((p, b, t), originals, strict=True):
                                value.copy_(original)
                            for state in optimizer.state.values():
                                for value in state.values():
                                    if isinstance(value, torch.Tensor):
                                        value.zero_()
                        group_ms = []
                        group_memory = []
                        for _group in range(args.fit_groups):
                            with torch.no_grad():
                                for value, original in zip((p, b, t), originals, strict=True):
                                    value.copy_(original)
                                for state in optimizer.state.values():
                                    for value in state.values():
                                        if isinstance(value, torch.Tensor):
                                            value.zero_()
                            sync(device)
                            if device.type == 'cuda':
                                torch.cuda.reset_peak_memory_stats(device)
                                memory_before = torch.cuda.memory_allocated(device)
                            begin = time.perf_counter()
                            for _ in range(steps):
                                step()
                            sync(device)
                            group_ms.append((time.perf_counter() - begin) * 1000)
                            if device.type == 'cuda':
                                group_memory.append((torch.cuda.max_memory_allocated(device) - memory_before) / 2**20)
                        # Independent diagnostic replay measures actual first threshold timestamps.
                        with torch.no_grad():
                            for value, original in zip((p, b, t), originals, strict=True):
                                value.copy_(original)
                            for state in optimizer.state.values():
                                for value in state.values():
                                    if isinstance(value, torch.Tensor):
                                        value.zero_()
                        threshold = .1 if task == 'vertices' else 1.
                        first = [None] * len(p)
                        curves = []
                        sync(device)
                        begin = time.perf_counter()
                        for iteration in range(steps + 1):
                            with torch.no_grad():
                                _, points = predict()
                                rmse = (points - target).square().sum(-1).mean(-1).sqrt() * 1000
                                finite(rmse)
                                values = rmse.cpu().tolist()
                            elapsed = (time.perf_counter() - begin) * 1000
                            curves.append(values)
                            for i, value in enumerate(values):
                                if first[i] is None and value <= threshold:
                                    first[i] = {'updates': iteration, 'diagnostic_elapsed_ms': elapsed}
                            if iteration < steps:
                                step()
                        with torch.no_grad():
                            out, _ = predict()
                            reference_v = torch.as_tensor(bank['v'][ix], device=device, dtype=torch.float32)
                            mesh_rmse = (out.verts - reference_v).square().sum(-1).mean(-1).sqrt() * 1000
                            penalty = prior(axis(out.transforms_abs)[2]).mean(-1)
                            finite(mesh_rmse, penalty)
                            metrics = torch.stack((rmse, mesh_rmse, penalty), -1).cpu().tolist()
                        for offset, values in enumerate(metrics):
                            index = start + offset
                            per_hand.append({'file': names[selected[index]], 'target_rmse_mm': values[0],
                                             'mesh_rmse_mm': values[1], 'anatomy_rad': values[2],
                                             'initial_rmse_mm': curves[0][offset],
                                             'curve_rmse_mm': [x[offset] for x in curves],
                                             'threshold': first[offset]})
                        totals.append(group_ms)
                        memories.append(group_memory)
                        crossings.extend(first)
                    row = {'batch': batch, 'samples': len(selected), 'selected_indices': selected,
                           'task': task, 'anatomy_weight': weight, 'steps': steps, 'device': str(device),
                           'group_shard_total_ms': totals, 'median_total_ms': sum(statistics.median(x) for x in totals),
                           'group_shard_peak_extra_allocated_mib': memories,
                           'threshold_mm': threshold, 'success_rate': sum(x is not None for x in crossings) / len(crossings),
                           'final_target_rmse_mm': stats([x['target_rmse_mm'] for x in per_hand]),
                           'final_mesh_rmse_mm': stats([x['mesh_rmse_mm'] for x in per_hand]), 'per_hand': per_hand,
                           'threshold_timing': 'separate synchronized diagnostic replay, includes metric/host overhead'}
                    result.append(row)
                    (args.output_dir / 'fitting.json').write_text(json.dumps(result, indent=2) + '\n')
                    print('fitting', {k: v for k, v in row.items() if k not in
                                     ('per_hand', 'selected_indices', 'group_shard_total_ms')}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/MANO_Poses/mano_poses_v1_0'))
    parser.add_argument('--mano-assets-root', type=Path, default=Path('assets/mano'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/benchmarks/mano_registrations'))
    parser.add_argument('--stages', nargs='+', choices=['audit', 'accuracy', 'stability', 'runtime', 'fitting'],
                        default=['audit', 'accuracy', 'stability', 'runtime', 'fitting'])
    parser.add_argument('--devices', nargs='+', default=['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
    parser.add_argument('--runtime-devices', nargs='+', default=['cuda'] if torch.cuda.is_available() else ['cpu'])
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32, 128, 512, 1024, 1554])
    parser.add_argument('--modes', nargs='+', choices=['eager', 'compile'], default=['eager'])
    parser.add_argument('--groups', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--accuracy-mm', type=float, default=.001)
    parser.add_argument('--derivative-samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--fit-device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fit-samples', type=int, default=128)
    parser.add_argument('--fit-batches', type=int, nargs='+', default=[32, 128])
    parser.add_argument('--fit-steps', type=int, nargs='+', default=[100, 300])
    parser.add_argument('--fit-groups', type=int, default=3)
    parser.add_argument('--anatomy-weights', type=float, nargs='+', default=[0., 1e-4])
    args = parser.parse_args()
    if min(args.groups, args.repeats, args.chunk, args.threads, args.derivative_samples,
           args.fit_samples, args.fit_groups, *args.batches, *args.fit_batches, *args.fit_steps) < 1:
        parser.error('Counts, batches, steps and threads must be positive')
    if not math.isfinite(args.accuracy_mm) or args.accuracy_mm <= 0 or any(
            not math.isfinite(w) or w < 0 for w in args.anatomy_weights):
        parser.error('accuracy-mm must be positive; anatomy weights finite and nonnegative')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data, names, manifest = load_data(args)
    if args.fit_samples > len(names):
        parser.error('fit-samples must not exceed independent registration count')
    result = {'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
                             'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                             'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                             'source_sha256': {str(p): digest(p) for p in
                                              [Path(__file__), *sorted(Path('manotorch').rglob('*.py'))]},
                             'float64_note': 'ManoLayer buffers are rounded to float32 at construction',
                             'timing': 'construction/transfers excluded; inference_mode; synchronized wall time',
                             'cuda_event_note': 'event spans include GPU idle gaps between host launches, not kernel-only sums'},
              'manifest_sha256': digest(args.output_dir / 'manifest.json'), 'registrations': manifest['registrations']}
    for stage, fn in [('accuracy', accuracy), ('stability', stability), ('runtime', runtime), ('fitting', fitting)]:
        if stage in args.stages:
            result[stage] = fn(args, data) if stage == 'runtime' else fn(args, data, names)
            (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
            if stage in ('accuracy', 'stability') and any(not row['pass'] for row in result[stage]):
                raise SystemExit(f'{stage} acceptance failed; see report.json')
    (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    if any(not row['pass'] for stage in ('accuracy', 'stability') for row in result.get(stage, [])):
        raise SystemExit('Accuracy/stability acceptance failed; see report.json')


if __name__ == '__main__':
    main()
