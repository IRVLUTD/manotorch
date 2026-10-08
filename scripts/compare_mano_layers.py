"""Compare manotorch with other MANO implementations, against the official chumpy model in float64.

The other implementations are not dependencies of manotorch. Clone them into one folder, and run this script in an
environment that has chumpy (they read the official pickles with it). The reference is the official MANO code, the
`webuser` folder of the MANO download (`assets/mano/webuser`): it is Python 2, so the script loads it with the minimal
syntax changes (cPickle, print, implicit relative imports) and no change to the computation.

    git clone https://github.com/hassony2/manopth.git    # 4f1dcad
    git clone https://github.com/vchoutas/smplx.git      # 1265df7
    git clone https://github.com/lixiny/manotorch.git manotorch_upstream   # a2a70c5
    python scripts/compare_mano_layers.py --thirdparty <that folder> --mano-assets-root assets/mano

Three settings, both hands, random global rotations, articulations, shapes and translations:
- "full": 48 axis-angle values, flat_hand_mean=True (no mean pose added), with a translation;
- "pca15": 3 + 15 PCA coefficients, flat_hand_mean=False (the mean pose is added). Not 45 components: smplx.MANO
  silently turns PCA off when num_pca_comps == 45 and reads the 45 values as axis-angles;
- "rotmat": 16 rotation matrices (smplx.MANOLayer, which never adds the mean pose).
Reported: the largest vertex and joint error, in millimeters, of each implementation in float64 and float32. The
joints compared are the 16 MANO joints; the implementations sample different fingertip vertices.
"""

import argparse
import importlib.util
import os
import re
import sys
import time
import types

import numpy as np
import torch

from manotorch.manolayer import ManoLayer
from manotorch.utils.geometry import axis_angle_to_matrix

MANO16_IN_21 = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]  # MANO joint order within the 21 joints
NCOMPS = 15  # PCA components of the "pca15" setting


def load_official_webuser(webuser_dir):
    """The official (Python 2) MANO webuser modules, with only the syntax changes Python 3 needs."""
    modules = {}
    for name in ("posemapper", "lbs", "verts", "serialization", "smpl_handpca_wrapper_HAND_only"):
        with open(os.path.join(webuser_dir, f"{name}.py")) as f:
            source = f.read()
        source = source.replace("import cPickle as pickle", "import pickle")
        source = re.sub(r"pickle\.load\(open\((\w+)\)\)", r"pickle.load(open(\1, 'rb'), encoding='latin1')", source)
        source = re.sub(r"print '([^']*)'", r"print('\1')", source)
        source = re.sub(r"from webuser\.(\w+) import", r"from \1 import", source)
        module = types.ModuleType(name)
        module.__file__ = os.path.join(webuser_dir, f"{name}.py")
        sys.modules[name] = module  # the files import each other by their bare names
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - user-supplied official source
        modules[name] = module
    return modules["serialization"].load_model, modules["smpl_handpca_wrapper_HAND_only"].load_model


def import_thirdparty(root, webuser_dir):
    """Import manopth, smplx, upstream manotorch (as `manotorch_upstream`) and the official chumpy webuser code."""
    sys.path.insert(0, os.path.join(root, "manopth"))  # manopth also ships a copy of the `mano.webuser` loaders
    sys.path.insert(1, os.path.join(root, "smplx"))
    import manopth.manolayer
    import smplx

    load_official_full, load_official_pca = load_official_webuser(webuser_dir)

    package = os.path.join(root, "manotorch_upstream", "manotorch")
    spec = importlib.util.spec_from_file_location(
        "manotorch_upstream", os.path.join(package, "__init__.py"), submodule_search_locations=[package]
    )
    upstream = importlib.util.module_from_spec(spec)
    sys.modules["manotorch_upstream"] = upstream
    spec.loader.exec_module(upstream)
    import manotorch_upstream.manolayer

    return manopth.manolayer, smplx, manotorch_upstream.manolayer, load_official_full, load_official_pca


def official(load_full, load_pca, model_file, setting, pose, betas, transl):
    """Vertices (B, 778, 3) and 16 joints of the official chumpy model, in float64 and meters."""
    model = load_full(model_file) if setting == "full" else load_pca(model_file, ncomps=NCOMPS, flat_hand_mean=False)
    verts, joints = [], []
    for p, b, t in zip(pose.numpy(), betas.numpy(), transl.numpy(), strict=True):
        model.pose[:] = p
        model.betas[:] = b
        model.trans[:] = t
        verts.append(np.array(model.r))
        joints.append(np.array(model.J_transformed))
    return torch.tensor(np.stack(verts)), torch.tensor(np.stack(joints))


def run_implementations(mods, mano_root, side, setting, pose, betas, transl, dtype, device):
    """{name: (verts, 16 joints, forward time in ms)} in meters, for one setting."""
    yana, smplx, upstream, _, _ = mods
    model_file = os.path.join(mano_root, "models", f"MANO_{side.upper()}.pkl")
    pca = setting == "pca15"
    flat = not pca
    p, b, t = (x.to(device, dtype) for x in (pose, betas, transl))
    out = {}

    def timed(fn):
        fn()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fn()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        return result, (time.perf_counter() - start) * 1e3

    if setting in ("full", "pca15"):
        ours = ManoLayer(side=side, mano_assets_root=mano_root, use_pca=pca, ncomps=NCOMPS, flat_hand_mean=flat)
        ours = ours.to(device, dtype)
        o, ms = timed(lambda: ours(p, b, t))
        out["manotorch (this fork)"] = (o.verts, o.joints[:, MANO16_IN_21], ms)

        up = upstream.ManoLayer(side=side, mano_assets_root=mano_root, use_pca=pca, ncomps=NCOMPS, flat_hand_mean=flat)
        up = up.to(device, dtype)
        o, ms = timed(lambda: up(p, b))
        out["manotorch (upstream)"] = (o.verts + t[:, None], o.joints[:, MANO16_IN_21] + t[:, None], ms)

        mp = yana.ManoLayer(
            side=side,
            mano_root=os.path.join(mano_root, "models"),
            use_pca=pca,
            ncomps=NCOMPS,
            flat_hand_mean=flat,
            center_idx=None,
        ).to(device, dtype)
        (v, j), ms = timed(lambda: mp(p, b, t))
        out["manopth"] = (v / 1000, j[:, MANO16_IN_21] / 1000, ms)

        sm = smplx.MANO(
            model_file, is_rhand=side == "right", use_pca=pca, num_pca_comps=NCOMPS, flat_hand_mean=flat, dtype=dtype
        ).to(device)
        o, ms = timed(lambda: sm(betas=b, global_orient=p[:, :3], hand_pose=p[:, 3:], transl=t))
        out["smplx.MANO"] = (o.vertices, o.joints[:, :16], ms)
    else:  # rotation matrices, flat
        R = axis_angle_to_matrix(p.view(-1, 16, 3))
        sml = smplx.MANOLayer(model_file, is_rhand=side == "right", dtype=dtype).to(device)
        o, ms = timed(lambda: sml(betas=b, global_orient=R[:, :1], hand_pose=R[:, 1:], transl=t))
        out["smplx.MANOLayer"] = (o.vertices, o.joints[:, :16], ms)
        ours = ManoLayer(side=side, mano_assets_root=mano_root).to(device, dtype)
        o, ms = timed(lambda: ours(p, b, t))
        out["manotorch (this fork)"] = (o.verts, o.joints[:, MANO16_IN_21], ms)
    return out


def main(args):
    torch.manual_seed(0)
    mods = import_thirdparty(args.thirdparty, os.path.join(args.mano_assets_root, "webuser"))
    n = args.samples
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for setting in ("full", "pca15", "rotmat"):
        for side in ("right", "left"):
            pose = torch.cat(
                [torch.randn(n, 3) * 0.8, torch.randn(n, 45) * 0.5 if setting != "pca15" else torch.randn(n, NCOMPS)], 1
            )
            pose = pose.double()
            betas, transl = torch.randn(n, 10, dtype=torch.float64), torch.randn(n, 3, dtype=torch.float64) * 0.2
            model_file = os.path.join(args.mano_assets_root, "models", f"MANO_{side.upper()}.pkl")
            ref_v, ref_j = official(
                mods[3], mods[4], model_file, "pca15" if setting == "pca15" else "full", pose, betas, transl
            )
            for dtype in (torch.float64, torch.float32):
                outs = run_implementations(
                    mods, args.mano_assets_root, side, setting, pose, betas, transl, dtype, device
                )
                for name, (v, j, ms) in outs.items():
                    dv = (v.double().cpu() - ref_v).abs().max().item() * 1e3
                    dj = (j.double().cpu() - ref_j).abs().max().item() * 1e3
                    rows.append((setting, side, name, str(dtype).replace("torch.", ""), dv, dj, ms))
    print(
        f"{'setting':7s} {'side':5s} {'implementation':24s} {'dtype':8s} {'max vertex err':>15s} {'max joint err':>14s}"
    )
    for setting, side, name, dtype, dv, dj, _ in rows:
        print(f"{setting:7s} {side:5s} {name:24s} {dtype:8s} {dv:12.2e} mm {dj:11.2e} mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thirdparty", required=True, help="folder with manopth, smplx and manotorch_upstream")
    parser.add_argument("--mano-assets-root", default="assets/mano")
    parser.add_argument("--samples", type=int, default=32, help="hands per setting and side")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    main(args)
