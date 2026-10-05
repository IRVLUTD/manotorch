"""Compare vendored downstream MANO vertices, 16 joints and gradients without editing downstream repos."""

import argparse
import ast
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer


def load_vendor(root, project):
    if project == "FoundationEgo_Annotation":
        sys.path.insert(0, str(root / "src"))
        from fdego_annot.wrapper.manotorch import _install_shim

        _install_shim()
    package = root / "third_party/manotorch/manotorch"
    alias = "vendor_" + project
    spec = importlib.util.spec_from_file_location(alias, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    from importlib import import_module

    return import_module(alias + ".manolayer").ManoLayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/jikaiwang/Projects/Deepreach_AI"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/downstream.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20261005)
    result = []
    # Deliberately exclude fingertips; the production wrapper replaces them using vertices in both projects.
    joint_ids = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    for project in ["dr_egostereo", "FoundationEgo_Annotation"]:
        root = args.root / project
        vendor_class = load_vendor(root, project)
        for side in ["right", "left"]:
            kwargs = {"side": side, "mano_assets_root": str(root / "weights/mano"),
                      "use_pca": False, "flat_hand_mean": True, "center_idx": None, "ncomps": 45}
            vendor = vendor_class(**kwargs).double()
            if side == "left":
                vendor.th_shapedirs[:, 0, :] *= -1  # existing wrapper's policy
            current = ManoLayer(**kwargs, fix_left_shapedirs=True).double()
            pose = (torch.randn(8, 48, dtype=torch.double) * 0.4).requires_grad_()
            betas = torch.randn(8, 10, dtype=torch.double, requires_grad=True)
            old = vendor(pose, betas)
            new = current(pose, betas)
            old_gradient = torch.autograd.grad(old.verts.square().sum(), (pose, betas), retain_graph=True)
            new_gradient = torch.autograd.grad(new.verts.square().sum(), (pose, betas))
            row = {"project": project, "side": side,
                   "max_vertex_error_mm": (old.verts - new.verts).abs().max().item() * 1000,
                   "max_mano_joint_error_mm": (old.joints[:, joint_ids] - new.joints[:, joint_ids]).abs().max().item() * 1000,
                   "max_gradient_error": max((a - b).abs().max().item() for a, b in zip(old_gradient, new_gradient, strict=True)),
                   "left_policy": "correct shapedirs exactly once", "fingertips": "excluded; wrapper replaces them"}
            result.append(row)
            print(json.dumps(row), flush=True)
        if project == "FoundationEgo_Annotation":
            vendor_fk = importlib.import_module("vendor_" + project + ".axislayer").AxisLayerFK
            vendor_loss = importlib.import_module("vendor_" + project + ".anatomy_loss").AnatomyConstraintLossEE
            source = root / "src/fdego_annot/mano_fitter/anatomy.py"
            tree = ast.parse(source.read_text())
            limits = next(ast.literal_eval(node.value) for node in tree.body
                          if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LIMITS_WIDE"
                                                                 for t in node.targets))
            with torch.no_grad():
                for side in ("right", "left"):
                    kwargs = {"side": side, "mano_assets_root": str(root / "weights/mano")}
                    old_fk, new_fk = vendor_fk(**kwargs), AxisLayerFK(**kwargs)
                    layer = ManoLayer(**kwargs, fix_left_shapedirs=True)
                    p = torch.randn(32, 48) * 0.4
                    transforms = layer(p).transforms_abs
                    old_angles, new_angles = old_fk(transforms)[2], new_fk(transforms)[2]
                    mapped = old_angles.clone()
                    if side == "left":
                        mapped[..., :2] *= -1
                    old_penalty, new_penalty = vendor_loss(), AnatomyConstraintLossEE()
                    old_penalty.setup(**limits)
                    new_penalty.setup(**limits)
                    row = {"project": project, "side": side, "audit": "anatomy convention",
                           "max_mapped_angle_error_rad": (mapped[:, 1:] - new_angles[:, 1:]).abs().max().item(),
                           "wide_loss_old": old_penalty(old_angles).item(), "wide_loss_new": new_penalty(new_angles).item(),
                           "interpretation": "asymmetric left-hand limits need review before migration"}
                    result.append(row)
                    print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
