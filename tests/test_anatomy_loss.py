import pytest
import torch

from manotorch.anatomy_loss import AnatomyConstraintLossEE


def legacy_result(loss, angles):
    groups = [("finger_mcp", [1, 4, 10, 7]), ("finger_pip", [2, 5, 11, 8]),
              ("finger_dip", [3, 6, 12, 9]), ("thumb_cmc", [13]), ("thumb_mcp", [14]), ("thumb_pip", [15])]
    return torch.cat([loss._cal_loss_one_joint(angles[:, ids], getattr(loss, name)) for name, ids in groups], 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_vectorized_loss_matches_legacy(dtype, reduction):
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("FP16 validation uses CUDA; PyTorch 2.0 CPU does not implement half-precision ReLU")
    device = "cuda" if dtype == torch.float16 else "cpu"
    loss = AnatomyConstraintLossEE(reduction=reduction)
    loss.setup(thumb_cmc=["+-:30", "+:20,-:7", "+:80,-:3"])
    a = (torch.randn(8, 16, 3, dtype=dtype, device=device) * 0.9).requires_grad_()
    old = legacy_result(loss, a)
    old = old if reduction == "none" else getattr(old, reduction)()
    new = loss(a)
    torch.testing.assert_close(new, old, atol=1e-6 if dtype == torch.float32 else 1e-12 if dtype == torch.float64 else 0.004,
                               rtol=1e-6 if dtype != torch.float16 else 0.002)
    actual = torch.autograd.grad(new.sum(), a, retain_graph=True)[0]
    expected = torch.autograd.grad(old.sum(), a)[0]
    torch.testing.assert_close(actual, expected)
    assert not loss.state_dict()  # limit/index buffers are derived; preserve the old checkpoint format


def test_loss_follows_changed_configuration_and_dtype():
    loss = AnatomyConstraintLossEE(reduction="none")
    loss.setup()
    a = torch.zeros(1, 16, 3, dtype=torch.double)
    a[:, 1, 2] = -0.1
    torch.testing.assert_close(loss(a), legacy_result(loss, a))
    loss.finger_mcp[2] = "+-:45"
    torch.testing.assert_close(loss(a), legacy_result(loss, a))
    assert loss(a).sum() == 0
    torch.testing.assert_close(loss(a.float()), legacy_result(loss, a.float()))
    a[:, 13, 0] = 0.7
    torch.testing.assert_close(loss(a), legacy_result(loss, a), atol=1e-12, rtol=0)


def test_module_dtype_conversion_preserves_limit_precision():
    loss = AnatomyConstraintLossEE(reduction="none")
    loss.setup()
    angles = torch.zeros(1, 16, 3, dtype=torch.double)
    angles[:, 13, 0] = 1
    loss(angles.float())
    loss.double()
    torch.testing.assert_close(loss(angles), legacy_result(loss, angles), atol=1e-12, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_limit_and_epsilon_boundary_gradients_match_legacy(dtype):
    loss = AnatomyConstraintLossEE(reduction="none")
    loss.setup(finger_mcp=["+-:3", "+:3,-:7", "+-:0"])
    upper, lower = torch.tensor(3 / 180 * torch.pi, dtype=dtype), torch.tensor(-7 / 180 * torch.pi, dtype=dtype)
    values = torch.stack([upper, torch.nextafter(upper, torch.full_like(upper, torch.inf)), lower,
                          torch.nextafter(lower, torch.full_like(lower, -torch.inf)),
                          torch.tensor(loss._eps, dtype=dtype), torch.tensor(-loss._eps, dtype=dtype)])
    angles = torch.zeros(len(values), 16, 3, dtype=dtype)
    angles[:, 1, :] = values[:, None]
    angles.requires_grad_()
    actual, expected = loss(angles), legacy_result(loss, angles)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), angles)[0],
                               torch.autograd.grad(expected.sum(), angles)[0], atol=0, rtol=0)
