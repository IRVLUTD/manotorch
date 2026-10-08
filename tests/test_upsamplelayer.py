import pytest
import torch

from manotorch.upsamplelayer import UpSampleLayer


def square():
    return torch.tensor([[0, 1, 2], [0, 2, 3]])


def test_shared_edges_and_winding():
    vertices = torch.arange(12, dtype=torch.float64).reshape(1, 4, 3)
    out, faces = UpSampleLayer()(vertices, square())
    pairs = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]])
    assert torch.equal(out, torch.cat([vertices, vertices[:, pairs].mean(2)], 1))
    expected = torch.tensor([[4, 5, 6], [0, 4, 6], [1, 5, 4], [2, 6, 5],
                             [6, 7, 8], [0, 6, 8], [2, 7, 6], [3, 8, 7]])
    assert torch.equal(faces[0], expected)


def test_cache_invalidation_and_output_ownership(monkeypatch):
    layer = UpSampleLayer()
    faces = square().unsqueeze(0)
    vertices = torch.randn(1, 4, 3)
    original = layer.calculate_faces
    calls = []

    def calculate(fs, vn):
        calls.append(vn)
        return original(fs, vn)

    monkeypatch.setattr(layer, "calculate_faces", calculate)
    _, first_faces = layer(vertices, faces)
    first_faces.zero_()
    _, second_faces = layer(vertices, faces)
    assert calls == [4]
    assert torch.count_nonzero(second_faces) > 0
    faces[0, 0, 0] = 3
    layer(vertices, faces)
    assert calls == [4, 4]
    layer(torch.randn(1, 5, 3), faces)
    assert calls == [4, 4, 5]


def test_expanded_batch_prepared_snapshot_and_gradients(monkeypatch):
    layer = UpSampleLayer()
    calls = []
    original = layer.calculate_faces

    def calculate(fs, vn):
        calls.append(vn)
        return original(fs, vn)

    monkeypatch.setattr(layer, "calculate_faces", calculate)
    faces = square().unsqueeze(0).expand(3, -1, -1)
    layer.prepare(faces, 4)
    assert calls == [4]
    faces[0, 0, 0] = 3
    vertices = torch.randn(3, 4, 3, dtype=torch.float64, requires_grad=True)
    out, _ = layer(vertices)
    out.sum().backward()
    expected = torch.tensor([2.5, 2.0, 2.5, 2.0], dtype=torch.float64).view(1, 4, 1).expand_as(vertices)
    assert torch.equal(vertices.grad, expected)
    assert torch.autograd.gradcheck(lambda v: layer(v)[0], (vertices,))
    assert torch.autograd.gradgradcheck(lambda v: layer(v)[0], (vertices,))
    assert layer.state_dict() == {}


def test_distinct_batched_topologies():
    faces = torch.stack([square(), square().flip(1)])
    vertices = torch.randn(2, 4, 3)
    out, fs = UpSampleLayer()(vertices, faces)
    for i in range(2):
        ref, ref_faces = UpSampleLayer()(vertices[i:i+1], faces[i])
        assert torch.equal(out[i], ref[0])
        assert torch.equal(fs[i], ref_faces[0])


def test_empty_topology_and_clear_cache():
    layer = UpSampleLayer().prepare(torch.empty(0, 3, dtype=torch.long), 4)
    vertices = torch.randn(2, 4, 3)
    out, fs = layer(vertices)
    assert torch.equal(out, vertices)
    assert fs.shape == (2, 0, 3)
    layer.clear_cache()
    with pytest.raises(ValueError, match="prepare"):
        layer(vertices)


def test_prepared_dtype_conversion_and_compile():
    layer = UpSampleLayer().prepare(square(), 4).double()
    assert layer._edge_indices.dtype == torch.int64
    vertices = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    expected, expected_faces = layer(vertices)
    out, faces = compiled(vertices)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert torch.equal(faces, expected_faces)
    tangent = torch.randn_like(vertices)
    _, actual_jvp = torch.func.jvp(lambda v: layer(v)[0], (vertices,), (tangent,))
    torch.testing.assert_close(actual_jvp, layer(tangent)[0], atol=0, rtol=0)


def test_inference_tensor_requires_explicit_snapshot(monkeypatch):
    with torch.inference_mode():
        faces = square()
    layer = UpSampleLayer().prepare(faces, 4)
    monkeypatch.setattr(layer, "calculate_faces", lambda *args: pytest.fail("snapshot rebuilt"))
    layer(torch.randn(1, 4, 3))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"))])
def test_inference_warmup_cache_can_be_reused_for_training(device, monkeypatch):
    layer = UpSampleLayer()
    with torch.inference_mode():
        layer.prepare(square(), 4)
        layer(torch.randn(2, 4, 3, device=device))  # Also exercise a cached CPU-to-CUDA transfer.
    monkeypatch.setattr(layer, "calculate_faces", lambda *args: pytest.fail("snapshot rebuilt"))
    vertices = torch.randn(2, 4, 3, device=device, requires_grad=True)
    layer(vertices)[0].sum().backward()
    expected = vertices.new_tensor([2.5, 2.0, 2.5, 2.0]).view(1, 4, 1).expand_as(vertices)
    torch.testing.assert_close(vertices.grad, expected, atol=0, rtol=0)


@pytest.mark.parametrize("faces, message", [
    (torch.zeros(2, 4, dtype=torch.long), "shape"),
    (square().float(), "int32"),
    (torch.tensor([[0, 1, 4]]), "within"),
    (torch.tensor([[0, -1, 2]]), "within"),
])
def test_invalid_faces(faces, message):
    with pytest.raises(ValueError, match=message):
        UpSampleLayer().prepare(faces, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cached_cuda_has_no_host_transfer(monkeypatch):
    faces = square().cuda()
    layer = UpSampleLayer().prepare(faces, 4).cuda()
    vertices = torch.randn(2, 4, 3, device="cuda", requires_grad=True)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *a, **kw: pytest.fail("host transfer"))
    monkeypatch.setattr(torch.Tensor, "numpy", lambda *a, **kw: pytest.fail("NumPy conversion"))
    out, fs = layer(vertices, faces)
    out.sum().backward()
    assert out.is_cuda and fs.is_cuda and torch.isfinite(vertices.grad).all()
