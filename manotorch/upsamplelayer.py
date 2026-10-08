import numpy as np
import torch
from torch.nn import Module


class UpSampleLayer(Module):
    """Subdivide each triangle into four using shared edge midpoints.

    Reusing the same faces tensor caches topology (including its in-place version).
    For a fixed mesh, call ``prepare(faces, vertex_count)`` once and then
    ``layer(vertices)``; this snapshot path avoids CPU transfers and supports compilation.
    The cache is bounded to one topology and is excluded from state dictionaries.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("_edge_indices", torch.empty(0, 0, 2, dtype=torch.long), persistent=False)
        self.register_buffer("_subdivided_faces", torch.empty(0, 0, 3, dtype=torch.long), persistent=False)
        self._source_faces = None
        self._source_version = None
        self._vertex_count = None

    @staticmethod
    def _version(faces):
        try:
            return faces._version
        except RuntimeError:  # Inference tensors have no version counter; use prepare() for a stable snapshot.
            return None

    def clear_cache(self):
        self._edge_indices = self._edge_indices.new_empty(0, 0, 2)
        self._subdivided_faces = self._subdivided_faces.new_empty(0, 0, 3)
        self._source_faces = None
        self._source_version = None
        self._vertex_count = None

    def prepare(self, faces, vertex_count):
        """Snapshot a shared (F,3)/(1,F,3) or batched (B,F,3) topology; return this module.

        Preparation may transfer faces to CPU. All subsequent ``layer(vertices)`` calls
        use the snapshot, even if the original faces are edited. Passing faces again
        enables identity/version-based invalidation. Edits through .data or external
        storage bypass PyTorch version counters; re-prepare explicitly after those edits.
        """
        if vertex_count < 0:
            raise ValueError("vertex_count must be nonnegative")
        if faces.ndim not in (2, 3) or faces.shape[-1] != 3:
            raise ValueError("faces must have shape (F,3) or (B,F,3)")
        if faces.dtype not in (torch.int32, torch.int64):
            raise ValueError("faces must use int32 or int64 indices")
        batched = faces.unsqueeze(0) if faces.ndim == 2 else faces
        if not batched.shape[0]:
            raise ValueError("faces must contain at least one topology")
        if batched.stride(0) == 0:
            batched = batched[:1]  # An expanded shared tensor needs only one edge map.
        arrays = batched.detach().cpu().numpy()
        if arrays.size and (arrays.min() < 0 or arrays.max() >= vertex_count):
            raise ValueError("face indices must be within vertex_count")
        topologies = [self.calculate_faces(fs, vertex_count) for fs in arrays]
        if len({edges.shape[0] for edges, _ in topologies}) != 1:
            raise ValueError("batched topologies must have the same number of unique edges")
        # These indices may later be saved by gather's backward, even after inference warmup.
        with torch.inference_mode(False):
            self._edge_indices = torch.from_numpy(np.stack([t[0] for t in topologies])).to(faces.device)
            self._subdivided_faces = torch.from_numpy(np.stack([t[1] for t in topologies])).to(faces.device)
        self._source_faces = faces
        self._source_version = self._version(faces)
        self._vertex_count = vertex_count
        return self

    @staticmethod
    def calculate_faces(faces, vn):
        edges = {}
        new_faces = []

        def get_edge_id(e):
            if e not in edges:
                edges[e] = len(edges)
            return edges[e]

        for f in faces:
            a, b, c = f[0], f[1], f[2]
            e1, e2, e3 = tuple(sorted([a, b])), tuple(sorted([b, c])), tuple(sorted([c, a]))
            x = get_edge_id(e1) + vn
            y = get_edge_id(e2) + vn
            z = get_edge_id(e3) + vn
            new_faces.append(np.array([x, y, z]))
            new_faces.append(np.array([a, x, z]))
            new_faces.append(np.array([b, y, x]))
            new_faces.append(np.array([c, z, y]))

        new_faces = np.asarray(new_faces, dtype=np.int64).reshape(-1, 3)
        new_vertices_idx = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)
        return new_vertices_idx, new_faces

    def forward(self, vertices, faces=None):
        r"""
            *
           / \
          /   \
         /     \
        * ----- *
            |
            *
           / \
          o - o
         / \ / \
        * --o-- *
        """
        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape (B,V,3)")
        batch_size, vertex_count, _ = vertices.shape
        if faces is not None:
            version = self._version(faces)
            if (faces is not self._source_faces or version is None or version != self._source_version
                    or vertex_count != self._vertex_count):
                self.prepare(faces, vertex_count)
        if self._vertex_count is None:
            raise ValueError("pass faces or call prepare() before forward")
        if self._vertex_count != vertex_count:
            raise ValueError("vertices do not match the prepared vertex_count")
        if self._edge_indices.shape[0] not in (1, batch_size):
            raise ValueError("faces batch must be one or match the vertices batch")
        if self._edge_indices.device != vertices.device:
            with torch.inference_mode(False):
                self._edge_indices = self._edge_indices.to(vertices.device)
                self._subdivided_faces = self._subdivided_faces.to(vertices.device)
        edges = self._edge_indices.expand(batch_size, -1, -1)
        indices = edges.reshape(batch_size, -1).unsqueeze(-1).expand(-1, -1, 3)
        new_verts = vertices.gather(1, indices).reshape(batch_size, edges.shape[1], 2, 3).mean(dim=2)
        new_verts = torch.cat([vertices, new_verts], dim=1)
        # Callers may edit returned faces; never expose the cached storage.
        return new_verts, self._subdivided_faces.expand(batch_size, -1, -1).clone()
