import os
from typing import Optional

import torch
from torch.nn import Module

from .utils.anchorutils import anchor_load, recover_anchor_batch

# anchor definitions shipped with the package
DEFAULT_ANCHOR_ROOT = os.path.join(os.path.dirname(__file__), "assets", "anchor")


class AnchorLayer(Module):
    def __init__(self, anchor_root: Optional[str] = None):
        """
        Args:
            anchor_root: folder with the anchor definitions; defaults to the copy shipped with manotorch.
        """
        super().__init__()

        anchor_root = DEFAULT_ANCHOR_ROOT if anchor_root is None else anchor_root
        face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(anchor_root)
        self.register_buffer("face_vert_idx", torch.from_numpy(face_vert_idx).long().unsqueeze(0))
        self.register_buffer("anchor_weight", torch.from_numpy(anchor_weight).float().unsqueeze(0))
        self.register_buffer("merged_vertex_assignment", torch.from_numpy(merged_vertex_assignment).long())
        self.anchor_mapping = anchor_mapping

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 778, 3]
        """
        return recover_anchor_batch(vertices, self.face_vert_idx, self.anchor_weight)
