"""DINOv2 projection head (MLP + L2 normalization + weight-normed linear)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    """3-layer projection MLP followed by L2-norm and a weight-normed linear.

    Used identically by the student and the EMA teacher; outputs are
    interpreted as logits over a fixed set of prototypes.
    """

    def __init__(
        self,
        in_dim,
        out_dim=65536,
        hidden_dim=2048,
        bottleneck_dim=256,
        nlayers=3,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2, eps=1e-6)
        return self.last_layer(x)
