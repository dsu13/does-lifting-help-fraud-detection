"""Camouflage-aware learnable lifting model (paper: L_theta then f_phi).

Node encoder, cell scorer, Hard-Concrete cell selection, node-cell message passing, readout.
B (n x m) is fixed; the only learned structure is the soft membership a in [0,1]^m.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HardConcrete(nn.Module):
    """Hard-Concrete L0 gates (Louizos et al., 2018), soft membership in [0,1]."""

    def __init__(self, beta: float = 2.0 / 3.0, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.beta, self.gamma, self.zeta = beta, gamma, zeta

    def sample(self, log_alpha: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            u = torch.rand_like(log_alpha).clamp_(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        else:
            s = torch.sigmoid(log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0.0, 1.0)


class HOMPLayer(nn.Module):
    """One round of node-cell message passing on the soft complex."""

    def __init__(self, hid: int, dropout: float = 0.0):
        super().__init__()
        self.lin_cell = nn.Linear(2 * hid, hid)
        self.lin_upd = nn.Linear(2 * hid, hid)
        self.dropout = dropout

    def forward(self, h, c, a, B, B_t, cell_size):
        # node to cell: mean of member states
        m_cell = torch.sparse.mm(B_t, h) / cell_size.unsqueeze(-1)
        c_new = F.relu(self.lin_cell(torch.cat([c, m_cell], dim=-1)))
        c_new = F.dropout(c_new, p=self.dropout, training=self.training)
        # cell to node: a-weighted mean of incident cells
        aw = a.unsqueeze(-1) * c_new                              # (m, hid)
        node_w = torch.sparse.mm(B, a.unsqueeze(-1)).squeeze(-1)  # (n,) sum of a
        m_node = torch.sparse.mm(B, aw) / (node_w.unsqueeze(-1) + 1e-6)
        h_new = F.relu(self.lin_upd(torch.cat([h, m_node], dim=-1)))
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)
        return h_new, c_new


class CamoLiftNet(nn.Module):
    def __init__(self, in_dim: int, struct_dim: int, hid: int = 64,
                 n_layers: int = 2, dropout: float = 0.5, use_lifting: bool = True):
        super().__init__()
        self.use_lifting = use_lifting
        self.encoder = MLP([in_dim, hid, hid], dropout=dropout)
        self.scorer = MLP([hid + struct_dim, hid, 1], dropout=dropout)
        self.hc = HardConcrete()
        self.layers = nn.ModuleList([HOMPLayer(hid, dropout) for _ in range(n_layers)])
        # readout sees Z next to the HOMP output, so it can fall back to features only
        self.readout = nn.Sequential(
            nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1),
        )

    def forward(self, X, B=None, B_t=None, cell_size=None, struct=None):
        """Return (logits[n], a[m]).

        With use_lifting=False the cells are skipped and a is None (arms A/B)."""
        Z = self.encoder(X)
        if not self.use_lifting:
            logits = self.readout(torch.cat([Z, Z], dim=-1)).squeeze(-1)
            return logits, None
        cell_pool = torch.sparse.mm(B_t, Z) / cell_size.unsqueeze(-1)   # (m, hid)
        log_alpha = self.scorer(torch.cat([cell_pool, struct], dim=-1)).squeeze(-1)
        a = self.hc.sample(log_alpha, self.training)                   # (m,)

        h, c = Z, cell_pool
        for layer in self.layers:
            h, c = layer(h, c, a, B, B_t, cell_size)
        logits = self.readout(torch.cat([Z, h], dim=-1)).squeeze(-1)   # (n,)
        return logits, a
