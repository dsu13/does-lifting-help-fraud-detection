"""Fixed-rule hypergraph baseline (fixed_hg), standing in for the TROPICAL/HCLNet family.

Not a reproduction of either: it keeps every cell of the candidate pool C(G) and runs
HGNN-style node-cell mean message passing with a pos-weighted BCE loss.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..torch_utils import scipy_to_torch_sparse
from .base import BaselineModel, register
from .metrics import evaluate

logger = logging.getLogger(__name__)


class _HGNN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hid)
        self.lins = nn.ModuleList([nn.Linear(2 * hid, hid) for _ in range(n_layers)])
        self.out = nn.Linear(hid, 1)
        self.dropout = dropout

    def forward(self, B, B_t, edge_deg, node_deg, x):
        h = F.relu(self.lin_in(x))
        for lin in self.lins:
            cell = torch.sparse.mm(B_t, h) / edge_deg.unsqueeze(-1)      # mean of member nodes
            m = torch.sparse.mm(B, cell) / node_deg.unsqueeze(-1)        # mean of incident cells
            h = F.relu(lin(torch.cat([h, m], dim=-1)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h).squeeze(-1)


class FixedHypergraphBaseline(BaselineModel):
    name = "fixed_hg"

    def __init__(self, hid: int = 64, n_layers: int = 2, epochs: int = 200, lr: float = 5e-3,
                 weight_decay: float = 5e-4, patience: int = 30, seed: int = 0,
                 candidates=None):
        self.candidates = candidates        # stored C(G) of the dataset
        self.hid, self.n_layers, self.epochs = hid, n_layers, epochs
        self.lr, self.weight_decay, self.patience, self.seed = lr, weight_decay, patience, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._B = self._Bt = self._ed = self._nd = self._x = None

    def fit(self, graph, train_idx, val_idx):
        torch.manual_seed(self.seed)
        if self.candidates is None:
            raise ValueError("fixed_hg needs the dataset's stored candidate pool (candidates=...)")
        B_csr = self.candidates.incidence.tocsr()
        self._B = scipy_to_torch_sparse(B_csr, self.device)
        self._Bt = scipy_to_torch_sparse(B_csr.T, self.device)
        self._ed = torch.tensor(np.asarray(B_csr.sum(0)).ravel(), dtype=torch.float32,
                                device=self.device).clamp_(min=1.0)
        self._nd = torch.tensor(np.asarray(B_csr.sum(1)).ravel(), dtype=torch.float32,
                                device=self.device).clamp_(min=1.0)
        feats = graph.features.astype(np.float64)
        mu = feats[train_idx].mean(0); sd = feats[train_idx].std(0); sd[sd == 0] = 1.0
        self._x = torch.tensor(((feats - mu) / sd).astype(np.float32), device=self.device)
        y = torch.tensor((graph.labels == 1).astype(np.float32), device=self.device)
        tr = torch.tensor(train_idx, dtype=torch.long, device=self.device)

        self.model = _HGNN(self._x.shape[1], self.hid, self.n_layers).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n_pos = float((graph.labels[train_idx] == 1).sum()); n_neg = float((graph.labels[train_idx] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device)
        y_val = (graph.labels[val_idx] == 1).astype(int)

        best_ap, best_state, wait = -1.0, None, 0
        for _ in range(self.epochs):
            self.model.train(); opt.zero_grad()
            logits = self.model(self._B, self._Bt, self._ed, self._nd, self._x)
            F.binary_cross_entropy_with_logits(logits[tr], y[tr], pos_weight=pos_weight).backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                proba = torch.sigmoid(self.model(self._B, self._Bt, self._ed, self._nd, self._x)).cpu().numpy()
            ap = evaluate(y_val, proba[val_idx])["ap"]
            if ap > best_ap:
                best_ap, wait = ap, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        logger.info("[fixed_hg] best val AP=%.4f (cells=%d)", best_ap, B_csr.shape[1])

    def predict_proba(self, graph) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(self._B, self._Bt, self._ed, self._nd, self._x)).cpu().numpy()


@register("fixed_hg")
def _fixed_hg(**kw) -> BaselineModel:
    return FixedHypergraphBaseline(seed=kw.get("seed", 0), candidates=kw.get("candidates"))
