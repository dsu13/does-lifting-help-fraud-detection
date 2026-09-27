"""GCN and BWGNN baselines on the homogeneous graph, torch only (no DGL/PyG).

Both train transductively with pos-weighted BCE and early stopping on validation AP.
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..torch_utils import scipy_to_torch_sparse
from .base import BaselineModel, register
from .metrics import evaluate

logger = logging.getLogger(__name__)


def _normalized_adj(A: sp.csr_matrix) -> sp.csr_matrix:
    """Â = D^{-1/2}(A + I)D^{-1/2}."""
    A = A.tocsr().astype(np.float32)
    A = A + sp.eye(A.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = np.power(deg, -0.5, where=deg > 0)
    dinv[deg == 0] = 0.0
    D = sp.diags(dinv)
    return (D @ A @ D).tocoo()


class _GCN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid)
        self.lin2 = nn.Linear(hid, 1)
        self.dropout = dropout

    def forward(self, ahat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(ahat, self.lin1(x))
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(ahat, self.lin2(h))
        return h.squeeze(-1)   # logits, shape (n,)


def _train_standardized(features: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Z-score with train statistics (constant columns map to 0), as for the other models."""
    mu = features[train_idx].mean(0)
    sd = features[train_idx].std(0)
    sd[sd < 1e-12] = 1.0
    return ((features - mu) / sd).astype(np.float32)


class GCNBaseline(BaselineModel):
    name = "gcn"

    def __init__(self, hid: int = 64, epochs: int = 200, lr: float = 1e-2,
                 weight_decay: float = 5e-4, patience: int = 30, seed: int = 0):
        self.hid = hid
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._ahat = None
        self._x = None

    def fit(self, graph, train_idx, val_idx):
        torch.manual_seed(self.seed)
        ahat = scipy_to_torch_sparse(_normalized_adj(graph.homo_adjacency()), self.device)
        x = torch.tensor(_train_standardized(graph.features, train_idx), device=self.device)
        y = torch.tensor((graph.labels == 1).astype(np.float32), device=self.device)
        tr = torch.tensor(train_idx, dtype=torch.long, device=self.device)
        self._ahat, self._x = ahat, x

        self.model = _GCN(x.shape[1], self.hid).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n_pos = float((graph.labels[train_idx] == 1).sum())
        n_neg = float((graph.labels[train_idx] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device)

        best_ap, best_state, wait = -1.0, None, 0
        y_val = (graph.labels[val_idx] == 1).astype(int)
        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            logits = self.model(ahat, x)
            loss = F.binary_cross_entropy_with_logits(logits[tr], y[tr], pos_weight=pos_weight)
            loss.backward()
            opt.step()

            self.model.eval()
            with torch.no_grad():
                proba = torch.sigmoid(self.model(ahat, x)).cpu().numpy()
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
        logger.info("[gcn] best val AP=%.4f", best_ap)

    def predict_proba(self, graph) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(self._ahat, self._x)).cpu().numpy()


@register("gcn")
def _gcn(**kw) -> BaselineModel:
    return GCNBaseline(seed=kw.get("seed", 0))


# ---- BWGNN (Tang et al., ICML 2022), homogeneous variant ----
# Beta-wavelet filters W_i(L) = (L/2)^i (I - L/2)^{C-i}, i = 0..C, concatenated into an MLP.
# The Beta normalization constant is omitted; the MLP absorbs the scale.
def _bwgnn_operators(A: sp.csr_matrix):
    """Return (P, Q) = (L/2, I - L/2) for the symmetric normalized Laplacian."""
    A = A.tocsr().astype(np.float32)
    A = A.copy(); A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv2 = np.zeros_like(deg); nz = deg > 0; dinv2[nz] = deg[nz] ** -0.5
    D = sp.diags(dinv2)
    Ahat = (D @ A @ D).tocsr()
    n = A.shape[0]
    Id = sp.eye(n, format="csr", dtype=np.float32)
    P = ((Id - Ahat) * 0.5).tocsr()   # L/2
    Q = ((Id + Ahat) * 0.5).tocsr()   # I - L/2
    return P, Q


class _BWGNN(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, order: int = 2, dropout: float = 0.5):
        super().__init__()
        self.order = order
        self.lin_in = nn.Linear(in_dim, hid)
        self.out = nn.Sequential(
            nn.Linear((order + 1) * hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1),
        )

    def forward(self, P, Q, x):
        h = F.relu(self.lin_in(x))
        outs = []
        for i in range(self.order + 1):
            z = h
            for _ in range(self.order - i):
                z = torch.sparse.mm(Q, z)
            for _ in range(i):
                z = torch.sparse.mm(P, z)
            outs.append(z)
        return self.out(torch.cat(outs, dim=-1)).squeeze(-1)


class BWGNNBaseline(BaselineModel):
    name = "bwgnn"

    def __init__(self, hid: int = 64, order: int = 2, epochs: int = 200, lr: float = 1e-2,
                 weight_decay: float = 5e-4, patience: int = 30, seed: int = 0):
        self.hid, self.order = hid, order
        self.epochs, self.lr, self.weight_decay, self.patience, self.seed = \
            epochs, lr, weight_decay, patience, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._P = self._Q = self._x = None

    def fit(self, graph, train_idx, val_idx):
        torch.manual_seed(self.seed)
        P, Q = _bwgnn_operators(graph.homo_adjacency())
        self._P = scipy_to_torch_sparse(P, self.device)
        self._Q = scipy_to_torch_sparse(Q, self.device)
        self._x = torch.tensor(_train_standardized(graph.features, train_idx), device=self.device)
        y = torch.tensor((graph.labels == 1).astype(np.float32), device=self.device)
        tr = torch.tensor(train_idx, dtype=torch.long, device=self.device)

        self.model = _BWGNN(self._x.shape[1], self.hid, self.order).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n_pos = float((graph.labels[train_idx] == 1).sum())
        n_neg = float((graph.labels[train_idx] == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device)

        best_ap, best_state, wait = -1.0, None, 0
        y_val = (graph.labels[val_idx] == 1).astype(int)
        for _ in range(self.epochs):
            self.model.train(); opt.zero_grad()
            logits = self.model(self._P, self._Q, self._x)
            loss = F.binary_cross_entropy_with_logits(logits[tr], y[tr], pos_weight=pos_weight)
            loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                proba = torch.sigmoid(self.model(self._P, self._Q, self._x)).cpu().numpy()
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
        logger.info("[bwgnn] best val AP=%.4f", best_ap)

    def predict_proba(self, graph) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(self._P, self._Q, self._x)).cpu().numpy()


@register("bwgnn")
def _bwgnn(**kw) -> BaselineModel:
    return BWGNNBaseline(seed=kw.get("seed", 0))
