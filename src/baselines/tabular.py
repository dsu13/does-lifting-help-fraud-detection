"""Feature-only baselines (rf, mlp): the no-structure floor (paper: sec:setup).

rf fits a small grid on train with balanced class weights and keeps the best validation
AP, with no refit on train+val so val stays clean. mlp uses pos-weighted BCE with early
stopping on val AP.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

from .base import BaselineModel, register
from .metrics import evaluate

logger = logging.getLogger(__name__)


def _val_ap(clf, X_val: np.ndarray, y_val: np.ndarray) -> float:
    col = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
    ap = evaluate(y_val, clf.predict_proba(X_val)[:, col])["ap"]
    return ap if ap == ap else -np.inf          # NaN (degenerate val set) never wins


class _GridSklearnBaseline(BaselineModel):
    """Fit every grid point on train; keep the one with the best validation AP."""

    def __init__(self, make_clf, grid: List[Dict], name: str):
        self.make_clf, self.grid, self.name = make_clf, grid, name
        self.clf = None
        self.selected: Dict = {}

    def fit(self, graph, train_idx, val_idx):
        X_tr, y_tr = graph.features[train_idx], graph.labels[train_idx]
        X_va, y_va = graph.features[val_idx], (graph.labels[val_idx] == 1).astype(int)
        best = -np.inf
        for params in self.grid:
            clf = self.make_clf(**params)
            clf.fit(X_tr, y_tr)
            ap = _val_ap(clf, X_va, y_va)
            logger.debug("[%s] %s -> val AP %.4f", self.name, params, ap)
            if ap > best:
                best, self.clf, self.selected = ap, clf, dict(params)
        logger.info("[%s] selected %s (val AP %.4f)", self.name, self.selected, best)

    def predict_proba(self, graph) -> np.ndarray:
        classes = list(self.clf.classes_)
        return self.clf.predict_proba(graph.features)[:, classes.index(1) if 1 in classes else -1]


@register("rf")
def _rf(**kw) -> BaselineModel:
    seed = kw.get("seed", 0)
    make = lambda **p: RandomForestClassifier(                      # noqa: E731
        n_estimators=300, class_weight="balanced_subsample", n_jobs=-1,
        random_state=seed, **p)
    grid = [{"max_depth": d, "min_samples_leaf": leaf}
            for d in (None, 16) for leaf in (1, 5)]
    return _GridSklearnBaseline(make, grid, name="rf")


class TorchMLPBaseline(BaselineModel):
    """(128, 64) ReLU MLP, positive-class-weighted BCE, Adam, early stop on val AP."""

    name = "mlp"

    def __init__(self, hidden=(128, 64), epochs: int = 200, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 2048, patience: int = 30,
                 seed: int = 0):
        self.hidden, self.epochs, self.lr = tuple(hidden), epochs, lr
        self.weight_decay, self.batch_size, self.patience, self.seed = \
            weight_decay, batch_size, patience, seed
        self.model = None
        self._mu = self._sd = None

    def _std(self, X: np.ndarray) -> np.ndarray:
        return ((X - self._mu) / self._sd).astype(np.float32)

    def fit(self, graph, train_idx, val_idx):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = graph.features
        self._mu = X[train_idx].mean(0)                       # train statistics only
        sd = X[train_idx].std(0); sd[sd < 1e-12] = 1.0
        self._sd = sd
        Xtr = torch.tensor(self._std(X[train_idx]))
        ytr = torch.tensor((graph.labels[train_idx] == 1).astype(np.float32))
        Xva = torch.tensor(self._std(X[val_idx]))
        yva = (graph.labels[val_idx] == 1).astype(int)

        dims = (X.shape[1],) + self.hidden
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.model = nn.Sequential(*layers, nn.Linear(dims[-1], 1))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        n_pos = float(ytr.sum()); n_neg = float(ytr.numel() - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])

        best_ap, best_state, wait = -np.inf, None, 0
        for _ in range(self.epochs):
            self.model.train()
            perm = rng.permutation(Xtr.shape[0])
            for s in range(0, perm.size, self.batch_size):
                b = torch.as_tensor(perm[s:s + self.batch_size])
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    self.model(Xtr[b]).squeeze(-1), ytr[b], pos_weight=pos_weight)
                loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                ap = evaluate(yva, torch.sigmoid(self.model(Xva).squeeze(-1)).numpy())["ap"]
            ap = ap if ap == ap else -np.inf
            if ap > best_ap:
                best_ap, wait = ap, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, graph) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(self._std(graph.features)))
                                 .squeeze(-1)).numpy().astype(np.float64)


@register("mlp")
def _mlp(**kw) -> BaselineModel:
    return TorchMLPBaseline(seed=kw.get("seed", 0))
