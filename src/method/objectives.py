"""Losses, the fraud-conditional regularizer R, and tensor-prep helpers.

Cell purity pi = (f-1)_+ / (ell-1)_+: the fraud share of a fraud member's co-members in
the cell (paper: sec:method).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


# ---- tensor prep ----
# provenance codes: 0 shared_attribute, 1 relation_group, 2 feature_knn, 3 refined,
# 4 temporal, 5 cross_step_knn, 6 same_step_knn (src/data/temporal.py); one slot each
NUM_PROVENANCE = 7


def build_struct_features(candidates, num_provenance: int = NUM_PROVENANCE) -> np.ndarray:
    """Per-cell structural descriptor: [log size, size/max, provenance one-hot],
    plus candidates.cell_features when set."""
    sizes = candidates.cell_sizes().astype(np.float64)
    max_size = max(sizes.max(), 1.0)
    log_size = np.log1p(sizes)[:, None]
    rel_size = (sizes / max_size)[:, None]
    prov = np.zeros((candidates.num_cells, num_provenance), dtype=np.float64)
    p = np.asarray(candidates.provenance, dtype=np.int64)
    if p.size and (p.min() < 0 or p.max() >= num_provenance):
        raise ValueError(f"provenance codes {sorted(set(p.tolist()))} outside 0..{num_provenance - 1}")
    prov[np.arange(candidates.num_cells), p] = 1.0
    parts = [log_size, rel_size, prov]
    extra = getattr(candidates, "cell_features", None)
    if extra is not None:
        parts.append(np.asarray(extra, dtype=np.float64).reshape(candidates.num_cells, -1))
    return np.hstack(parts).astype(np.float32)


def fraud_and_usable_mass(B_cand: sp.csr_matrix, q: np.ndarray, m: np.ndarray):
    """Per-cell soft fraud mass f = B^T q and usable mass ell = B^T m."""
    f = np.asarray(B_cand.T @ q).ravel()
    ell = np.asarray(B_cand.T @ m).ravel()
    return f.astype(np.float32), ell.astype(np.float32)


def purity_from_mass(f: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
    """Co-member fraud purity pi = (f-1)_+ / (ell-1)_+, clamped to [0,1]."""
    num = torch.clamp(f - 1.0, min=0.0)
    den = torch.clamp(ell - 1.0, min=0.0)
    return torch.clamp(num / (den + 1e-6), 0.0, 1.0)


# ---- losses, regularizer, diagnostics ----
def focal_bce(logits, targets, gamma: float = 2.0, pos_weight=None):
    """Focal BCE for class imbalance; gamma=0 gives weighted BCE."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if gamma > 0:
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p)
        bce = (1 - pt).pow(gamma) * bce
    return bce.mean()


def coverage_coupled_regularizer(a, pi, B, q_node, tau: float = 1.0,
                                 gamma_cov: float = 1.0, *, best_pi):
    """Training regularizer: relative purity plus coverage, averaged over train fraud nodes.

    Per node: relu(best_pi - sum(a*pi) / sum(a)) + gamma_cov * relu(tau - sum(a)).
    The relative target spares fraudsters with no pure cell.
    """
    api = (a * pi).unsqueeze(-1)                                  # (m,1)
    cov = torch.sparse.mm(B, api).squeeze(-1)                     # (n,)  sum a*pi
    tot = torch.sparse.mm(B, a.unsqueeze(-1)).squeeze(-1)         # (n,)  sum a
    node_purity = cov / (tot + 1e-6)
    per_node = torch.clamp(best_pi - node_purity, min=0.0) \
        + gamma_cov * torch.clamp(tau - tot, min=0.0)
    denom = q_node.sum() + 1e-6
    return (q_node * per_node).sum() / denom


def density_penalty(a: torch.Tensor, target_density: float) -> torch.Tensor:
    """Soft budget: pull the mean gate toward target_density (avoids all-off and all-on)."""
    return (a.mean() - target_density) ** 2


def fraud_coverage(a, B, q_node) -> float:
    """Fraction of train fraud nodes in at least one active cell (a > 0.5)."""
    hard = (a.detach() > 0.5).float().unsqueeze(-1)
    tot = torch.sparse.mm(B, hard).squeeze(-1)
    is_fraud = q_node > 0
    covered = ((tot > 0) & is_fraud).float().sum()
    return float(covered / (is_fraud.float().sum() + 1e-6))


def selected_cell_fraud_purity(a: torch.Tensor, f_full: torch.Tensor, pi_full: torch.Tensor) -> float:
    """Fraud-mass-weighted purity of the selected cells, from full labels (evaluation only)."""
    w = a.detach() * f_full
    den = w.sum()
    if float(den) <= 0:
        return float("nan")
    return float((w * pi_full).sum() / den)
