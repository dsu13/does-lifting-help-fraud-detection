"""Purity-aware candidate refinement: carve purer sub-cells using train labels only.

Label mode adds fraud-side sub-cells and train-fraud co-occurrence components (provenance 3);
'unsup' and 'random' are count-matched label-free controls.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..graph.candidate_cells import normalize_rows
from ..graph.schema import CandidateCells

logger = logging.getLogger(__name__)

PROVENANCE_REFINED = 3


def per_node_best_purity(incidence: sp.spmatrix, pi: np.ndarray) -> np.ndarray:
    """best_pi[v] = max purity over the cells containing v (0 if none)."""
    coo = incidence.tocoo()
    best = np.zeros(incidence.shape[0], dtype=np.float64)
    np.maximum.at(best, coo.row, pi[coo.col].astype(np.float64))
    return best


def _fraud_side_subcells(
    B_csc: sp.csc_matrix, Xn: np.ndarray, q: np.ndarray,
    sim_percentile: float, max_cell_size: int, min_cell_size: int, max_new: int,
) -> List[np.ndarray]:
    """Fraud-side sub-cells: train fraudsters plus the co-members closest to their centroid.

    Unlabeled co-members stay in, so the fraud signal can reach them."""
    fraud_mass = np.asarray(B_csc.T @ q).ravel()
    cols = np.where(fraud_mass >= 1.0)[0]
    out: List[np.ndarray] = []
    for j in cols:
        nodes = B_csc.indices[B_csc.indptr[j]:B_csc.indptr[j + 1]]
        is_f = q[nodes] > 0
        f_nodes = nodes[is_f]
        others = nodes[~is_f]
        if f_nodes.size == 0:
            continue
        if others.size:
            centroid = Xn[f_nodes].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 0:
                centroid = centroid / nc
            sims = Xn[others] @ centroid
            thr = np.percentile(sims, sim_percentile)
            keep = others[sims >= thr]
        else:
            keep = others
        sub = np.unique(np.concatenate([f_nodes, keep]))
        # strict, non-trivial sub-cells only
        if min_cell_size <= sub.size < nodes.size and sub.size <= max_cell_size:
            out.append(sub)
        if len(out) >= max_new:
            break
    return out


def _fraud_cooccurrence_cells(
    B_csr: sp.csr_matrix, q: np.ndarray, max_cell_size: int, min_cell_size: int,
) -> List[np.ndarray]:
    """Connected components of the train-fraud co-occurrence graph (linked if they share a cell)."""
    fraud_rows = np.where(q > 0)[0]
    if fraud_rows.size < 2:
        return []
    F = B_csr[fraud_rows]                      # (n_f, m)
    co = (F @ F.T).tocsr()                     # shared-cell counts among fraudsters
    co.setdiag(0); co.eliminate_zeros()
    n_comp, comp = connected_components(co, directed=False)
    out: List[np.ndarray] = []
    for c in range(n_comp):
        members = fraud_rows[comp == c]
        if members.size < min_cell_size:
            continue
        out.append(members[:max_cell_size])    # deterministic cap
    return out


def _dedup(cells: List[np.ndarray]) -> List[np.ndarray]:
    seen, out = set(), []
    for cell in cells:
        c = np.unique(cell)
        key = tuple(c.tolist())
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def _control_subcells(
    B_csc: sp.csc_matrix, Xn: np.ndarray, n_target: int, mode: str,
    min_cell_size: int, seed: int,
) -> List[np.ndarray]:
    """Label-free control sub-cells: up to n_target unique ones, columns in seeded order.

    'unsup' keeps members above the median cosine similarity to the cell centroid;
    'random' keeps a random half.
    """
    rng = np.random.default_rng(seed)
    sizes = np.diff(B_csc.indptr)
    splittable = np.where(sizes >= 2 * min_cell_size)[0]
    if splittable.size == 0 or n_target <= 0:
        return []
    out: List[np.ndarray] = []
    seen = set()
    for j in rng.permutation(splittable):
        if len(out) >= n_target:
            break
        nodes = B_csc.indices[B_csc.indptr[j]:B_csc.indptr[j + 1]]
        if mode == "unsup":
            centroid = Xn[nodes].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 0:
                centroid = centroid / nc
            sims = Xn[nodes] @ centroid
            keep = nodes[sims >= np.median(sims)]
        else:  # random
            keep = rng.choice(nodes, size=max(nodes.size // 2, min_cell_size), replace=False)
        keep = np.unique(keep)
        key = tuple(keep.tolist())
        if min_cell_size <= keep.size < nodes.size and key not in seen:
            seen.add(key); out.append(keep)
    if len(out) < n_target:
        logger.warning("control '%s' reached only %d of the %d cells the label mode adds",
                       mode, len(out), n_target)
    return out


def refine_candidates(
    candidates: CandidateCells, features: np.ndarray, labels: np.ndarray,
    train_idx: np.ndarray, mode: str = "label", sim_percentile: float = 50.0,
    max_new: int = 50_000, max_cell_size: int = 50, min_cell_size: int = 2,
    seed: int = 0,
) -> Tuple[CandidateCells, int]:
    """Return (augmented CandidateCells, number of cells added).

    mode: 'label' (train labels only), 'unsup' or 'random' (controls), 'none'.
    """
    if mode == "none":
        return candidates, 0
    if mode not in ("label", "unsup", "random"):
        raise ValueError(f"unknown refine mode '{mode}'")

    n = candidates.num_nodes
    q = np.zeros(n, dtype=np.float32)
    tr_fraud = train_idx[labels[train_idx] == 1]
    q[tr_fraud] = 1.0

    Xn = normalize_rows(features.astype(np.float64))
    B_csc = candidates.incidence.tocsc()
    B_csr = candidates.incidence.tocsr()

    label_cells = _dedup(
        _fraud_side_subcells(B_csc, Xn, q, sim_percentile, max_cell_size, min_cell_size, max_new)
        + _fraud_cooccurrence_cells(B_csr, q, max_cell_size, min_cell_size))[:max_new]
    if mode == "label":
        uniq = label_cells
    else:
        # match the number of cells the label mode actually adds
        uniq = _control_subcells(B_csc, Xn, len(label_cells), mode, min_cell_size, seed)
        logger.info("control '%s': %d cells, matched to the label mode's %d",
                    mode, len(uniq), len(label_cells))
    if not uniq:
        logger.warning("refinement added 0 cells (no fraud-containing candidates?)")
        return candidates, 0

    rows = np.concatenate(uniq)
    cols = np.concatenate([np.full(c.size, i, dtype=np.int64) for i, c in enumerate(uniq)])
    B_new = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, len(uniq)))

    augmented = CandidateCells(
        incidence=sp.hstack([candidates.incidence.tocsr(), B_new], format="csr"),
        provenance=np.concatenate([candidates.provenance,
                                   np.full(len(uniq), PROVENANCE_REFINED, dtype=np.int64)]),
        provenance_names={**candidates.provenance_names, PROVENANCE_REFINED: "refined"},
    )
    logger.info("refinement[%s]: +%d cells (%d -> %d)", mode, len(uniq),
                candidates.num_cells, augmented.num_cells)
    return augmented, len(uniq)
