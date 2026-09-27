"""Dataset statistics for tab:datasets and the sizes of tab:dynamic.

Size, fraud rate, node homophily, the benign-neighbor share of fraud nodes and the fraud
purity of the candidate cells. Homophily is computed over labeled node pairs only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.sparse as sp

from ..graph.schema import MultiRelationGraph, CandidateCells, UNLABELED

logger = logging.getLogger(__name__)


# ---- pairwise homophily ----
def _class_neighbor_counts(A: sp.csr_matrix, labels: np.ndarray):
    """Benign and fraud neighbor counts per node, as (neigh0, neigh1)."""
    A = A.tocsr()
    y0 = (labels == 0).astype(np.float32)
    y1 = (labels == 1).astype(np.float32)
    neigh0 = np.asarray(A @ y0).ravel()
    neigh1 = np.asarray(A @ y1).ravel()
    return neigh0, neigh1


def node_homophily(A: sp.csr_matrix, labels: np.ndarray) -> float:
    """Mean over labeled nodes of the same-label fraction among labeled neighbors."""
    neigh0, neigh1 = _class_neighbor_counts(A, labels)
    labeled_neigh = neigh0 + neigh1
    same = np.where(labels == 0, neigh0, neigh1).astype(np.float32)
    valid = (labels != UNLABELED) & (labeled_neigh > 0)
    if not valid.any():
        return float("nan")
    return float((same[valid] / labeled_neigh[valid]).mean())


def fraud_neighbor_heterophily(A: sp.csr_matrix, labels: np.ndarray) -> float:
    """Camouflage proxy: mean fraction of benign neighbors around fraud nodes."""
    neigh0, neigh1 = _class_neighbor_counts(A, labels)
    labeled_neigh = neigh0 + neigh1
    valid = (labels == 1) & (labeled_neigh > 0)
    if not valid.any():
        return float("nan")
    return float((neigh0[valid] / labeled_neigh[valid]).mean())


# ---- cell purity ----
def fraud_cell_purity(incidence: sp.csr_matrix, labels: np.ndarray) -> float:
    """Fraud share of each fraud node's distinct labeled co-members, averaged over fraud nodes.

    The node itself is excluded; nan when no fraud node has a labeled co-member.
    """
    H = incidence.tocsr()
    y0 = (labels == 0).astype(np.float64)
    y1 = (labels == 1).astype(np.float64)
    fraud_idx = np.flatnonzero(labels == 1)
    if not fraud_idx.size:
        return float("nan")
    co = (H[fraud_idx] @ H.T).tocsr()          # fraud rows x all nodes: shared-cell counts
    co.data[:] = 1.0                           # count each co-member once
    me = sp.csr_matrix((np.ones(fraud_idx.size), (np.arange(fraud_idx.size), fraud_idx)),
                       shape=co.shape)         # each fraud row's own column
    co = (co - co.multiply(me)).tocsr()        # drop the node itself
    co.eliminate_zeros()
    n_lab = np.asarray(co @ (y0 + y1)).ravel()
    n_fr = np.asarray(co @ y1).ravel()
    ok = n_lab > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(n_fr[ok] / n_lab[ok]))


# ---- report ----
def compute_stats(graph: MultiRelationGraph, candidates: CandidateCells) -> Dict:
    homo = graph.homo_adjacency()
    return {
        "name": graph.name,
        "num_nodes": graph.num_nodes,
        "num_labeled": int(graph.labeled_mask().sum()),
        "fraud_rate": graph.fraud_rate(),
        "homo": {
            "num_edges": int(homo.nnz // 2),
            "node_homophily": node_homophily(homo, graph.labels),
            "fraud_neighbor_heterophily": fraud_neighbor_heterophily(homo, graph.labels),
        },
        "candidates": {
            "mean_fraud_purity": fraud_cell_purity(candidates.incidence, graph.labels),
        },
    }


def save_stats(stats: Dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("wrote %s", out_dir / "stats.json")
