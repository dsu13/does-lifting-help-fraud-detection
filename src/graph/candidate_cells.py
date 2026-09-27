"""Candidate-cell generator C(G) (paper: def:cells; families in sec:benchmark).

Families: shared attribute value, relation neighborhood {v} U N_r(v), feature kNN {v} U kNN_k(v).
Identical node sets are merged; oversized groups keep their highest-degree members.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from .schema import MultiRelationGraph, CandidateCells

try:
    from tqdm import tqdm
except ImportError:  # run without progress bars
    def tqdm(x, **kwargs):  # type: ignore
        return x

logger = logging.getLogger(__name__)

PROVENANCE: Dict[int, str] = {
    0: "shared_attribute",
    1: "relation_group",
    2: "feature_knn",
}


def normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _clamp_group(nodes: np.ndarray, degree: np.ndarray, max_size: int,
                 center: int = None) -> np.ndarray:
    """Cap a group at max_size nodes, keeping the highest-degree ones.

    If `center` is given (closed neighborhood) it is always kept.
    """
    if nodes.size <= max_size:
        return nodes
    if center is None:
        order = np.argsort(-degree[nodes], kind="stable")
        return nodes[order[:max_size]]
    others = nodes[nodes != center]
    order = np.argsort(-degree[others], kind="stable")
    return np.sort(np.concatenate([[center], others[order[:max_size - 1]]]))


def _feature_knn_groups(features: np.ndarray, k: int, n_jobs: int = -1) -> List[np.ndarray]:
    n = features.shape[0]
    k_eff = min(k + 1, n)  # +1 for the query point itself
    X = normalize_rows(features)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", n_jobs=n_jobs)
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    # {v} U kNN_k(v); v is added explicitly because ties can drop it from its own list
    return [np.unique(np.concatenate([[v], idx[v][idx[v] != v][:k_eff - 1]])) for v in range(n)]


def _relation_groups(
    graph: MultiRelationGraph, max_size: int, degree: np.ndarray
) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for r, A in graph.relations.items():
        A = A.tocsr()
        for v in range(graph.num_nodes):
            nbrs = A.indices[A.indptr[v]:A.indptr[v + 1]]
            if nbrs.size == 0:
                continue
            grp = np.unique(np.concatenate([[v], nbrs]))
            groups.append(_clamp_group(grp, degree, max_size, center=v))
    return groups


def _attribute_groups_from_incidence(
    attribute_incidence: sp.csr_matrix, max_size: int, degree: np.ndarray
) -> List[np.ndarray]:
    """One group per column (attribute value) of the incidence."""
    groups: List[np.ndarray] = []
    A = attribute_incidence.tocsc()
    for j in range(A.shape[1]):
        nodes = A.indices[A.indptr[j]:A.indptr[j + 1]]
        if nodes.size == 0:
            continue
        groups.append(_clamp_group(np.unique(nodes), degree, max_size))
    return groups


def _dedup_to_incidence(
    groups_with_prov: List[Tuple[np.ndarray, int]],
    num_nodes: int,
    min_size: int,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Merge duplicate node sets; return the (n x m) incidence and provenance."""
    seen: Dict[tuple, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    prov: List[int] = []
    col = 0
    for nodes, p in tqdm(groups_with_prov, desc="dedup->incidence", unit="grp"):
        nodes = np.unique(nodes)
        if nodes.size < min_size:
            continue
        key = tuple(nodes.tolist())   # sorted tuple, cheaper to hash than a frozenset
        if key in seen:
            continue
        seen[key] = col
        rows.extend(nodes.tolist())
        cols.extend([col] * nodes.size)
        prov.append(p)
        col += 1
    m = col
    data = np.ones(len(rows), dtype=np.float32)
    incidence = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, max(m, 1)))
    if m == 0:
        logger.warning("candidate generator produced 0 cells")
    return incidence, np.asarray(prov, dtype=np.int64)


def _subsample_groups(
    groups: List[Tuple[np.ndarray, int]], budget: int, seed: int
) -> List[Tuple[np.ndarray, int]]:
    """Keep at most `budget` (group, prov) items, chosen with a fixed seed."""
    if budget <= 0:
        return []
    if len(groups) <= budget:
        return groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(groups), size=budget, replace=False)
    idx.sort()
    return [groups[i] for i in idx]


def generate_candidate_cells(
    graph: MultiRelationGraph,
    cfg,
    attribute_incidence: Optional[sp.csr_matrix] = None,
) -> CandidateCells:
    """Build C(G) from the template families; `cfg` is config.CandidateConfig.

    Attribute groups come from `attribute_incidence` when one is given, relation groups from
    every relation. Feature kNN is skipped above `cfg.knn_max_nodes` (brute-force cosine is
    O(n^2)). Relation and kNN groups are subsampled to fit `cfg.max_total_cells`; attribute
    groups are always kept.
    """
    homo = graph.homo_adjacency()
    degree = np.asarray(homo.sum(axis=1)).ravel().astype(np.int64)
    n = graph.num_nodes

    attr_groups: List[Tuple[np.ndarray, int]] = []
    other_groups: List[Tuple[np.ndarray, int]] = []

    if attribute_incidence is not None:
        attr = _attribute_groups_from_incidence(attribute_incidence, cfg.max_cell_size, degree)
        attr_groups += [(g, 0) for g in attr]
        logger.info("candidate: %d shared-attribute groups", len(attr))

    if graph.relations:
        rel = _relation_groups(graph, cfg.max_cell_size, degree)
        other_groups += [(g, 1) for g in rel]
        logger.info("candidate: %d relation groups", len(rel))

    if n > cfg.knn_max_nodes:
        logger.warning(
            "candidate: SKIPPING feature-kNN (n=%d > knn_max_nodes=%d); "
            "brute cosine kNN is O(n^2).", n, cfg.knn_max_nodes,
        )
    else:
        knn = _feature_knn_groups(graph.features, cfg.knn_k, n_jobs=cfg.knn_n_jobs)
        knn = [_clamp_group(g, degree, cfg.max_cell_size) for g in knn]
        other_groups += [(g, 2) for g in knn]
        logger.info("candidate: %d feature-kNN groups", len(knn))

    # cap the pool: keep every attribute group (the fraud-ring cells), subsample the rest
    budget_other = max(cfg.max_total_cells - len(attr_groups), 0)
    if len(other_groups) > budget_other:
        logger.warning(
            "candidate: subsampling relation/kNN groups %d -> %d to respect max_total_cells=%d",
            len(other_groups), budget_other, cfg.max_total_cells,
        )
        other_groups = _subsample_groups(other_groups, budget_other, cfg.subsample_seed)

    groups_with_prov = attr_groups + other_groups
    incidence, prov = _dedup_to_incidence(groups_with_prov, graph.num_nodes, cfg.min_cell_size)
    logger.info(
        "candidate generator: %d unique cells (from %d raw groups)",
        incidence.shape[1], len(groups_with_prov),
    )
    return CandidateCells(incidence=incidence, provenance=prov, provenance_names=PROVENANCE)
