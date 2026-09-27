"""DGraph-Fin loader -> MultiRelationGraph plus per-node time (first incident edge).

Expects data/raw/dgraph/dgraphfin.npz (x, y, edge_index, edge_timestamp); y: 1 fraud,
0 normal, 2/3 background (unlabeled). The graph is a labeled-centred sample of max_nodes nodes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from ..graph.schema import MultiRelationGraph, UNLABELED

logger = logging.getLogger(__name__)


def _node_time_from_edges(edge_index: np.ndarray, edge_time: np.ndarray, n: int) -> np.ndarray:
    """Per-node time: earliest incident edge timestamp."""
    t = np.full(n, np.inf, dtype=np.float64)
    np.minimum.at(t, edge_index[0], edge_time)
    np.minimum.at(t, edge_index[1], edge_time)
    if np.isinf(t).any():                       # isolated nodes get the latest time
        finite_max = t[~np.isinf(t)].max() if (~np.isinf(t)).any() else 0.0
        t[np.isinf(t)] = finite_max
    return t


def _subsample(edge_index, y, max_nodes: int, seed: int) -> np.ndarray:
    """Labeled-centred sample of at most max_nodes nodes.

    Labeled nodes are added in seeded random order, each with its 1-hop neighbourhood, until
    the budget is spent. Only whether a node is labeled is used, never its class.
    """
    rng = np.random.default_rng(seed)
    labeled = np.where((y == 0) | (y == 1))[0]
    n = y.size
    # int32 counts: duplicate (u, v) pairs are summed when the matrix is built
    A = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=np.int32),
                       (edge_index[0], edge_index[1])), shape=(n, n))
    A = ((A + A.T) > 0).tocsr()
    keep = np.zeros(n, dtype=bool)
    count = 0
    for v in rng.permutation(labeled):
        if count >= max_nodes:
            break
        if keep[v]:
            continue
        nb = A.indices[A.indptr[v]:A.indptr[v + 1]]
        new = nb[~keep[nb]]
        room = max_nodes - count - 1
        if new.size > room:
            new = rng.choice(new, size=max(room, 0), replace=False)
        keep[v] = True; keep[new] = True
        count += 1 + new.size
    return np.flatnonzero(keep)


def load_dgraph(raw_dir: Path, max_nodes: int,
                seed: int = 0) -> Tuple[MultiRelationGraph, np.ndarray]:
    path = raw_dir / "dgraphfin.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run python scripts/00_download_data.py, "
            "which downloads it or says where to get it."
        )
    logger.info("loading %s", path)
    npz = np.load(path, allow_pickle=True)
    x = np.asarray(npz["x"], dtype=np.float32)
    y_raw = np.asarray(npz["y"]).ravel()
    edge_index = np.asarray(npz["edge_index"]).astype(np.int64).T   # stored as (E, 2)
    edge_time = np.asarray(npz["edge_timestamp"], dtype=np.float64).ravel()
    n_full = x.shape[0]

    # 1 fraud, 0 normal, 2/3 background stay unlabeled
    y = np.full(n_full, UNLABELED, dtype=np.int64)
    y[y_raw == 1] = 1
    y[y_raw == 0] = 0

    # node times from the full edge stream, before subsampling drops edges
    times_full = _node_time_from_edges(edge_index, edge_time, n_full)
    keep = _subsample(edge_index, y, max_nodes, seed)
    remap = -np.ones(n_full, dtype=np.int64)
    remap[keep] = np.arange(keep.size)
    emask = (remap[edge_index[0]] >= 0) & (remap[edge_index[1]] >= 0)
    edge_index = np.vstack([remap[edge_index[0][emask]], remap[edge_index[1][emask]]])
    x, y = x[keep], y[keep]
    logger.info("DGraph subsampled: %d -> %d nodes (%d labeled)",
                n_full, keep.size, int((y != UNLABELED).sum()))
    n = x.shape[0]

    times = times_full[keep]
    A = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=np.float32),
                       (edge_index[0], edge_index[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    A = A.tolil(); A.setdiag(0); A = A.tocsr(); A.eliminate_zeros()

    graph = MultiRelationGraph(
        name="dgraph", features=x, labels=y, relations={"rel": A},
        meta={"source": "dgraph-fin", "n_edges": int(A.nnz // 2),
              "node_time": "min incident edge timestamp (first appearance)"},
    )
    graph.validate()
    return graph, times
