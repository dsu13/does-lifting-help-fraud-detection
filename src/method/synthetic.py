"""Planted camouflage model D(mu) with decoys and degree matching (paper: def:dmu).

Structure is informative but not identifiable from features, cell size, provenance or degree;
train-label coherence is the obvious (not the only) way to tell true rings from decoys.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from ..graph.schema import MultiRelationGraph, CandidateCells

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SyntheticConfig:
    n: int = 5000          # nodes
    d: int = 16            # feature dimension
    k: int = 10            # cell size
    pi1: float = 0.05      # fraud prevalence
    delta: float = 1.0     # class-mean separation ||mu1 - mu0||
    sigma: float = 1.0     # feature noise
    mu: float = 0.9        # camouflage strength of the planted fraud cell
    rho: float = 0.7       # fraud fraction of true rings
    n_decoys: int = 4      # feature-kNN decoy/social cells per node
    seed: int = 0


def generate_camouflage_instance(cfg: SyntheticConfig) -> Tuple[MultiRelationGraph, CandidateCells]:
    rng = np.random.default_rng(cfg.seed)

    # ---- labels ----
    y = (rng.random(cfg.n) < cfg.pi1).astype(np.int64)
    # repair (def:dmu): if a class is too small to fill the cells, relabel two disjoint random sets
    need_pos, need_neg = cfg.k + 1, (cfg.n_decoys + 1) * (cfg.k - 1) + 1
    if y.sum() < need_pos or (cfg.n - y.sum()) < need_neg:
        idx = rng.choice(cfg.n, size=need_pos + need_neg, replace=False)
        y[idx[:need_pos]] = 1
        y[idx[need_pos:]] = 0
    fraud_ids = np.where(y == 1)[0]
    benign_ids = np.where(y == 0)[0]

    # ---- features: class-conditional Gaussians ----
    direction = rng.normal(size=cfg.d)
    direction /= np.linalg.norm(direction)
    X = rng.normal(scale=cfg.sigma, size=(cfg.n, cfg.d))
    X[y == 1] += cfg.delta * direction

    # ---- kNN pools: benign only for decoys, both classes for social cells ----
    per_node_needed = (cfg.n_decoys + 1) * (cfg.k - 1) + 1   # +1 to drop self
    nn = NearestNeighbors(n_neighbors=min(per_node_needed, benign_ids.size))
    nn.fit(X[benign_ids])
    _, nbr_idx = nn.kneighbors(X)                            # rows: all nodes
    knn_benign = benign_ids[nbr_idx]                         # map to node ids
    knn_all = NearestNeighbors(n_neighbors=min(per_node_needed, cfg.n)).fit(X).kneighbors(X)[1]

    def _knn_chunks(v: int, n_chunks: int, pool_rows: np.ndarray) -> List[np.ndarray]:
        pool = pool_rows[v]
        pool = pool[pool != v][: n_chunks * (cfg.k - 1)]
        return [pool[i * (cfg.k - 1):(i + 1) * (cfg.k - 1)] for i in range(n_chunks)]

    def _others(pool: np.ndarray, exclude: int, size: int) -> np.ndarray:
        pool = pool[pool != exclude]
        return rng.choice(pool, size=min(size, pool.size), replace=False)

    cells: List[np.ndarray] = []
    social: List[Tuple[int, int]] = []       # (cell index, contributing benign node)
    n_camo = n_rings = n_decoy_cells = 0
    n_ring_fraud = int(np.floor(cfg.rho * (cfg.k - 1) + 0.5))   # def:dmu: floor(rho(k-1)+1/2)

    for v in range(cfg.n):
        # (1) planted cell
        if y[v] == 0:
            members = _others(benign_ids, v, cfg.k - 1)
        elif rng.random() < cfg.mu:
            members = _others(benign_ids, v, cfg.k - 1)
            n_camo += 1
        else:
            members = _others(fraud_ids, v, cfg.k - 1)
        cells.append(np.unique(np.concatenate([[v], members])))

        # (2) kNN cells: decoys for fraud (benign neighbours), social cells for benign
        #     (either class); benign gets one extra to match the fraud ring
        if y[v] == 1:
            chunks = _knn_chunks(v, cfg.n_decoys, knn_benign)
        else:
            chunks = _knn_chunks(v, cfg.n_decoys + 1, knn_all)
        for chunk in chunks:
            if chunk.size >= 1:
                if y[v] == 0:
                    social.append((len(cells), v))
                cells.append(np.unique(np.concatenate([[v], chunk])))
                n_decoy_cells += 1

        # (3) impure true ring for fraudsters
        if y[v] == 1:
            f_part = _others(fraud_ids, v, n_ring_fraud)
            b_part = _others(benign_ids, v, cfg.k - 1 - n_ring_fraud)
            cells.append(np.unique(np.concatenate([[v], f_part, b_part])))
            n_rings += 1

    # (4) degree matching: the membership count must not carry the label
    n_moved = _match_fraud_degrees(cells, social, y, fraud_ids, benign_ids, rng)

    rows = np.concatenate(cells)
    cols = np.concatenate([np.full(c.size, j, dtype=np.int64) for j, c in enumerate(cells)])
    incidence = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(cfg.n, len(cells)),
    )

    graph = MultiRelationGraph(
        name=f"synthetic_mu{cfg.mu}_delta{cfg.delta}",
        features=X.astype(np.float32),
        labels=y,
        relations={},
        meta={"config": cfg.__dict__, "n_fraud": int(y.sum()),
              "n_camouflaged_cells": n_camo, "n_ring_cells": n_rings,
              "n_decoy_or_social_cells": n_decoy_cells,
              "n_degree_matching_moves": n_moved},
    )
    candidates = CandidateCells(
        incidence=incidence,
        provenance=np.full(len(cells), 1, dtype=np.int64),   # same provenance, no label leak
        provenance_names={1: "planted_or_ring_or_decoy"},
    )
    logger.info(
        "D(mu): n=%d fraud=%d cells=%d (camo=%d rings[rho=%.1f]=%d decoy/social=%d, "
        "%d degree-matching moves) delta=%.2f mu=%.2f", cfg.n, int(y.sum()), len(cells),
        n_camo, cfg.rho, n_rings, n_decoy_cells, n_moved, cfg.delta, cfg.mu)
    return graph, candidates


def _match_fraud_degrees(cells: List[np.ndarray], social: List[Tuple[int, int]],
                         y: np.ndarray, fraud_ids: np.ndarray, benign_ids: np.ndarray,
                         rng: np.random.Generator) -> int:
    """Stage (4) of def:dmu, in place: give each fraud node the benign |C(v)| at its quantile.

    Only co-member slots of social cells move (never the contributing node), so sizes are
    kept and planted cells, decoys and rings are untouched. Returns the number of moves."""
    n = y.size
    deg = np.zeros(n, dtype=np.int64)
    for c in cells:
        deg[c] += 1
    n1 = fraud_ids.size
    if n1 == 0 or benign_ids.size == 0 or not social:
        return 0
    order = fraud_ids[np.argsort(deg[fraud_ids] + rng.random(n1), kind="stable")]
    target = np.round(np.quantile(deg[benign_ids], (np.arange(n1) + 0.5) / n1)).astype(np.int64)

    benign_slots = [(j, int(u)) for j, v in social for u in cells[j] if u != v and y[u] == 0]
    perm = rng.permutation(len(benign_slots))
    in_social: dict = {}                     # social cells each fraud node is a co-member of
    for j, v in social:
        for u in cells[j]:
            if y[u] == 1:
                in_social.setdefault(int(u), []).append(j)

    moved, bi = 0, 0
    for v, t in zip(order.tolist(), target.tolist()):
        need = t - int(deg[v])
        if need > 0:
            # below target: take random benign slots in cells without v
            got = 0
            while got < need and bi < perm.size:
                j, u = benign_slots[perm[bi]]; bi += 1
                c = cells[j]
                if v in c or u not in c:
                    continue
                cells[j] = np.sort(np.where(c == u, v, c))
                got += 1; moved += 1
            if got < need:
                logger.warning("D(mu) degree matching: benign slots exhausted (%d of %d)", got, need)
        elif need < 0:
            # above target: hand own slots to random benign nodes
            js = list(in_social.get(v, []))
            rng.shuffle(js)
            for j in js[:-need]:
                c = cells[j]
                if v not in c:
                    continue
                while True:
                    w = int(benign_ids[rng.integers(benign_ids.size)])
                    if w not in c:
                        break
                cells[j] = np.sort(np.where(c == v, w, c))
                moved += 1
    return moved
