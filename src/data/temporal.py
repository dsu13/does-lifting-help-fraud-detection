"""Temporal cells and features for the crossover (timestamps and structure, no labels).

Arm D splits cells into bursts within a window omega; arm B gets the same cells as node columns.
Both are transductive within omega, not strictly causal. The past-cone variant is causal in
its message passing: each node receives only from strictly earlier co-members (cone_cells,
cone_node_features). Its inputs are not: the candidate cells it cuts and, on Ethereum, the
node features draw on the whole history.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from ..graph.candidate_cells import normalize_rows
from ..graph.schema import MultiRelationGraph, CandidateCells

logger = logging.getLogger(__name__)

PROVENANCE_TEMPORAL = 4
PROVENANCE_CROSS_STEP = 5
PROVENANCE_SAME_STEP = 6

B_COLUMNS = ["log_n_cells", "log_comembers", "mean_cell_size", "mean_cell_span",
             "frac_comembers_earlier"]
CONE_COLUMNS = ["log_n_cones", "log_co_members", "mean_cone_size", "mean_cone_span",
                "mean_nearest_gap"]
TEMPORAL_DESCRIPTOR = ["span_over_window", "gap_burstiness", "time_centre",
                       "frac_simultaneous"]


def _cells_from_lists(cells: List[np.ndarray], n: int, prov: int, name: str) -> CandidateCells:
    if not cells:
        cells = [np.array([0])]             # placeholder so the matrix is never empty
    rows = np.concatenate(cells)
    cols = np.concatenate([np.full(c.size, i, dtype=np.int64) for i, c in enumerate(cells)])
    inc = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                        shape=(n, len(cells)))
    return CandidateCells(
        incidence=inc,
        provenance=np.full(len(cells), prov, dtype=np.int64),
        provenance_names={prov: name},
    )


def temporal_coherence_cells(
    candidates: CandidateCells, times: np.ndarray, window: float,
    min_size: int = 2, max_cells: int = 300_000, seed: int = 0,
) -> CandidateCells:
    """Arm D: split each candidate cell into bursts whose members co-occur within `window`."""
    times = times.astype(np.float64)
    B = candidates.incidence.tocsc()
    sub_cells: List[np.ndarray] = []
    n_split = n_considered = 0
    for j in range(B.shape[1]):
        nodes = B.indices[B.indptr[j]:B.indptr[j + 1]]
        if nodes.size < min_size:
            continue
        n_considered += 1
        order = np.argsort(times[nodes], kind="stable")
        nodes_sorted = nodes[order]; t_sorted = times[nodes_sorted]
        pieces = 0
        start = 0
        for end in range(1, nodes_sorted.size + 1):
            if end == nodes_sorted.size or (t_sorted[end] - t_sorted[start]) > window:
                win = nodes_sorted[start:end]
                if win.size >= min_size:
                    sub_cells.append(np.unique(win))
                pieces += 1
                start = end
        n_split += int(pieces > 1)
    frac = n_split / max(n_considered, 1)
    logger.info("temporal lifting: window=%g split %d of %d cells (%.1f%%)",
                window, n_split, n_considered, 100 * frac)
    seen = set(); uniq: List[np.ndarray] = []
    for c in sub_cells:
        key = tuple(c.tolist())
        if key not in seen:
            seen.add(key); uniq.append(c)
    if len(uniq) > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(uniq), size=max_cells, replace=False))
        uniq = [uniq[i] for i in idx]
    if not uniq:
        logger.warning("temporal lifting produced 0 cells (window too small?)")
    spanning = sum(int(np.ptp(times[c]) > 0) for c in uniq)
    frac_span = spanning / max(len(uniq), 1)
    if spanning == 0:
        logger.warning(
            "NO temporal cell spans more than one time (window=%g, %d cells): arm D carries "
            "no temporal information on this dataset (e.g. Elliptic, where no edge crosses a "
            "time step)", window, len(uniq))
    out = _cells_from_lists(uniq, candidates.num_nodes, PROVENANCE_TEMPORAL, "temporal")
    out.frac_split = frac                   # share of input cells split by the window
    out.frac_spanning = frac_span           # share of output cells spanning more than one time
    logger.info("temporal-coherence lifting: %d cells, %.1f%% span more than one time",
                len(uniq), 100 * frac_span)
    return out


def cell_temporal_features(cells: CandidateCells, times: np.ndarray) -> np.ndarray:
    """Five per-node columns (B_COLUMNS) of a cell pool: arm D's time-cut cells for arm B,
    so B and D differ only in structure vs columns, or the uncut pool for B_static_cols.

    Nodes in no cell get zeros; returned unscaled (z-scored downstream with train statistics).
    """
    times = times.astype(np.float64)
    B = cells.incidence.tocsc()
    n = B.shape[0]
    n_cells = np.zeros(n); co = np.zeros(n); size_sum = np.zeros(n)
    span_sum = np.zeros(n); earlier = np.zeros(n)
    for j in range(B.shape[1]):
        nodes = B.indices[B.indptr[j]:B.indptr[j + 1]]
        s = nodes.size
        if s < 2:
            continue
        t = times[nodes]
        # members strictly earlier than each node in this cell
        earlier_j = np.searchsorted(np.sort(t), t, side="left")
        n_cells[nodes] += 1
        co[nodes] += s - 1
        size_sum[nodes] += s
        span_sum[nodes] += float(t.max() - t.min())
        earlier[nodes] += earlier_j
    has = n_cells > 0
    feats = np.zeros((n, len(B_COLUMNS)), dtype=np.float64)
    feats[:, 0] = np.log1p(n_cells)
    feats[:, 1] = np.log1p(co)
    feats[has, 2] = size_sum[has] / n_cells[has]
    feats[has, 3] = span_sum[has] / n_cells[has]
    feats[has, 4] = np.divide(earlier[has], co[has], out=np.zeros(has.sum()), where=co[has] > 0)
    sd = feats.std(0)
    if (sd < 1e-12).all():
        logger.warning("columns are ALL constant (%d nodes, %d in some cell): they carry no "
                       "information here", n, int(has.sum()))
    logger.info("columns from %d cells: %s (per-column std %s)",
                B.shape[1], B_COLUMNS, np.round(sd, 4).tolist())
    return feats.astype(np.float32)


def cell_temporal_descriptor(cells: CandidateCells, times: np.ndarray,
                             window: float) -> np.ndarray:
    """Temporal shape of each cell for the gate scorer (see TEMPORAL_DESCRIPTOR).

    span / window; burstiness (sd - mean) / (sd + mean) of the gaps between consecutive
    members (0 with fewer than two gaps); centre of the member times within the span
    (0.5 if the span is 0); share of zero gaps. Member times only, never the date itself.
    """
    times = times.astype(np.float64)
    B = cells.incidence.tocsc()
    out = np.zeros((B.shape[1], len(TEMPORAL_DESCRIPTOR)))
    for j in range(B.shape[1]):
        t = np.sort(times[B.indices[B.indptr[j]:B.indptr[j + 1]]])
        if t.size < 2:
            out[j, 2] = 0.5
            continue
        span, gaps = t[-1] - t[0], np.diff(t)
        out[j, 0] = span / window if window > 0 else 0.0
        mu, sd = gaps.mean(), gaps.std()
        if gaps.size >= 2 and sd + mu > 0:
            out[j, 1] = (sd - mu) / (sd + mu)
        out[j, 2] = (t.mean() - t[0]) / span if span > 0 else 0.5
        out[j, 3] = float((gaps == 0).mean())
    logger.info("gate descriptor for %d cells: %s (per-column mean %s)", B.shape[1],
                TEMPORAL_DESCRIPTOR, np.round(out.mean(0), 3).tolist())
    return out.astype(np.float32)


def cone_cells(
    candidates: CandidateCells, times: np.ndarray, window: float, direction: str = "past",
    max_members: int = 49, max_cells: int = 300_000, seed: int = 0,
) -> CandidateCells:
    """Cones: for each cell and member v, v plus its co-members u with
    t_v - window <= t_u < t_v (direction='past', the `max_members` most recent) or
    t_v < t_u <= t_v + window (direction='future', the `max_members` soonest).

    v is the cone's apex and its only receiver (CandidateCells.apex). Past cones are the
    lifting that is causal in its message passing: a node receives from strictly earlier
    nodes only, although the candidate cells they are cut from use the whole history.
    Future cones are their mirror image, a probe of what later co-members carry. Over
    `max_cells`, every apex keeps one random cone first, then a second, and so on.
    """
    if direction not in ("past", "future"):
        raise ValueError(f"direction must be 'past' or 'future', got {direction!r}")
    times = times.astype(np.float64)
    B = candidates.incidence.tocsc()
    seen = set()
    cones: List[np.ndarray] = []
    apexes: List[int] = []
    for j in range(B.shape[1]):
        nodes = B.indices[B.indptr[j]:B.indptr[j + 1]].astype(np.int64)
        if nodes.size < 2:
            continue
        nodes = nodes[np.argsort(times[nodes], kind="stable")]
        t = times[nodes]
        if direction == "past":
            lo = np.searchsorted(t, t - window, side="left")
            hi = np.searchsorted(t, t, side="left")      # members strictly earlier
            lo = np.maximum(lo, hi - max_members)
        else:
            lo = np.searchsorted(t, t, side="right")     # members strictly later
            hi = np.searchsorted(t, t + window, side="right")
            hi = np.minimum(hi, lo + max_members)
        for i in np.flatnonzero(hi > lo):
            v = int(nodes[i])
            cone = np.sort(np.append(nodes[lo[i]:hi[i]], v))
            key = (v, cone.tobytes())
            if key not in seen:
                seen.add(key); cones.append(cone); apexes.append(v)
    if not cones:
        raise ValueError(f"no node has a {direction} co-member within window={window:g} "
                         "(e.g. Elliptic, where every cell lies in one time step)")
    apex = np.asarray(apexes, dtype=np.int64)
    total = apex.size
    n_apex = np.unique(apex).size
    if total > max_cells:
        # rank of each cone among its apex's cones, in a random order
        order = np.random.default_rng(seed).permutation(total)
        by_apex = order[np.argsort(apex[order], kind="stable")]
        starts = np.flatnonzero(np.r_[True, np.diff(apex[by_apex]) != 0])
        rank = np.empty(total, dtype=np.int64)
        rank[by_apex] = np.arange(total) - np.repeat(starts, np.diff(np.r_[starts, total]))
        pos = np.empty(total, dtype=np.int64); pos[order] = np.arange(total)
        keep = np.sort(np.lexsort((pos, rank))[:max_cells])
        cones = [cones[k] for k in keep]
        apex = apex[keep]
        if n_apex > max_cells:
            logger.warning("%d nodes have a cone but the cap is %d: some keep none",
                           n_apex, max_cells)
    # one family per pool, so the cones reuse the temporal slot of the family one-hot
    out = _cells_from_lists(cones, candidates.num_nodes, PROVENANCE_TEMPORAL,
                            f"{direction}_cone")
    out.apex = apex
    out.n_cones_total = int(total)
    out.frac_nodes_with_cone = float(np.unique(apex).size / candidates.num_nodes)
    logger.info("%s cones: %d of %d kept (window=%g), %.1f%% of nodes are an apex, "
                "mean size %.2f", direction, apex.size, total, window,
                100 * out.frac_nodes_with_cone, float(np.mean([c.size for c in cones])))
    return out


def cone_node_features(cones: CandidateCells, times: np.ndarray) -> np.ndarray:
    """Arm B for the cones: per-node columns from the cones it is the apex of (CONE_COLUMNS).

    Only the apex side counts, so a node's past-cone columns involve only its strictly
    earlier co-members (with the cones' input caveat). Nodes that are no apex get zeros;
    returned unscaled (z-scored downstream with train statistics).
    """
    times = times.astype(np.float64)
    B = cones.incidence.tocsc()
    n = B.shape[0]
    cnt = np.zeros(n); others = np.zeros(n); size_sum = np.zeros(n)
    span = np.zeros(n); nearest = np.zeros(n)
    for j in range(B.shape[1]):
        v = cones.apex[j]
        nodes = B.indices[B.indptr[j]:B.indptr[j + 1]]
        gap = np.abs(times[nodes[nodes != v]] - times[v])
        cnt[v] += 1; others[v] += gap.size; size_sum[v] += nodes.size
        span[v] += gap.max(); nearest[v] += gap.min()
    has = cnt > 0
    feats = np.zeros((n, len(CONE_COLUMNS)), dtype=np.float64)
    feats[:, 0] = np.log1p(cnt)
    feats[:, 1] = np.log1p(others)
    feats[has, 2] = size_sum[has] / cnt[has]
    feats[has, 3] = span[has] / cnt[has]
    feats[has, 4] = nearest[has] / cnt[has]
    logger.info("cone columns from %d cones: %s (per-column std %s)", B.shape[1],
                CONE_COLUMNS, np.round(feats.std(0), 4).tolist())
    return feats.astype(np.float32)


def cross_step_knn_cells(
    features: np.ndarray, times: np.ndarray, k: int, window: float, mode: str = "past",
    n_jobs: int = -1, max_distinct_times: int = 1000,
) -> CandidateCells:
    """Time-windowed feature-kNN cells for discrete-step graphs (e.g. Elliptic).

    mode='past': v plus its k cosine neighbours with t_v - window <= t_u < t_v. Built from
                 the past only, but not causal under message passing.
    mode='same': control, neighbours within v's own step. Nodes with no neighbour get no cell.
    """
    from sklearn.neighbors import NearestNeighbors

    if mode not in ("past", "same"):
        raise ValueError(f"mode must be 'past' or 'same', got {mode!r}")
    times = times.astype(np.float64)
    steps = np.unique(times)
    if steps.size > max_distinct_times:
        raise ValueError(f"{steps.size} distinct times: cross_step_knn_cells is meant for "
                         f"discrete time steps (e.g. Elliptic's 49)")
    X = normalize_rows(features.astype(np.float64))
    cells: List[np.ndarray] = []
    for t in steps:
        q = np.flatnonzero(times == t)
        pool = (np.flatnonzero((times >= t - window) & (times < t)) if mode == "past"
                else q)
        kk = min(k + (1 if mode == "same" else 0), pool.size)
        if pool.size == 0 or kk == 0:
            continue
        nn = NearestNeighbors(n_neighbors=kk, metric="cosine", n_jobs=n_jobs).fit(X[pool])
        _, idx = nn.kneighbors(X[q])
        for row, v in enumerate(q):
            nb = pool[idx[row]]
            nb = nb[nb != v][:k]
            if nb.size:
                cells.append(np.unique(np.concatenate([[v], nb])))
    prov, name = ((PROVENANCE_CROSS_STEP, "cross_step_knn") if mode == "past"
                  else (PROVENANCE_SAME_STEP, "same_step_knn"))
    logger.info("%s cells: %d (k=%d, window=%g)", name, len(cells), k, window)
    return _cells_from_lists(cells, features.shape[0], prov, name)


def subsample_cells(cells: CandidateCells, max_cells: int, seed: int = 0) -> CandidateCells:
    """Seeded random subset of at most `max_cells` cells, order preserved."""
    m = cells.num_cells
    if m <= max_cells:
        return cells
    keep = np.sort(np.random.default_rng(seed).choice(m, size=max(max_cells, 0), replace=False))
    logger.info("subsampled %d -> %d cells (seed %d)", m, keep.size, seed)
    return CandidateCells(
        incidence=cells.incidence.tocsc()[:, keep].tocsr(),
        provenance=cells.provenance[keep],
        provenance_names=dict(cells.provenance_names),
    )


def merge_cells(a: CandidateCells, b: CandidateCells) -> CandidateCells:
    """Union of two candidate pools (columns concatenated, duplicates removed)."""
    ca, cb = a.incidence.tocsc(), b.incidence.tocsc()
    seen, cells, prov = set(), [], []
    for C, P in ((ca, a.provenance), (cb, b.provenance)):
        for j in range(C.shape[1]):
            c = np.sort(C.indices[C.indptr[j]:C.indptr[j + 1]])
            key = tuple(c.tolist())
            if c.size >= 2 and key not in seen:
                seen.add(key); cells.append(c); prov.append(int(P[j]))
    if not cells:
        raise ValueError("merge_cells: both pools are empty")
    out = _cells_from_lists(cells, a.num_nodes, 0, "")
    out.provenance = np.asarray(prov, dtype=np.int64)
    out.provenance_names = {**a.provenance_names, **b.provenance_names}
    return out


def add_step_knn_family(
    features: np.ndarray, times: np.ndarray, static: CandidateCells, mode: str, k: int,
    window: float, max_total_cells: int, seed: int = 0, n_jobs: int = -1,
) -> Tuple[CandidateCells, dict]:
    """Static pool plus one step-kNN family ('past' or the 'same' control).

    The family is subsampled to a target that does not depend on `mode`, so both variants
    add the same number of new cells. Returns (merged pool, report dict).
    """
    times = times.astype(np.float64)
    budget = int(max_total_cells) - static.num_cells
    if budget <= 0:
        raise ValueError(f"static pool ({static.num_cells}) already fills max_total_cells "
                         f"({max_total_cells}); raise candidate.max_total_cells")
    steps, counts = np.unique(times, return_counts=True)
    csum = np.concatenate([[0], np.cumsum(counts)])
    # nodes with a non-empty window [t - window, t) get a 'past' cell
    lo = np.searchsorted(steps, steps - window, side="left")
    eligible = int(sum(c for c, a, b in zip(counts, csum[lo], csum[:-1]) if b - a > 0))
    target = min(budget, eligible)
    raw = cross_step_knn_cells(features, times, k=k, window=window, mode=mode, n_jobs=n_jobs)
    # dedup and drop static cells before subsampling, so both modes add `target` new cells
    S = static.incidence.tocsc()
    have = {tuple(np.sort(S.indices[S.indptr[j]:S.indptr[j + 1]]).tolist())
            for j in range(S.shape[1])}
    R = raw.incidence.tocsc()
    keep = []
    for j in range(R.shape[1]):
        key = tuple(np.sort(R.indices[R.indptr[j]:R.indptr[j + 1]]).tolist())
        if key not in have:
            have.add(key); keep.append(j)
    keep = np.asarray(keep, dtype=np.int64)
    extra = CandidateCells(incidence=R[:, keep].tocsr(), provenance=raw.provenance[keep],
                           provenance_names=dict(raw.provenance_names))
    if extra.num_cells < target:
        logger.warning("%s family has %d new cells, below the matched target %d",
                       mode, extra.num_cells, target)
    extra = subsample_cells(extra, target, seed=seed)
    merged = merge_cells(static, extra)
    report = {"mode": mode, "k": int(k), "window": float(window),
              "static_cells": int(static.num_cells), "family_raw_cells": int(raw.num_cells),
              "family_new_unique_cells": int(keep.size), "family_cells": int(extra.num_cells),
              "matched_target": int(target), "merged_cells": int(merged.num_cells)}
    logger.info("step-kNN family: %s", report)
    return merged, report


def subsample_train_positives(labels: np.ndarray, train_idx: np.ndarray,
                              keep_frac: float, seed: int) -> np.ndarray:
    """Train indices with all negatives and only `keep_frac` of the positives (label scarcity)."""
    rng = np.random.default_rng(seed)
    pos = train_idx[labels[train_idx] == 1]
    neg = train_idx[labels[train_idx] == 0]
    k = max(1, int(round(keep_frac * pos.size)))
    kept_pos = rng.choice(pos, size=min(k, pos.size), replace=False)
    out = np.concatenate([kept_pos, neg])
    rng.shuffle(out)
    return out


def augment_features(graph: MultiRelationGraph, extra: np.ndarray) -> MultiRelationGraph:
    """Shallow copy of the graph with `extra` columns appended to the features."""
    return MultiRelationGraph(
        name=graph.name + "+temporal",
        features=np.hstack([graph.features, extra]).astype(np.float32),
        labels=graph.labels,
        relations=graph.relations,
        meta={**graph.meta, "temporal_cols": int(extra.shape[1])},
    )
