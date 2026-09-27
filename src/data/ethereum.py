"""Ethereum phishing loader (XBlock, Wu et al. 2022) -> address transaction graph.

Input in data/raw/ethereum/: MulDiGraph.pkl, the XBlock MulDiGraph release (1,165 phishers).
The graph was crawled outward from phishers, a bias no sampling here removes.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.schema import MultiRelationGraph, UNLABELED

logger = logging.getLogger(__name__)


def _as_int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _read_pkl(path: Path) -> Tuple[pd.DataFrame, set]:
    """Read the XBlock MulDiGraph.pkl; attribute keys vary between mirrors, so they are guessed."""
    import networkx  # noqa: F401  (required to unpickle the graph object)
    logger.info("loading networkx graph %s (large: several GB / minutes)", path)
    with open(path, "rb") as f:
        G = pickle.load(f)
    node_attrs = next(iter(G.nodes(data=True)))[1] if G.number_of_nodes() else {}
    label_key = next((k for k in ("isp", "label", "phishing", "isphishing", "y", "tag")
                      if k in node_attrs), None)
    if label_key is None and G.number_of_nodes():
        keys = set().union(*(set(d.keys()) for _, d in G.nodes(data=True)))
        for k in keys:
            pos = sum(1 for _, d in G.nodes(data=True) if _as_int(d.get(k)) == 1)
            if 0 < pos < 0.2 * G.number_of_nodes():
                label_key = k
                break
    if label_key is None:
        raise KeyError(f"no phishing-label node attribute found in {path} "
                       f"(node attrs seen: {list(node_attrs.keys())})")
    phishers = {str(n) for n, d in G.nodes(data=True) if _as_int(d.get(label_key)) == 1}
    edge_attrs = next(iter(G.edges(data=True)))[2] if G.number_of_edges() else {}
    amt_key = next((k for k in ("amount", "value", "wei", "eth") if k in edge_attrs), None)
    ts_key = next((k for k in ("timestamp", "time", "block", "blocknumber") if k in edge_attrs), None)
    logger.info("pkl: %d nodes, %d edges; label='%s' (%d phishing), amount='%s', ts='%s'",
                G.number_of_nodes(), G.number_of_edges(), label_key, len(phishers), amt_key, ts_key)
    u, v, amt, ts = [], [], [], []
    for a, b, d in G.edges(data=True):
        u.append(str(a)); v.append(str(b))
        amt.append(float(d.get(amt_key, 0.0)) if amt_key else 0.0)
        ts.append(float(d[ts_key]) if ts_key and d.get(ts_key) is not None else np.nan)
    return pd.DataFrame({"from": u, "to": v, "amount": amt, "ts": ts}), phishers


def _centred_sample(fi, ti, pos_mask, max_nodes: int, seed: int):
    """Centred sample of at most max_nodes nodes; returns (kept indices, centre mask over them).

    Centres are all phishers, then non-phishers in seeded random order, each with its full
    1-hop neighbourhood, so every centre keeps all its transactions. Only centres get labels.
    """
    n = pos_mask.size
    # int32 counts: duplicate (u, v) pairs are summed when the matrix is built
    A = sp.csr_matrix((np.ones(fi.size, dtype=np.int32), (fi, ti)), shape=(n, n))
    A = ((A + A.T) > 0).tocsr()
    keep = np.zeros(n, dtype=bool)
    centre = np.zeros(n, dtype=bool)
    count = 0

    def _add(v) -> int:
        nb = A.indices[A.indptr[v]:A.indptr[v + 1]]
        new = np.unique(np.concatenate([[v], nb]))
        return new[~keep[new]]

    for v in np.flatnonzero(pos_mask):
        new = _add(v)
        keep[new] = True; count += new.size; centre[v] = True
    if count > max_nodes:
        raise ValueError(f"the phishers and their neighbourhoods alone take {count} nodes "
                         f"> --max-nodes {max_nodes}; raise --max-nodes")
    n_pos_part = count
    rng = np.random.default_rng(seed)
    n_benign = 0
    for v in rng.permutation(np.flatnonzero(~pos_mask)):
        new = _add(v)
        # stop at the first centre that does not fit, so no centre is chosen by neighbourhood size
        if count + new.size > max_nodes:
            break
        keep[new] = True; count += new.size; centre[v] = True; n_benign += 1
    logger.info("ethereum centred sample: %d phishers (+neighbours: %d nodes), %d benign "
                "centres, %d nodes in total, %d labelled (%.2f%% phishing)",
                int(pos_mask.sum()), n_pos_part, n_benign, count, int(centre.sum()),
                100.0 * pos_mask.sum() / max(centre.sum(), 1))
    idx = np.flatnonzero(keep)
    return idx, centre[idx]


def _node_features_and_time(fi, ti, amt, ts, n: int):
    """Five log features (out/in degree, out/in value, total degree) and first tx time per node."""
    out_deg = np.zeros(n); in_deg = np.zeros(n)
    out_amt = np.zeros(n); in_amt = np.zeros(n)
    np.add.at(out_deg, fi, 1.0); np.add.at(in_deg, ti, 1.0)
    np.add.at(out_amt, fi, amt); np.add.at(in_amt, ti, amt)
    X = np.column_stack([
        np.log1p(out_deg), np.log1p(in_deg),
        np.log1p(out_amt), np.log1p(in_amt),
        np.log1p(out_deg + in_deg),
    ]).astype(np.float32)
    if not np.isfinite(ts).any():
        raise ValueError("no edge timestamps found")
    t = np.full(n, np.inf)
    np.minimum.at(t, fi, np.where(np.isfinite(ts), ts, np.inf))
    np.minimum.at(t, ti, np.where(np.isfinite(ts), ts, np.inf))
    t[~np.isfinite(t)] = t[np.isfinite(t)].max()
    return X, t


def load_ethereum(raw_dir: Path, max_nodes: int, seed: int = 0
                  ) -> Tuple[MultiRelationGraph, np.ndarray]:
    pkl = raw_dir / "MulDiGraph.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"{pkl} not found; run python scripts/00_download_data.py, "
            "which downloads it or says where to get it."
        )
    edges, phishers = _read_pkl(pkl)

    addrs = pd.unique(pd.concat([edges["from"], edges["to"]], ignore_index=True))
    id2idx = {a: i for i, a in enumerate(addrs)}
    n0 = len(addrs)
    fi = edges["from"].map(id2idx).to_numpy()
    ti = edges["to"].map(id2idx).to_numpy()
    amt = edges["amount"].to_numpy(dtype=np.float64)
    ts = edges["ts"].to_numpy(dtype=np.float64)
    pos_mask = np.array([a in phishers for a in addrs], dtype=bool)
    logger.info("ethereum: %d addresses, %d edges, %d phishing (%.3f%%)",
                n0, len(edges), int(pos_mask.sum()), 100.0 * pos_mask.mean())

    # features and times from the full edge list, before sampling, so the sample
    # cannot make them differ by class
    X_full, times_full = _node_features_and_time(fi, ti, amt, ts, n0)
    keep, centre = _centred_sample(fi, ti, pos_mask, max_nodes, seed)
    remap = -np.ones(n0, dtype=np.int64); remap[keep] = np.arange(keep.size)
    emask = (remap[fi] >= 0) & (remap[ti] >= 0)
    fi, ti = remap[fi[emask]], remap[ti[emask]]
    pos_mask = pos_mask[keep]
    n = keep.size
    logger.info("ethereum subsampled: %d -> %d nodes (%d phishing)", n0, n, int(pos_mask.sum()))

    X = X_full[keep]
    times = times_full[keep]

    A = sp.csr_matrix((np.ones(fi.size, dtype=np.float32), (fi, ti)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    A = A.tolil(); A.setdiag(0); A = A.tocsr(); A.eliminate_zeros()

    # centres get 0/1, all other nodes are unlabeled
    labels = np.where(centre, pos_mask.astype(np.int64), UNLABELED).astype(np.int64)
    graph = MultiRelationGraph(
        name="ethereum", features=X, labels=labels, relations={"tx": A},
        meta={"source": "xblock-eth-phishing", "n_edges": int(A.nnz // 2)},
    )
    graph.validate()
    return graph, times
