"""Elliptic and Elliptic++ loaders: transaction graph plus the time step (1-49) of each node.

Elliptic expects elliptic_txs_features.csv, _classes.csv and _edgelist.csv in data/raw/elliptic/;
Elliptic++ (github.com/git-disl/EllipticPlusPlus), the same network with augmented features,
expects txs_features.csv, txs_classes.csv and txs_edgelist.csv in data/raw/ellipticpp/.
Labels: '1' (illicit) = 1, '2' (licit) = 0, unknown = UNLABELED.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.schema import MultiRelationGraph, UNLABELED

logger = logging.getLogger(__name__)


def _csvs(raw_dir: Path, prefix: str) -> List[Path]:
    """The features, classes and edgelist CSVs of one release."""
    paths = [raw_dir / f"{prefix}{part}.csv" for part in ("features", "classes", "edgelist")]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found; run python scripts/00_download_data.py, "
                "which downloads it or says where to get it."
            )
    return paths


def _has_header(path: Path) -> bool:
    row0 = pd.read_csv(path, header=None, nrows=1)
    cell = str(row0.iloc[0, 0]).lstrip("-").replace(".", "", 1)
    return not cell.isdigit()


def _pick(cols: List[str], keys, default_idx: int) -> str:
    """First column whose name contains one of `keys`, else cols[default_idx]."""
    low = {c: c.strip().lower() for c in cols}
    for c in cols:
        if any(k in low[c] for k in keys):
            return c
    return cols[default_idx]


def _build(feats_path: Path, classes_path: Path, edges_path: Path, name: str,
           relation: str) -> Tuple[MultiRelationGraph, np.ndarray]:
    logger.info("reading %s", feats_path)
    header = 0 if _has_header(feats_path) else None
    fdf = pd.read_csv(feats_path, header=header)
    cols = list(fdf.columns)
    if header is None:
        id_col, time_col = cols[0], cols[1]
    else:
        id_col = _pick(cols, ("txid", "id"), 0)
        time_col = _pick(cols, ("time", "step"), 1)

    ids = fdf[id_col].astype(str).to_numpy()
    time_step = pd.to_numeric(fdf[time_col], errors="coerce").to_numpy().astype(np.float64)
    feat_cols = [c for c in cols if c not in (id_col, time_col)]
    X = fdf[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all").fillna(0.0).to_numpy().astype(np.float32)
    n = X.shape[0]
    id2idx = {t: i for i, t in enumerate(ids)}
    logger.info("%s: n=%d nodes, d=%d features, id_col=%s time_col=%s",
                name, n, X.shape[1], id_col, time_col)

    # ---- labels ----
    cdf = pd.read_csv(classes_path)
    cdf.columns = [c.strip().lower() for c in cdf.columns]
    idc = _pick(list(cdf.columns), ("txid", "id"), 0)
    clsc = _pick(list(cdf.columns), ("class", "label"), len(cdf.columns) - 1)
    labels = np.full(n, UNLABELED, dtype=np.int64)
    for raw_id, cls in zip(cdf[idc].astype(str).to_numpy(), cdf[clsc].astype(str).to_numpy()):
        i = id2idx.get(raw_id)
        if i is None:
            continue
        if cls == "1":
            labels[i] = 1        # illicit
        elif cls == "2":
            labels[i] = 0        # licit; unknown and '3' stay UNLABELED

    # ---- edges, symmetrized into one relation ----
    edf = pd.read_csv(edges_path)
    a_raw = edf.iloc[:, 0].astype(str).to_numpy()
    b_raw = edf.iloc[:, 1].astype(str).to_numpy()
    src, dst = [], []
    for a, b in zip(a_raw, b_raw):
        ia, ib = id2idx.get(a), id2idx.get(b)
        if ia is None or ib is None:
            continue
        src.append(ia); dst.append(ib)
    A = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    A = A.tolil(); A.setdiag(0); A = A.tocsr(); A.eliminate_zeros()

    graph = MultiRelationGraph(
        name=name, features=X, labels=labels, relations={relation: A},
        meta={"source": name, "n_edges": int(A.nnz // 2),
              "n_timesteps": int(np.nanmax(time_step) - np.nanmin(time_step) + 1)},
    )
    graph.validate()
    return graph, time_step


def load_elliptic(raw_dir: Path) -> Tuple[MultiRelationGraph, np.ndarray]:
    """Return (graph, time_step); time steps are the integers 1-49."""
    paths = _csvs(raw_dir, "elliptic_txs_")
    return _build(*paths, name="elliptic", relation="tx")


def load_ellipticpp(raw_dir: Path) -> Tuple[MultiRelationGraph, np.ndarray]:
    """Return (graph, time_step) of the Elliptic++ transactions network; integer time steps."""
    paths = _csvs(raw_dir, "txs_")
    return _build(*paths, name="ellipticpp", relation="rel")
