"""IEEE-CIS: transactions -> multi-relation graph + attribute-group incidence.

Each linking attribute gives a relation (star edges per shared value and time window) and one
hyperedge per window.
Reads train_transaction.csv and train_identity.csv from data/raw/ieee_cis/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.schema import MultiRelationGraph

logger = logging.getLogger(__name__)


# ---- loading and features ----
def _load_raw(raw_dir: Path) -> pd.DataFrame:
    tx_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"
    for path in (tx_path, id_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run python scripts/00_download_data.py, "
                "which downloads it or says where to get it."
            )
    logger.info("Reading %s", tx_path)
    tx = pd.read_csv(tx_path)
    logger.info("Reading %s", id_path)
    idf = pd.read_csv(id_path)
    tx = tx.merge(idf, on="TransactionID", how="left")
    logger.info("IEEE-CIS raw shape: %s", tx.shape)
    return tx


def _build_card_combo(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.Series:
    combo = df[list(cols)].astype("string").fillna("NA").agg("|".join, axis=1)
    return combo.rename("card_combo")


def _select_numeric_features(
    df: pd.DataFrame, exclude: List[str], max_features: int, na_fraction_drop: float
) -> Tuple[np.ndarray, List[str]]:
    """Keep the most-populated numeric columns; median-impute; z-score."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in exclude]
    na_frac = df[numeric].isna().mean()
    numeric = [c for c in numeric if na_frac[c] <= na_fraction_drop]
    # keep the top-N most populated
    numeric = sorted(numeric, key=lambda c: na_frac[c])[:max_features]
    # copy=True: with copy-on-write this can be a read-only view, and X is imputed in place
    X = df[numeric].to_numpy(dtype=np.float64, copy=True)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    logger.info("IEEE-CIS: kept %d numeric feature columns", len(numeric))
    return X.astype(np.float32), numeric


# ---- shared-value groups -> edges and hyperedges ----
def _grouped_indices(
    values: pd.Series,
    time: np.ndarray,
    time_window: int,
    max_group_size: int,
) -> List[np.ndarray]:
    """Node-index groups for one attribute column.

    Each value's nodes are split into time windows of width `time_window`; each window of
    at least 2 nodes keeps its earliest `max_group_size` members.
    """
    groups: List[np.ndarray] = []
    s = values.fillna("__NA__").astype("string").to_numpy()
    # bucket node indices by value
    by_val: Dict[str, List[int]] = {}
    for i in range(s.shape[0]):
        v = s[i]
        if v == "__NA__":
            continue
        by_val.setdefault(v, []).append(i)

    for _, idxs in by_val.items():
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        if idxs_arr.size < 2:
            continue
        t = time[idxs_arr]
        o = np.argsort(t, kind="stable")
        idxs_sorted = idxs_arr[o]
        t_sorted = t[o]
        start = 0
        for end in range(1, idxs_sorted.size + 1):
            if end == idxs_sorted.size or (t_sorted[end] - t_sorted[start]) > time_window:
                win = idxs_sorted[start:end]
                if win.size >= 2:
                    groups.append(win[:max_group_size])
                start = end
    return groups


def _groups_to_star_edges(groups: List[np.ndarray], n: int) -> sp.csr_matrix:
    """Star adjacency: every member linked to the group's first member (O(g) edges, not a clique)."""
    rows: List[int] = []
    cols: List[int] = []
    for g in groups:
        if g.size < 2:
            continue
        hub = int(g[0])
        for v in g[1:]:
            rows.append(hub); cols.append(int(v))
            rows.append(int(v)); cols.append(hub)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = (A > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _groups_to_incidence(all_groups: List[np.ndarray], n: int) -> sp.csr_matrix:
    """Attribute groups as hyperedges: (n x #groups) incidence."""
    rows: List[int] = []
    cols: List[int] = []
    col = 0
    for g in all_groups:
        if g.size < 1:
            continue
        rows.extend(int(v) for v in g)
        cols.extend([col] * g.size)
        col += 1
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, max(col, 1)))


# ---- entry point ----
def build_ieee_cis_graph(raw_dir: Path, cfg) -> Tuple[MultiRelationGraph, sp.csr_matrix, np.ndarray]:
    """Build the IEEE-CIS multi-relation graph and attribute-group incidence.

    Returns (graph, attribute_incidence of shape (n, #groups), TransactionDT per node).
    """
    df = _load_raw(raw_dir)
    ic = cfg.ieee_cis

    # card-combo key (physical card proxy)
    df["card_combo"] = _build_card_combo(df, ic.build_card_combo_from)

    n = len(df)
    labels = df[ic.label_column].to_numpy().astype(np.int64)
    time = df[ic.time_column].to_numpy().astype(np.float64)

    # time is not a feature: arms B and D supply it, so arm A stays time-free
    exclude = [ic.label_column, ic.id_column, ic.time_column]
    features, feat_cols = _select_numeric_features(df, exclude, ic.max_numeric_features, ic.na_fraction_drop)

    relations: Dict[str, sp.csr_matrix] = {}
    all_attr_groups: List[np.ndarray] = []
    for col in ic.link_columns:
        groups = _grouped_indices(
            df[col], time, cfg.candidate.time_window_seconds, ic.max_group_size,
        )
        relations[col] = _groups_to_star_edges(groups, n)
        all_attr_groups.extend(groups)
        logger.info(
            "relation '%s': %d groups -> %d edges", col, len(groups), int(relations[col].nnz // 2)
        )

    attribute_incidence = _groups_to_incidence(all_attr_groups, n)
    logger.info("attribute-group incidence: %d hyperedges", attribute_incidence.shape[1])

    graph = MultiRelationGraph(
        name="ieee_cis",
        features=features,
        labels=labels,
        relations=relations,
        meta={"feature_columns": feat_cols, "link_columns": list(relations.keys())},
    )
    graph.validate()
    return graph, attribute_incidence, time
