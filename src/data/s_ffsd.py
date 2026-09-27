"""S-FFSD loader: transactions -> event graph + attribute-group incidence.

Rows sharing a Source, Target or Location within `link_window` Time units get star edges.
Expects data/raw/s_ffsd/S-FFSD.csv, unzipped from data/S-FFSD.zip of
github.com/AI4Risk/antifraud (Labels 2 = unlabelled).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.schema import MultiRelationGraph, UNLABELED
from .ieee_cis_graph import _grouped_indices, _groups_to_star_edges, _groups_to_incidence

logger = logging.getLogger(__name__)


def _freq(s: pd.Series) -> np.ndarray:
    vc = s.astype(str).value_counts()
    return s.astype(str).map(vc).to_numpy(dtype=np.float64)


def _zscore(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


def load_s_ffsd(raw_dir: Path, link_window: int = 100, max_group_size: int = 50
                ) -> Tuple[MultiRelationGraph, sp.csr_matrix, np.ndarray]:
    path = raw_dir / "S-FFSD.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run python scripts/00_download_data.py, "
            "which downloads it or says where to get it."
        )
    logger.info("reading %s", path)
    df = pd.read_csv(path)
    time_c, src_c, tgt_c, loc_c = "Time", "Source", "Target", "Location"
    typ_c, amt_c, lab_c = "Type", "Amount", "Labels"

    n = len(df)
    time = pd.to_numeric(df[time_c], errors="coerce").fillna(0).to_numpy().astype(np.float64)

    # 0 benign, 1 fraud, anything else (2 or NaN) unlabelled
    raw_lab = pd.to_numeric(df[lab_c], errors="coerce").to_numpy()
    labels = np.full(n, UNLABELED, dtype=np.int64)
    labels[raw_lab == 0] = 0
    labels[raw_lab == 1] = 1

    # log amount + frequency encodings; time is left out so arm A stays time-free
    amount = np.log1p(pd.to_numeric(df[amt_c], errors="coerce").fillna(0).clip(lower=0).to_numpy())
    X = np.column_stack([
        amount,
        _freq(df[typ_c]), _freq(df[loc_c]),
        _freq(df[src_c]), _freq(df[tgt_c]),
    ])
    X = _zscore(X)
    logger.info("s_ffsd: n=%d, d=%d, fraud=%.3f%% (labelled=%d)",
                n, X.shape[1], 100.0 * (labels == 1).mean(), int((labels != UNLABELED).sum()))

    # relations and shared-attribute groups, same construction as IEEE-CIS
    relations: Dict[str, sp.csr_matrix] = {}
    all_groups: List[np.ndarray] = []
    for name, col in [("source", src_c), ("target", tgt_c), ("location", loc_c)]:
        groups = _grouped_indices(df[col], time, link_window, max_group_size)
        relations[name] = _groups_to_star_edges(groups, n)
        all_groups.extend(groups)
        logger.info("relation '%s': %d groups -> %d edges", name, len(groups), int(relations[name].nnz // 2))

    attribute_incidence = _groups_to_incidence(all_groups, n)
    graph = MultiRelationGraph(
        name="s_ffsd", features=X, labels=labels, relations=relations,
        meta={"source": "s-ffsd", "link_window": link_window,
              "n_hyperedges": int(attribute_incidence.shape[1])},
    )
    graph.validate()
    return graph, attribute_incidence, time
