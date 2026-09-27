"""Amazon and YelpChi review-fraud graphs as MultiRelationGraph.

Reads the CARE-GNN Amazon.mat / YelpChi.mat from data/raw/gad/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.sparse as sp

from ..graph.schema import MultiRelationGraph, UNLABELED

logger = logging.getLogger(__name__)

_RELATION_KEYS = {
    "amazon": ["net_upu", "net_usu", "net_uvu"],
    "yelpchi": ["net_rur", "net_rsr", "net_rtr"],
}
_MAT_FILE = {"amazon": "Amazon.mat", "yelpchi": "YelpChi.mat"}
# Amazon's first 3305 users were never annotated (label 0 in the .mat). They stay as
# structure but are left out of the splits, which gives GADBench's 9.5% fraud rate.
_UNLABELED_PREFIX = {"amazon": 3305}


def _mark_unlabeled(dataset: str, labels: np.ndarray) -> np.ndarray:
    k = _UNLABELED_PREFIX.get(dataset, 0)
    if k:
        if int((labels[:k] == 1).sum()) != 0:
            raise ValueError(f"{dataset}: expected no fraud among the first {k} background "
                             f"nodes, found {int((labels[:k] == 1).sum())}; is this the "
                             f"CARE-GNN release?")
        labels = labels.copy()
        labels[:k] = UNLABELED
        logger.info("%s: first %d background nodes marked UNLABELED", dataset, k)
    return labels


def _to_csr(x) -> sp.csr_matrix:
    A = sp.csr_matrix(x).astype(np.float32)
    A.setdiag(0)
    A = ((A + A.T) > 0).astype(np.float32)   # symmetrize and binarize
    A.eliminate_zeros()
    return A


def _load_from_mat(dataset: str, gad_raw: Path) -> MultiRelationGraph:
    from scipy.io import loadmat

    path = gad_raw / _MAT_FILE[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run python scripts/00_download_data.py, "
            "which downloads it or says where to get it."
        )
    logger.info("Loading %s", path)
    mat = loadmat(str(path))

    features = np.asarray(mat["features"].todense() if sp.issparse(mat["features"])
                          else mat["features"], dtype=np.float32)
    labels = _mark_unlabeled(dataset, np.asarray(mat["label"]).ravel().astype(np.int64))

    relations: Dict[str, sp.csr_matrix] = {}
    for key in _RELATION_KEYS[dataset]:
        relations[key.replace("net_", "")] = _to_csr(mat[key])

    graph = MultiRelationGraph(
        name=dataset, features=features, labels=labels, relations=relations,
        meta={"source": "mat", "file": _MAT_FILE[dataset]},
    )
    graph.validate()
    return graph


def load_gad_dataset(dataset: str, gad_raw: Path) -> MultiRelationGraph:
    dataset = dataset.lower()
    if dataset not in _RELATION_KEYS:
        raise ValueError(f"unknown GAD dataset '{dataset}', expected one of {list(_RELATION_KEYS)}")
    return _load_from_mat(dataset, gad_raw)
