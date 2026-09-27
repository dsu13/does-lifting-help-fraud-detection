"""Imbalance-aware metrics for fraud detection (paper: sec:metrics); AP is primary.

recall@K uses K = number of positives, so precision@K = recall@K. best_f1 picks its
threshold on the evaluation set itself, so it is an optimistic operating point.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

METRIC_NAMES: List[str] = ["ap", "roc_auc", "best_f1", "recall@k"]


def _best_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    prec, rec, _ = precision_recall_curve(y_true, scores)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.maximum(denom, 1e-12), 0.0)
    return float(np.max(f1))


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray) -> float:
    k = int(y_true.sum())
    if k == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    topk = order[:k]
    hits = int(y_true[topk].sum())
    return hits / k


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute the full metric suite for one prediction vector.

    y_true in {0,1} and scores = P(fraud), both 1-D over the evaluation nodes.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        # degenerate eval set; AP/ROC undefined
        return {m: float("nan") for m in METRIC_NAMES}
    return {
        "ap": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "best_f1": _best_f1(y_true, scores),
        "recall@k": _recall_at_k(y_true, scores),
    }


def summarize(values) -> Dict[str, object]:
    """Mean, sample std (ddof=1, 0.0 for one seed) and n of one metric across seeds.

    NaN seeds are dropped (counted in n_nan); `values` lists every seed, NaN as None.
    """
    raw = [float(v) if v is not None else float("nan") for v in values]
    arr = np.array([v for v in raw if not np.isnan(v)], dtype=float)
    n = int(arr.size)
    return {
        "mean": float(arr.mean()) if n else float("nan"),
        "std": float(arr.std(ddof=1)) if n > 1 else (0.0 if n == 1 else float("nan")),
        "std_ddof": 1,
        "n": n,
        "n_nan": int(len(raw) - n),
        "values": [None if np.isnan(v) else round(v, 6) for v in raw],
    }


def aggregate_runs(runs: List[Dict[str, float]], keys=None) -> Dict[str, Dict[str, object]]:
    """summarize() every numeric key across a list of per-seed metric dicts."""
    if not runs:
        return {}
    if keys is None:
        keys = []
        for r in runs:          # union of keys: a failed seed may carry fewer
            for k, v in r.items():
                if isinstance(v, (int, float, np.floating)) and k not in keys:
                    keys.append(k)
    return {k: summarize([r.get(k, float("nan")) for r in runs]) for k in keys}


def aggregate(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    """Per-metric summary across seeds for the baseline metric suite."""
    return aggregate_runs(runs, keys=METRIC_NAMES)
