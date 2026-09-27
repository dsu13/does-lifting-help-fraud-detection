"""Train/val/test splits per seed: label-stratified by default, chronological for IEEE-CIS.

Each split file holds the index arrays and boolean masks over all n nodes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from ..seeding import make_rng
from ..graph.schema import UNLABELED

logger = logging.getLogger(__name__)


def _masks_from_indices(n: int, train, val, test) -> Dict[str, np.ndarray]:
    def m(idx):
        z = np.zeros(n, dtype=bool)
        z[idx] = True
        return z
    return {
        "train_idx": np.asarray(train, dtype=np.int64),
        "val_idx": np.asarray(val, dtype=np.int64),
        "test_idx": np.asarray(test, dtype=np.int64),
        "train_mask": m(train),
        "val_mask": m(val),
        "test_mask": m(test),
    }


def stratified_split(
    labels: np.ndarray, seed: int, train_ratio: float, val_ratio: float
) -> Dict[str, np.ndarray]:
    """Label-stratified random split over labeled nodes only."""
    rng = make_rng(seed)
    n = labels.shape[0]
    labeled = np.where(labels != UNLABELED)[0]
    train, val, test = [], [], []
    for cls in np.unique(labels[labeled]):
        idx = labeled[labels[labeled] == cls]
        rng.shuffle(idx)
        n_tr = int(round(train_ratio * idx.size))
        n_va = int(round(val_ratio * idx.size))
        train.append(idx[:n_tr])
        val.append(idx[n_tr:n_tr + n_va])
        test.append(idx[n_tr + n_va:])
    train = np.concatenate(train); val = np.concatenate(val); test = np.concatenate(test)
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    logger.info("seed %d stratified: train=%d val=%d test=%d", seed, train.size, val.size, test.size)
    return _masks_from_indices(n, train, val, test)


def temporal_split(
    labels: np.ndarray, time: np.ndarray, seed: int,
    val_fraction: float, test_fraction: float,
) -> Dict[str, np.ndarray]:
    """Chronological split over labeled nodes; the last fractions go to val/test.

    The cut does not depend on the seed; the seed only shuffles the train order.
    """
    n = labels.shape[0]
    labeled = np.where(labels != UNLABELED)[0]
    order = labeled[np.argsort(time[labeled], kind="stable")]
    k = order.size
    n_test = int(round(test_fraction * k))
    n_val = int(round(val_fraction * k))
    train = order[: k - n_val - n_test]
    val = order[k - n_val - n_test: k - n_test]
    test = order[k - n_test:]
    rng = make_rng(seed)
    train = train.copy(); rng.shuffle(train)
    logger.info("seed %d temporal: train=%d val=%d test=%d", seed, train.size, val.size, test.size)
    return _masks_from_indices(n, train, val, test)


def make_and_save_splits(
    labels: np.ndarray, out_dir: Path, seeds, split_cfg,
    time: np.ndarray | None = None, temporal: bool = False,
) -> None:
    if temporal and time is None:
        raise ValueError("temporal split needs time")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if temporal:
            d = temporal_split(
                labels, time, seed, split_cfg.temporal_val_fraction, split_cfg.temporal_test_fraction
            )
        else:
            d = stratified_split(labels, seed, split_cfg.train_ratio, split_cfg.val_ratio)
        np.savez_compressed(out_dir / f"seed{seed}.npz", **d)
    logger.info("saved %d split files to %s (temporal=%s)", len(list(seeds)), out_dir, temporal)
