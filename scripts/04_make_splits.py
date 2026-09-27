"""Step 04: 5-seed splits, stratified for Amazon/YelpChi and chronological for IEEE-CIS.

Needs the processed graphs from steps 02/03.
Run: python scripts/04_make_splits.py [datasets...]  (default: amazon yelpchi ieee_cis)
"""
from __future__ import annotations

import logging
import sys
from typing import List

import numpy as np

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.graph.schema import MultiRelationGraph
from src.data.splits import make_and_save_splits

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("make_splits")


def process_one(dataset: str) -> None:
    cfg = config.CONFIG
    ddir = config.DATA_PROCESSED / dataset
    if not (ddir / "graph_meta.json").exists():
        log.warning("skip %s: no processed graph at %s (run step 02/03 first)", dataset, ddir)
        return
    graph = MultiRelationGraph.load(ddir)
    temporal = dataset == "ieee_cis"
    time = np.load(ddir / "time.npz")["transaction_dt"] if temporal else None
    make_and_save_splits(
        graph.labels, ddir / "splits", config.SEEDS, cfg.split, time=time, temporal=temporal
    )
    log.info("%s splits written (temporal=%s)", dataset, temporal)


def main(argv: List[str]) -> int:
    datasets = argv[1:] if len(argv) > 1 else ["amazon", "yelpchi", "ieee_cis"]
    for ds in datasets:
        process_one(ds.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
