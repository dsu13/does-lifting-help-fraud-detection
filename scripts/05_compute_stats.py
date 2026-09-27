"""Step 05: dataset statistics (size, fraud rate, node homophily, fraud->benign, cell purity).

Writes results/<dataset>/stats.json (paper: tab:datasets; the sizes quoted in tab:dynamic).
Run: python scripts/05_compute_stats.py [datasets...]  (default: amazon yelpchi ieee_cis)
"""
from __future__ import annotations

import json
import logging
import sys
from typing import List

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.graph.schema import MultiRelationGraph, CandidateCells
from src.data.stats import compute_stats, save_stats

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("stats")


def process_one(dataset: str) -> None:
    ddir = config.DATA_PROCESSED / dataset
    if not (ddir / "graph_meta.json").exists():
        log.warning("skip %s: no processed graph (run step 02/03 first)", dataset)
        return
    graph = MultiRelationGraph.load(ddir)
    stats = compute_stats(graph, CandidateCells.load(ddir))
    save_stats(stats, config.results_dir(dataset))
    log.info("=== %s ===\n%s", dataset, json.dumps(stats, indent=2))


def main(argv: List[str]) -> int:
    datasets = argv[1:] if len(argv) > 1 else ["amazon", "yelpchi", "ieee_cis"]
    for ds in datasets:
        process_one(ds.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
