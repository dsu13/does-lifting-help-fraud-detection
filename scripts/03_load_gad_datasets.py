"""Step 03: load Amazon and YelpChi into the unified schema and build their candidate cells.

Run: python scripts/03_load_gad_datasets.py [amazon] [yelpchi]  (default: both)
"""
from __future__ import annotations

import logging
import sys
from typing import List

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.seeding import set_seed
from src.data.gad_datasets import load_gad_dataset
from src.graph.candidate_cells import generate_candidate_cells

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("load_gad")


def process_one(dataset: str) -> None:
    cfg = config.CONFIG
    out = config.processed_dir(dataset)
    graph = load_gad_dataset(dataset, config.GAD_RAW)
    graph.save(out)
    # review graphs have no attribute groups: relation and feature-kNN cells only
    cands = generate_candidate_cells(graph, cfg.candidate)
    cands.save(out)
    log.info("%s done: n=%d, cells=%d -> %s", dataset, graph.num_nodes, cands.num_cells, out)


def main(argv: List[str]) -> int:
    set_seed(config.SEEDS[0])
    datasets = argv[1:] if len(argv) > 1 else ["amazon", "yelpchi"]
    for ds in datasets:
        process_one(ds.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
