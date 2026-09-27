"""Step 02: build the IEEE-CIS graph, shared-attribute hyperedges and candidate cells.

Writes data/processed/ieee_cis/, with TransactionDT in time.npz for the temporal split.
Run: python scripts/02_build_ieee_cis.py
"""
from __future__ import annotations

import logging

import numpy as np

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.seeding import set_seed
from src.data.ieee_cis_graph import build_ieee_cis_graph
from src.graph.candidate_cells import generate_candidate_cells

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("build_ieee_cis")


def main() -> int:
    set_seed(config.SEEDS[0])
    cfg = config.CONFIG
    out = config.processed_dir("ieee_cis")

    graph, attr_incidence, time = build_ieee_cis_graph(config.IEEE_CIS_RAW, cfg)
    graph.save(out)
    np.savez_compressed(out / "time.npz", transaction_dt=time)

    cands = generate_candidate_cells(graph, cfg.candidate, attribute_incidence=attr_incidence)
    cands.save(out)
    log.info("IEEE-CIS done: n=%d, cells=%d -> %s", graph.num_nodes, cands.num_cells, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
