"""Step 02c: build DGraph-Fin (graph, node times, candidate cells, 5-seed splits).

Node time is the first incident-edge timestamp. The full graph has ~3.7M nodes, so
--max-nodes takes a subgraph centred on labeled nodes.
Run: python scripts/02c_build_dgraph.py --max-nodes 150000
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.seeding import set_seed
from src.data.dgraph import load_dgraph
from src.data.splits import make_and_save_splits
from src.graph.candidate_cells import generate_candidate_cells

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("build_dgraph")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build DGraph-Fin")
    ap.add_argument("--max-nodes", type=int, default=150_000,
                    help="node budget of the labeled-node-centred sample")
    args = ap.parse_args()

    set_seed(config.SEEDS[0])
    cfg = config.CONFIG
    out = config.processed_dir("dgraph")

    graph, times = load_dgraph(config.DGRAPH_RAW, max_nodes=args.max_nodes, seed=config.SEEDS[0])
    graph.save(out)
    np.savez_compressed(out / "time.npz", transaction_dt=times)
    log.info("DGraph node-time range: [%.3g, %.3g] (span %.3g)",
             float(times.min()), float(times.max()), float(times.max() - times.min()))

    cands = generate_candidate_cells(graph, cfg.candidate)
    cands.save(out)
    make_and_save_splits(graph.labels, out / "splits", config.SEEDS, cfg.split)
    log.info("DGraph ready: n=%d, cells=%d -> %s", graph.num_nodes, cands.num_cells, out)
    log.info("Next: the dgraph crossover commands in `python main.py --list`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
