"""Step 02f: build the XBlock Ethereum phishing graph (features, node times, cells, 5-seed splits).

~3M addresses, so --max-nodes keeps every phisher and a sample of non-phishers with their
neighbourhoods; only these centres are labelled (src/data/ethereum.py).
Run: python scripts/02f_build_ethereum.py --max-nodes 150000  (MulDiGraph.pkl in data/raw/ethereum/)
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
from src.data.ethereum import load_ethereum
from src.data.splits import make_and_save_splits
from src.graph.candidate_cells import generate_candidate_cells

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("build_ethereum")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Ethereum phishing graph")
    ap.add_argument("--max-nodes", type=int, default=150_000,
                    help="node budget of the centred sample (the full graph is huge)")
    args = ap.parse_args()

    set_seed(config.SEEDS[0])
    cfg = config.CONFIG
    out = config.processed_dir("ethereum")

    graph, times = load_ethereum(config.ETHEREUM_RAW, max_nodes=args.max_nodes, seed=config.SEEDS[0])
    graph.save(out)
    np.savez_compressed(out / "time.npz", transaction_dt=times)
    log.info("Ethereum node-time range: [%.3g, %.3g] (span %.3g)",
             float(times.min()), float(times.max()), float(times.max() - times.min()))

    cands = generate_candidate_cells(graph, cfg.candidate)
    cands.save(out)
    make_and_save_splits(graph.labels, out / "splits", config.SEEDS, cfg.split)
    log.info("Ethereum ready: n=%d, cells=%d -> %s", graph.num_nodes, cands.num_cells, out)
    log.info("Ethereum: %d nodes, %d labelled (%.2f%% phishing)", graph.num_nodes,
             int(graph.labeled_mask().sum()), 100 * graph.fraud_rate())
    log.info("Next: the ethereum crossover commands in `python main.py --list`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
