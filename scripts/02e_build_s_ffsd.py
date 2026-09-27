"""Step 02e: build S-FFSD (card fraud), one node per transaction.

Star edges within shared Source/Target/Location groups. Writes graph, time.npz,
candidate cells (with the shared-attribute family) and 5-seed splits.
Run: python scripts/02e_build_s_ffsd.py  (expects data/raw/s_ffsd/S-FFSD.csv)
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
from src.data.s_ffsd import load_s_ffsd
from src.data.splits import make_and_save_splits
from src.graph.candidate_cells import generate_candidate_cells

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("build_s_ffsd")


def main() -> int:
    set_seed(config.SEEDS[0])
    cfg = config.CONFIG
    out = config.processed_dir("s_ffsd")

    graph, attr_inc, times = load_s_ffsd(config.SFFSD_RAW)
    graph.save(out)
    np.savez_compressed(out / "time.npz", transaction_dt=times)
    log.info("S-FFSD node-time range: [%.3g, %.3g] (span %.3g)",
             float(times.min()), float(times.max()), float(times.max() - times.min()))

    cands = generate_candidate_cells(graph, cfg.candidate, attribute_incidence=attr_inc)
    cands.save(out)
    make_and_save_splits(graph.labels, out / "splits", config.SEEDS, cfg.split)
    log.info("S-FFSD ready: n=%d, cells=%d -> %s", graph.num_nodes, cands.num_cells, out)
    log.info("Next: the s_ffsd crossover commands in `python main.py --list`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
