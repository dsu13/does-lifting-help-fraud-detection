"""Step 02b: build Elliptic (graph, candidate cells, time steps, 5-seed splits) for the crossover.

No Elliptic edge crosses a time step, so no static cell spans time. --extra-cells writes
elliptic_xstep (kNN links to the previous step) or elliptic_samestep (count-matched control).
Run: python scripts/02b_build_elliptic.py [--extra-cells xstep|samestep]
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

import sys as _sys
from pathlib import Path as _Path

# repo root on the path, for config and src
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.seeding import set_seed
from src.data.elliptic import load_elliptic
from src.data.splits import make_and_save_splits
from src.graph.candidate_cells import generate_candidate_cells
from src.data.temporal import add_step_knn_family

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("build_elliptic")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Elliptic")
    ap.add_argument("--extra-cells", choices=["none", "xstep", "samestep"], default="none",
                    help="add a label-free step-kNN family to the static pool and write it as "
                         "a separate dataset (<name>_xstep / <name>_samestep): 'xstep' links "
                         "each node to its nearest nodes in the previous step (cells that "
                         "span time), 'samestep' is the count-matched same-step control")
    args = ap.parse_args()

    set_seed(config.SEEDS[0])
    cfg = config.CONFIG
    name = "elliptic" if args.extra_cells == "none" else f"elliptic_{args.extra_cells}"
    out = config.processed_dir(name)

    graph, times = load_elliptic(config.ELLIPTIC_RAW)
    graph.save(out)
    np.savez_compressed(out / "time.npz", transaction_dt=times)

    # no attribute groups, and feature-kNN is skipped (n > knn_max_nodes): relation groups only
    cands = generate_candidate_cells(graph, cfg.candidate)
    if args.extra_cells != "none":
        mode = "past" if args.extra_cells == "xstep" else "same"
        # each node plus its k nearest nodes of the previous step ('past') or its own ('same')
        cands, report = add_step_knn_family(
            graph.features, times, cands, mode=mode,
            k=cfg.candidate.knn_k, window=1.0,
            max_total_cells=cfg.candidate.max_total_cells,
            seed=cfg.candidate.subsample_seed, n_jobs=cfg.candidate.knn_n_jobs)
        # family report goes to results/ (tracked), since the paper reports it
        (config.results_dir(name) / "extra_cells.json").write_text(json.dumps(report, indent=2))
    cands.save(out)

    # stratified 5-seed splits over labeled nodes
    make_and_save_splits(graph.labels, out / "splits", config.SEEDS, cfg.split)

    log.info("%s ready: n=%d, cells=%d -> %s", name, graph.num_nodes, cands.num_cells, out)
    log.info("Next: the %s crossover commands in `python main.py --list`", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
