"""Step 07: train the lifting model over refine modes x lambdas, on every seed.

Default is the generation ablation (tab:ablation): lambda=0 with refine modes
label (carving guided by train labels), none (stored pool) and the count-matched
controls unsup and random. Its label run is also the lambda=0 row of tab:selection.

Run:  python scripts/07_train_method.py [--datasets amazon]
      tab:selection (lambda > 0): add --refine-modes label --lambdas 0.5 1 2
      --out-name method_results.json
Writes results/<ds>/ablation_results.json, merged with the entries already there.
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List

import numpy as np

import sys as _sys
from pathlib import Path as _Path

# Repo root (config.py, src/) onto the path, so this runs as `python scripts/<name>.py`.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.seeding import set_seed
from src.graph.schema import MultiRelationGraph, CandidateCells
from src.method import MethodConfig, fit_and_eval
from src.baselines.metrics import aggregate_runs
from src.results_io import merge_json

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("method")


def run_dataset(dataset: str, refine_modes: List[str], lambdas: List[float],
                seeds, epochs: int) -> Dict:
    ddir = config.DATA_PROCESSED / dataset
    if not (ddir / "graph_meta.json").exists() or not (ddir / "candidates_incidence.npz").exists():
        log.warning("skip %s: need processed graph + candidates (steps 02/03/04)", dataset)
        return {}
    graph = MultiRelationGraph.load(ddir)
    candidates = CandidateCells.load(ddir)

    results: Dict[str, Dict] = {}
    for mode in refine_modes:
        for lam in lambdas:
            per_seed = []
            for seed in seeds:
                set_seed(seed)
                split = dict(np.load(ddir / "splits" / f"seed{seed}.npz"))
                cfg = MethodConfig(seed=seed, epochs=epochs, refine_mode=mode)
                try:
                    per_seed.append(fit_and_eval(graph, candidates, split, lam, cfg))
                except Exception as e:  # noqa: BLE001
                    log.exception("[%s refine=%s lam=%.3g seed=%d] failed: %s",
                                  dataset, mode, lam, seed, e)
                    per_seed.append({"ap": float("nan")})   # keep the slot so values stay in seed order
            if any(r["ap"] == r["ap"] for r in per_seed):
                results[f"{mode}|lam{lam:g}"] = aggregate_runs(per_seed)
                results[f"{mode}|lam{lam:g}"]["seeds"] = list(seeds)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Learnable lifting: generation ablation / selection study")
    ap.add_argument("--datasets", nargs="+", default=["amazon", "yelpchi", "ieee_cis"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0],
                    help="regularizer weights (default: 0.0 = the generation ablation)")
    ap.add_argument("--refine-modes", nargs="+", default=["label", "none", "unsup", "random"],
                    choices=["label", "none", "unsup", "random"],
                    help="cell-generation variants to compare")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out-name", default="ablation_results.json",
                    help="output filename (kept separate from method_results.json)")
    args = ap.parse_args()

    for ds in args.datasets:
        ds = ds.lower()
        log.info("=== %s : refine_modes=%s lambdas=%s ===", ds, args.refine_modes, args.lambdas)
        res = run_dataset(ds, args.refine_modes, args.lambdas, args.seeds, args.epochs)
        if not res:
            continue
        out = config.results_dir(ds) / args.out_name
        # merge (under a lock) so rerunning one mode keeps the others
        merged, replaced = merge_json(out, res)
        log.info("wrote %s (updated %s; kept %s)", out, sorted(res),
                 sorted(set(merged) - set(res)) or "nothing else")
        if replaced:
            log.info("  replaced previous results for: %s", replaced)
        log.info("--- %s : %s ---", ds, args.out_name)
        log.info("  %-16s %-16s %-12s %-16s %-12s",
                 "mode|lambda", "test_AP", "cells_added", "purity_ceiling", "active")
        for key, a in res.items():
            log.info("  %-16s %.4f+/-%.4f  %-12d %.4f          %d",
                     key, a["ap"]["mean"], a["ap"]["std"],
                     int(a["cells_added"]["mean"]), a["purity_ceiling"]["mean"],
                     int(a["active_cells"]["mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
