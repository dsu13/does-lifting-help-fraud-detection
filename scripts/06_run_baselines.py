"""Step 06: train, select on val and test the baselines on every seed split.

Default set: rf, mlp, gcn, bwgnn and fixed_hg (on the candidate pool C(G)).

Run:  python scripts/06_run_baselines.py [--datasets amazon] [--baselines mlp gcn]
Writes results/<dataset>/baselines.json, merged with the entries already there.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys as _sys
from pathlib import Path as _Path

# Repo root (config.py, src/) onto the path, so this runs as `python scripts/<name>.py`.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.seeding import set_seed
from src.graph.schema import MultiRelationGraph, CandidateCells
from src.baselines import REGISTRY, get_model, evaluate, available
from src.baselines.metrics import aggregate, METRIC_NAMES
from src.results_io import merge_json

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("baselines")

DEFAULT_BASELINES = ["rf", "mlp", "gcn", "bwgnn", "fixed_hg"]


def _load_split(ddir: Path, seed: int) -> Dict[str, np.ndarray]:
    return dict(np.load(ddir / "splits" / f"seed{seed}.npz"))


def run_dataset(dataset: str, baselines: List[str], seeds) -> Dict:
    ddir = config.DATA_PROCESSED / dataset
    if not (ddir / "graph_meta.json").exists():
        log.warning("skip %s: no processed graph (run steps 02/03/04 first)", dataset)
        return {}
    graph = MultiRelationGraph.load(ddir)
    y = (graph.labels == 1).astype(int)
    # fixed_hg uses the same candidate pool C(G) as the lifting model
    cands = CandidateCells.load(ddir) if "fixed_hg" in baselines else None

    results: Dict[str, Dict] = {}
    for bname in baselines:
        if bname not in REGISTRY:
            log.warning("unknown baseline '%s' (available: %s)", bname, available())
            continue
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            split = _load_split(ddir, seed)
            try:
                extra = {"candidates": cands} if bname == "fixed_hg" else {}
                model = get_model(bname, seed=seed, **extra)
                model.fit(graph, split["train_idx"], split["val_idx"])
                proba = model.predict_proba(graph)
                test_idx = split["test_idx"]
                m = evaluate(y[test_idx], proba[test_idx])
                per_seed.append(m)
                log.info("[%s/%s seed %d] AP=%.4f ROC=%.4f bestF1=%.4f",
                         dataset, bname, seed, m["ap"], m["roc_auc"], m["best_f1"])
            except Exception as e:  # noqa: BLE001
                log.exception("[%s/%s seed %d] failed: %s", dataset, bname, seed, e)
                # keep the slot so per-seed values stay in seed order (NaN = failed)
                per_seed.append({k: float("nan") for k in METRIC_NAMES})
                continue
        if any(r["ap"] == r["ap"] for r in per_seed):
            results[bname] = aggregate(per_seed)
            results[bname]["seeds"] = list(seeds)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the baselines")
    ap.add_argument("--datasets", nargs="+", default=["amazon", "yelpchi", "ieee_cis"])
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES,
                    help="baseline names (default: the five of the paper)")
    args = ap.parse_args()

    for ds in args.datasets:
        ds = ds.lower()
        log.info("=== baselines on %s: %s ===", ds, args.baselines)
        res = run_dataset(ds, args.baselines, config.SEEDS)
        if not res:
            continue
        out = config.results_dir(ds) / "baselines.json"
        # merge (under a lock) so a partial rerun keeps the other baselines
        merged, replaced = merge_json(out, res)
        log.info("wrote %s (updated %s; kept %s)", out, sorted(res),
                 sorted(set(merged) - set(res)) or "nothing else")
        if replaced:
            log.info("  replaced previous results for: %s", replaced)
        res = merged
        # leaderboard by AP
        board = sorted(res.items(), key=lambda kv: -kv[1]["ap"]["mean"])
        log.info("--- %s leaderboard (AP mean +/- std) ---", ds)
        for name, agg in board:
            log.info("  %-10s AP=%.4f+/-%.4f  ROC=%.4f  bestF1=%.4f",
                     name, agg["ap"]["mean"], agg["ap"]["std"],
                     agg["roc_auc"]["mean"], agg["best_f1"]["mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
