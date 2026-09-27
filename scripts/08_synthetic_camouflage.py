"""Step 08: synthetic D(mu) study (paper: def:dmu, tab:synthetic).

Sweeps feature separation Delta x camouflage strength mu and compares rf (features only),
the lifting with lambda=0 (task loss only) and with lambda>0 (purity regularizer).

Run:  python scripts/08_synthetic_camouflage.py --mus 0 0.5 0.9 --deltas 0.5 1.0 2.0
Writes results/synthetic/synthetic_results.json and prints the grid.
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List, Tuple


import sys as _sys
from pathlib import Path as _Path

# Repo root (config.py, src/) onto the path, so this runs as `python scripts/<name>.py`.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import config
from src.seeding import set_seed
from src.data.splits import stratified_split
from src.baselines import get_model, evaluate
from src.method import MethodConfig, fit_and_eval
from src.baselines.metrics import aggregate_runs
from src.method.synthetic import SyntheticConfig, generate_camouflage_instance

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("synthetic")


def run_cell(delta: float, mu: float, lambdas: List[float], seeds: List[int],
             n: int, k: int, pi1: float, epochs: int,
             rho: float, n_decoys: int, rf_cache: Dict[Tuple[float, int], dict]) -> Dict:
    """One (delta, mu) grid cell: rf plus the method for each lambda, over seeds.

    rf sees only the features, labels and split, which D(mu) draws before mu enters, so
    each (delta, seed) is fitted once and its test metrics are kept in rf_cache for every mu.
    """
    per_lambda: Dict[str, List[dict]] = {str(l): [] for l in lambdas}
    rf_runs: List[dict] = []
    for seed in seeds:
        set_seed(seed)
        graph, candidates = generate_camouflage_instance(
            SyntheticConfig(n=n, k=k, pi1=pi1, delta=delta, mu=mu,
                            rho=rho, n_decoys=n_decoys, seed=seed))
        split = stratified_split(graph.labels, seed,
                                 config.CONFIG.split.train_ratio,
                                 config.CONFIG.split.val_ratio)

        if (delta, seed) not in rf_cache:
            rf = get_model("rf", seed=seed)
            rf.fit(graph, split["train_idx"], split["val_idx"])
            y_test = (graph.labels[split["test_idx"]] == 1).astype(int)
            rf_cache[delta, seed] = evaluate(y_test, rf.predict_proba(graph)[split["test_idx"]])
        rf_runs.append(rf_cache[delta, seed])

        for lam in lambdas:
            cfg = MethodConfig(seed=seed, epochs=epochs, hid=64,
                               warmup_epochs=0, ramp_epochs=10, patience=60,
                               # rings are already in the pool, so this tests selection only
                               refine_mode="none")
            per_lambda[str(lam)].append(fit_and_eval(graph, candidates, split, lam, cfg))

    return {"rf": aggregate_runs(rf_runs), **{l: aggregate_runs(r) for l, r in per_lambda.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic D(mu) sweep: rf vs lambda=0 vs lambda>0")
    ap.add_argument("--deltas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--mus", nargs="+", type=float, default=[0.0, 0.5, 0.9])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 1.0])
    # 8 seeds for the paired lambda test (under 9 min on GPU)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pi1", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--rho", type=float, default=0.7, help="fraud fraction of true rings")
    ap.add_argument("--n-decoys", type=int, default=4, help="feature-kNN decoy/social cells per node")
    args = ap.parse_args()

    results: Dict[str, Dict] = {}
    rf_cache: Dict[Tuple[float, int], dict] = {}
    for delta in args.deltas:
        results[str(delta)] = {}
        for mu in args.mus:
            log.info("=== D(mu): delta=%.2f mu=%.2f rho=%.2f decoys=%d ===",
                     delta, mu, args.rho, args.n_decoys)
            results[str(delta)][str(mu)] = run_cell(
                delta, mu, args.lambdas, args.seeds, args.n, args.k, args.pi1,
                args.epochs, args.rho, args.n_decoys, rf_cache)

    results["_config"] = {"lambdas": args.lambdas,
                          "seeds": args.seeds, "n": args.n, "k": args.k, "pi1": args.pi1,
                          "rho": args.rho, "n_decoys": args.n_decoys, "epochs": args.epochs}
    out = config.results_dir("synthetic") / "synthetic_results.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out)

    # ---- print the grid ----
    l0, l1 = str(args.lambdas[0]), str(args.lambdas[-1])
    for delta in args.deltas:
        log.info("--- delta=%.2f (feature strength) ---", delta)
        log.info("  %-6s %-14s %-14s %-14s %-10s %-18s %-14s",
                 "mu", "rf_AP", f"AP(l={l0})", f"AP(l={l1})", "gain",
                 "purity l0 -> l1", "soft_a l0 -> l1")
        for mu in args.mus:
            c = results[str(delta)][str(mu)]
            gain = c[l1]["ap"]["mean"] - c[l0]["ap"]["mean"]
            log.info("  %-6.2f %.3f+/-%.3f  %.3f+/-%.3f  %.3f+/-%.3f  %+.3f     %.2f -> %.2f       %.2f -> %.2f",
                     mu, c["rf"]["ap"]["mean"], c["rf"]["ap"]["std"],
                     c[l0]["ap"]["mean"], c[l0]["ap"]["std"],
                     c[l1]["ap"]["mean"], c[l1]["ap"]["std"], gain,
                     c[l0]["sel_fraud_purity"]["mean"], c[l1]["sel_fraud_purity"]["mean"],
                     c[l0]["mean_soft_a"]["mean"], c[l1]["mean_soft_a"]["mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
