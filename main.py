"""Reproduce the paper in one pass: every step of scripts/, in order, with the paper's arguments.

This file is the single list of the commands behind the stored results. Each step runs as
its own process (python scripts/<step>.py ...) and the first failure stops the run. The
first step downloads the datasets that have a public link and stops the run, with the
steps to follow, while any other is missing from data/raw/ (README.md, Data).

Run:  python main.py                          # every stage
      python main.py --stages temporal sweep  # some stages, in pipeline order
      python main.py --list                   # print the commands without running them
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import config

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("main")

HERE = Path(__file__).resolve().parent
WINDOW = "0.02"                          # omega as a share of the node-time span
SWEEP_WINDOWS = ("0.01", "0.05")
# seeds per temporal dataset (paper, Reproducibility); a cell pool uses its base dataset's
N_SEEDS = {"elliptic": 8, "ellipticpp": 8, "ieee_cis": 3, "dgraph": 3, "s_ffsd": 3, "ethereum": 16}


def _crossover(datasets: List[str], *args: str, window: str = WINDOW) -> List[str]:
    """A 10_temporal_crossover.py call; the datasets of one call share their seed count."""
    counts = {N_SEEDS[ds.replace("_xstep", "").replace("_samestep", "")] for ds in datasets}
    assert len(counts) == 1, datasets
    return (["10_temporal_crossover.py", "--datasets", *datasets, "--window-frac", window,
             "--seeds", *map(str, range(counts.pop()))] + list(args))


def stages() -> Dict[str, List[List[str]]]:
    """Stage name -> commands (script file plus arguments), in pipeline order."""
    full_b_d = ("--levels", "1.0", "--arms", "B_temporal_cols", "D_temporal_cells")
    cones_gate = ("--levels", "1.0", "--arms", "D_temporal_gate", "B_cone_cols", "D_cone_cells",
                  "--out-name", "temporal_variants.json")
    return {
        # raw data, graphs, candidate pools, splits and dataset statistics
        "data": [
            ["00_download_data.py"],
            ["01_setup_check.py"],
            ["02_build_ieee_cis.py"],
            ["02b_build_elliptic.py"],
            ["02b_build_elliptic.py", "--extra-cells", "xstep"],
            ["02b_build_elliptic.py", "--extra-cells", "samestep"],
            ["02c_build_dgraph.py", "--max-nodes", "150000"],
            ["02d_build_ellipticpp.py"],
            ["02d_build_ellipticpp.py", "--extra-cells", "xstep"],
            ["02d_build_ellipticpp.py", "--extra-cells", "samestep"],
            ["02e_build_s_ffsd.py"],
            ["02f_build_ethereum.py", "--max-nodes", "150000"],
            ["03_load_gad_datasets.py"],
            ["04_make_splits.py"],
            ["05_compute_stats.py", "amazon", "yelpchi", "ieee_cis", "elliptic", "dgraph", "ethereum"],
        ],
        # static benchmark and D(mu); the ablation's label run is the lambda=0 selection row
        "static": [
            ["06_run_baselines.py"],
            ["07_train_method.py"],
            ["07_train_method.py", "--refine-modes", "label", "--lambdas", "0.5", "1", "2",
             "--out-name", "method_results.json"],
            ["08_synthetic_camouflage.py"],
        ],
        # main crossovers, then the Elliptic cell-pool redesign at full labels
        "temporal": [
            _crossover(["elliptic"], "--levels", "1.0", "0.1", "0.02"),
            _crossover(["ellipticpp"], "--levels", "1.0", "--with-temporal-gnn"),
            _crossover(["ieee_cis"], "--levels", "1.0", "0.1", "0.02"),
            _crossover(["dgraph"], "--levels", "1.0"),
            _crossover(["s_ffsd"], "--levels", "1.0", "--with-temporal-gnn"),
            _crossover(["ethereum"], "--levels", "1.0", "--with-temporal-gnn"),
            _crossover(["elliptic_xstep", "ellipticpp_xstep"], *full_b_d),
            _crossover(["elliptic_samestep", "ellipticpp_samestep"],
                       "--levels", "1.0", "--arms", "D_temporal_cells"),
        ],
        # D - B at the other windows; on the Elliptic families every window gives the same cells
        "sweep": [
            _crossover([ds], *full_b_d, "--out-name", f"temporal_crossover_w{w}.json", window=w)
            for ds in ("ieee_cis", "dgraph", "s_ffsd", "ethereum") for w in SWEEP_WINDOWS
        ],
        # temporal variants and the uncut-pool controls, paired with the main runs
        "variants": [
            _crossover(["s_ffsd"], *cones_gate),
            _crossover(["dgraph"], *cones_gate),
            _crossover(["ethereum"], *cones_gate),
            _crossover(["ethereum"], "--levels", "1.0", "--arms", "B_future_cone_cols",
                       "D_future_cone_cells", "--out-name", "temporal_variants_future.json"),
            _crossover(["ethereum"], "--levels", "1.0", "--arms", "D_static_cells",
                       "--out-name", "temporal_variants_static.json"),
            _crossover(["ethereum"], "--levels", "1.0", "--arms", "B_static_cols",
                       "--out-name", "temporal_variants_static_cols.json"),
            _crossover(["dgraph", "s_ffsd"], "--levels", "1.0", "--arms", "B_static_cols",
                       "--out-name", "temporal_variants_static_cols.json"),
        ],
        # LaTeX tables and statistics macros from results/, written to tables/
        "tables": [["09_make_tables.py"]],
    }


def main() -> int:
    plan = stages()
    ap = argparse.ArgumentParser(description="Run every step that reproduces the paper, in order")
    ap.add_argument("--stages", nargs="+", choices=list(plan), default=list(plan),
                    help="stages to run (default: all); they run in pipeline order")
    ap.add_argument("--list", action="store_true", help="print the commands and exit")
    args = ap.parse_args()
    steps = [cmd for name in plan if name in args.stages for cmd in plan[name]]

    if args.list:
        for cmd in steps:
            print("python scripts/" + " ".join(cmd))
        return 0

    t0 = time.time()
    for i, cmd in enumerate(steps, 1):
        log.info("=== [%d/%d] python scripts/%s", i, len(steps), " ".join(cmd))
        t = time.time()
        rc = subprocess.run([sys.executable, str(HERE / "scripts" / cmd[0]), *cmd[1:]],
                            cwd=HERE).returncode
        if rc != 0:
            log.error("step failed (exit %d); fix it and rerun the remaining stages with --stages", rc)
            return rc
        log.info("    done in %.1f min", (time.time() - t) / 60)
    log.info("done in %.1f h (%d steps); results in %s, tables in %s",
             (time.time() - t0) / 3600, len(steps), config.RESULTS, HERE / "tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
