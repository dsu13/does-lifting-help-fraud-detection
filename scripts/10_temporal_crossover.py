"""Step 10: temporal crossover over a label-scarcity sweep (paper: sec:arms).

Same backbone and class reweighting in every arm:
  A  features only                      B  A + columns from D's time-split cells
  C  A + label-carved cells (lam_c)     D  A + label-free temporal-coherence cells
  E  (optional) GCN on B's augmented graph
The decisive test is a significant D > B*, B* the stronger of B and the uncut-pool columns
(B_static_cols, a variant) where both ran; the cones face their own columns. Tests are
paired over seeds in 09_make_tables.py. Needs time.npz.
If no cell of D spans two times, B's columns carry no time spans and _config has
temporal_arms_degenerate = true. If the window also splits no cell, D's cells are the static
pool's under one 'temporal' family: the static lifting on a single-family pool (Elliptic,
Elliptic++), not on the two-family *_samestep pools.

Temporal variants (--arms), paired in 09 with A, B and D of the main run:
  D_temporal_gate  D's cells, with their temporal shape added to the gate scorer's input
  D_cone_cells     causal cones: a node receives only from strictly earlier co-members
  B_cone_cols      A + columns from the same cones (the cones' decisive test)
  D_future_cone_cells, B_future_cone_cols
                   the mirror image: co-members up to omega later (a probe, not causal)
  D_static_cells   arm D on the unsplit static pool: does cutting the cells by time help?
  B_static_cols    A + the same five columns computed from the unsplit static cells

Run:  python scripts/10_temporal_crossover.py --datasets <name> --window-frac 0.02 --seeds ...
      (the paper's runs: stages temporal, sweep and variants of main.py; --list prints them)
Writes results/<dataset>/<--out-name>, temporal_crossover.json by default.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
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
from src.data.splits import stratified_split, temporal_split
from src.method import MethodConfig, fit_and_eval
from src.baselines import get_model, evaluate
from src.data.temporal import (
    cell_temporal_features, temporal_coherence_cells, subsample_train_positives,
    augment_features, cell_temporal_descriptor, cone_cells, cone_node_features,
)

logging.basicConfig(level=config.CONFIG.log_level,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("crossover")

ARMS = ["A_features", "B_temporal_cols", "C_label_cells", "D_temporal_cells"]
VARIANT_ARMS = ["D_temporal_gate", "B_cone_cols", "D_cone_cells",
                "B_future_cone_cols", "D_future_cone_cells", "D_static_cells", "B_static_cols"]
# arms built on D's temporal-coherence cells, and those built on B's columns of them
_USES_D_CELLS = {"B_temporal_cols", "D_temporal_cells", "D_temporal_gate", "E_temporal_gnn"}
_USES_B_COLS = {"B_temporal_cols", "E_temporal_gnn"}


def _load_or_make_split(ddir, graph, seed: int, times: np.ndarray, temporal: bool):
    """Return (split, protocol) for this seed.

    temporal=True gives a time-ordered split. Otherwise the saved stratified split, or one
    drawn the same way when it is missing."""
    sc = config.CONFIG.split
    if temporal:
        return (temporal_split(graph.labels, times, seed,
                               sc.temporal_val_fraction, sc.temporal_test_fraction),
                "drawn:temporal")
    p = ddir / "splits" / f"seed{seed}.npz"
    if p.exists():
        return dict(np.load(p)), "saved:stratified"
    return stratified_split(graph.labels, seed, sc.train_ratio, sc.val_ratio), "drawn:stratified"


def _agg(runs: List[dict]) -> Dict[str, float]:
    """Aggregate per-seed metrics: AP with per-seed values for the paired test,
    plus AUC and gate diagnostics when present."""
    ap_vals = [float(r["ap"]) for r in runs]
    a = np.array([v for v in ap_vals if v == v], dtype=float)      # finite seeds only
    out = {"mean": float(a.mean()) if a.size else float("nan"),
           "std": float(a.std(ddof=1)) if a.size > 1 else (0.0 if a.size else float("nan")),
           "std_ddof": 1, "n": int(a.size), "n_failed": int(len(ap_vals) - a.size),
           # per-seed in seed order (None = failed), for the paired test
           "aps": [round(v, 6) if v == v else None for v in ap_vals]}
    auc_vals = [float(r.get("roc_auc", float("nan"))) for r in runs]
    au = np.array([v for v in auc_vals if v == v], dtype=float)
    if au.size:
        out["auc_mean"] = float(au.mean())
        out["auc_std"] = float(au.std(ddof=1)) if au.size > 1 else 0.0
        out["aucs"] = [round(v, 6) if v == v else None for v in auc_vals]
    # per-seed diagnostics: selected epoch (every arm but E) and gate statistics (lifting
    # arms only)
    for key in ("active_cells", "mean_soft_a", "frac_fraud_covered", "selected_epoch"):
        if any(key in r for r in runs):
            out[f"{key}_per_seed"] = [None if r.get(key) is None or r.get(key) != r.get(key)
                                      else round(float(r[key]), 6) for r in runs]
    return out


def run(dataset: str, seeds: List[int], levels: List[float], epochs: int, times: np.ndarray,
        window: float, lam_c: float, time_ordered: bool, arms: List[str],
        diag: Dict) -> Dict:
    """Train every arm per label fraction and seed; run diagnostics go into `diag`."""
    ddir = config.DATA_PROCESSED / dataset
    graph = MultiRelationGraph.load(ddir)
    static_cells = CandidateCells.load(ddir)

    temporal_cells, graph_B = None, None
    if _USES_D_CELLS & set(arms):
        log.info("building temporal-coherence cells (arm D) ...")
        temporal_cells = temporal_coherence_cells(static_cells, times, window=window)
        degenerate = temporal_cells.frac_spanning == 0.0
        if degenerate:
            log.warning("[%s] DEGENERATE at window=%g: no cell of arm D spans more than one "
                        "time, so B's columns carry no time spans (%.0f%% of cells split; with "
                        "none split, D is the static lifting on a single-family pool)",
                        dataset, window, 100 * temporal_cells.frac_split)
        diag.update({"frac_cells_split": float(temporal_cells.frac_split),
                     "frac_cells_spanning_time": float(temporal_cells.frac_spanning),
                     "temporal_arms_degenerate": bool(degenerate),
                     "n_temporal_cells": int(temporal_cells.num_cells)})
    if _USES_B_COLS & set(arms):
        # B's columns come from D's own cells and window, so both get the same temporal info
        temporal_feats = cell_temporal_features(temporal_cells, times)
        graph_B = augment_features(graph, temporal_feats)
        diag["b_column_std"] = [round(float(s), 6) for s in temporal_feats.std(0)]
    # variants: D's cells with a time-aware gate; past and future cones, each with its own
    # columns; columns of the unsplit static cells
    gate_cells = None
    cones: Dict[str, CandidateCells] = {}
    cone_graphs: Dict[str, MultiRelationGraph] = {}
    graph_static = None
    if "B_static_cols" in arms:
        static_feats = cell_temporal_features(static_cells, times)
        graph_static = augment_features(graph, static_feats)
        diag["static_column_std"] = [round(float(s), 6) for s in static_feats.std(0)]
    if "D_temporal_gate" in arms:
        gate_cells = dataclasses.replace(
            temporal_cells, cell_features=cell_temporal_descriptor(temporal_cells, times, window))
    for direction, tag in (("past", ""), ("future", "future_")):
        if {f"B_{tag}cone_cols", f"D_{tag}cone_cells"} & set(arms):
            cc = cone_cells(static_cells, times, window=window, direction=direction)
            feats = cone_node_features(cc, times)
            cones[direction], cone_graphs[direction] = cc, augment_features(graph, feats)
            diag.update({f"n_{tag}cone_cells": int(cc.num_cells),
                         f"n_{tag}cones_before_cap": cc.n_cones_total,
                         f"frac_nodes_with_{tag}cone": cc.frac_nodes_with_cone,
                         f"{tag}cone_column_std": [round(float(s), 6) for s in feats.std(0)]})

    def cfg_for(arm: str, seed: int) -> MethodConfig:
        base = dict(seed=seed, epochs=epochs)
        if arm.startswith(("A_", "B_")):
            return MethodConfig(use_lifting=False, **base)
        if arm.startswith("C_"):
            return MethodConfig(use_lifting=True, refine_mode="label", **base)
        return MethodConfig(use_lifting=True, refine_mode="none", **base)  # D

    protocols: Dict[str, str] = {}
    y_all = (graph.labels == 1).astype(int)
    results: Dict[str, Dict[str, Dict]] = {a: {} for a in arms}
    for level in levels:
        per_arm: Dict[str, List[dict]] = {a: [] for a in arms}
        for seed in seeds:
            set_seed(seed)
            split, protocol = _load_or_make_split(ddir, graph, seed, times, time_ordered)
            protocols[str(seed)] = protocol
            split["train_idx"] = subsample_train_positives(
                graph.labels, split["train_idx"], keep_frac=level, seed=seed)
            for arm in arms:
                try:
                    if arm == "E_temporal_gnn":
                        # GCN on B's augmented graph (time-aware, not a TGN)
                        gcn = get_model("gcn", seed=seed)
                        gcn.fit(graph_B, split["train_idx"], split["val_idx"])
                        proba = gcn.predict_proba(graph_B)
                        m = evaluate(y_all[split["test_idx"]], proba[split["test_idx"]])
                    else:
                        g = {"B_temporal_cols": graph_B, "B_cone_cols": cone_graphs.get("past"),
                             "B_future_cone_cols": cone_graphs.get("future"),
                             "B_static_cols": graph_static}.get(arm, graph)
                        cands = {"D_temporal_cells": temporal_cells, "D_temporal_gate": gate_cells,
                                 "D_cone_cells": cones.get("past"),
                                 "D_future_cone_cells": cones.get("future"),
                                 "D_static_cells": static_cells}.get(arm, static_cells)
                        lam = lam_c if arm == "C_label_cells" else 0.0
                        m = fit_and_eval(g, cands, split, lam, cfg_for(arm, seed))
                    per_arm[arm].append(m)
                    log.info("[%s level=%.3g seed=%d] AP=%.4f AUC=%.4f",
                             arm, level, seed, m["ap"], m.get("roc_auc", float("nan")))
                except Exception as e:  # noqa: BLE001
                    log.exception("[%s level=%.3g seed=%d] failed: %s", arm, level, seed, e)
                    # keep the slot: 09 pairs arms by seed
                    per_arm[arm].append({"ap": float("nan"), "roc_auc": float("nan")})
        for arm in arms:
            results[arm][f"{level:g}"] = _agg(per_arm[arm])
            results[arm][f"{level:g}"]["seeds"] = list(seeds)
    diag["split_protocol"] = protocols
    return results


_GAPS = [("D_temporal_cells", "B_temporal_cols", "D-B (temporal lifting vs its columns)"),
         ("D_temporal_cells", "A_features", "D-A (temporal lifting vs features)"),
         ("C_label_cells", "A_features", "C-A (label-derived cells vs features)"),
         ("D_cone_cells", "B_cone_cols", "cones-B (causal cones vs their columns)"),
         ("D_future_cone_cells", "B_future_cone_cols", "future cones-B (vs their columns)")]


def _verdict(results: Dict) -> None:
    levels = sorted({lv for arm in results.values() for lv in arm}, key=float, reverse=True)
    present = [a for a in ARMS + VARIANT_ARMS + ["E_temporal_gnn"] if a in results]
    log.info("=== TEMPORAL CROSSOVER (AP mean+/-std) ===")
    log.info("  %-18s %s", "arm \\ keep-frac", "  ".join(f"{lv:>14}" for lv in levels))
    for arm in present:
        cells = "  ".join(f"{results[arm][lv]['mean']:.3f}+/-{results[arm][lv]['std']:.3f}"
                          for lv in levels)
        log.info("  %-18s %s", arm, cells)
    if any("auc_mean" in results[a][lv] for a in present for lv in levels):
        log.info("--- ROC-AUC (diagnostic: high AUC + flat AP => feature-dominated, not no-signal) ---")
        for arm in present:
            cells = "  ".join(f"{results[arm][lv].get('auc_mean', float('nan')):.3f}" for lv in levels)
            log.info("  %-18s %s", arm, cells)
    # gaps between arms
    def gap(a1, a2, lv):
        return results[a1][lv]["mean"] - results[a2][lv]["mean"]
    log.info("--- gaps as the labeled-positive fraction DECREASES ---")
    for a1, a2, text in _GAPS:
        if a1 in results and a2 in results:
            log.info("  %-42s %s", text + ":",
                     "  ".join(f"{lv}:{gap(a1, a2, lv):+.3f}" for lv in levels))
    log.info("Decisive tests: D > B*, the stronger of B and the uncut-pool columns "
             "(B_static_cols); cones > their own columns (paired over seeds in "
             "09_make_tables.py).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Four-arm temporal crossover")
    ap.add_argument("--datasets", nargs="+", default=["ieee_cis"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--levels", nargs="+", type=float, default=[1.0, 0.1, 0.02])
    ap.add_argument("--epochs", type=int, default=300,
                    help="per arm; arms A-D select checkpoints from epoch 120 (after the "
                         "lambda warm-up and ramp), so 300 leaves up to 37 candidates (every "
                         "5th epoch from 120, plus the last), fewer if patience stops early")
    ap.add_argument("--window-frac", type=float, default=0.02,
                    help="omega = FRAC x (node-time span), rounded up to whole steps on "
                         "integer time; the paper uses 0.02, and 0.01 and 0.05 in the "
                         "window sweep")
    ap.add_argument("--out-name", default="temporal_crossover.json",
                    help="result file name (use distinct names for a window sweep)")
    ap.add_argument("--lam-c", type=float, default=1.0, help="lambda for arm C")
    ap.add_argument("--temporal-split", action="store_true",
                    help="split strictly by time a dataset that is split by label by "
                         "default (IEEE-CIS is always split chronologically)")
    ap.add_argument("--arms", nargs="+", choices=ARMS + VARIANT_ARMS, default=None,
                    help="arms to run (default: A B C D); the variants are paired in 09 "
                         "with the main run, so use its seeds and window")
    ap.add_argument("--with-temporal-gnn", action="store_true",
                    help="add arm E, a pairwise temporal-GNN baseline (GCN on the "
                         "temporal-feature-augmented graph), alongside arms A-D")
    args = ap.parse_args()

    for ds in args.datasets:
        ds = ds.lower()
        log.info("=== temporal crossover on %s (heavy: IEEE-CIS is 590k nodes) ===", ds)
        tpath = config.DATA_PROCESSED / ds / "time.npz"
        if not tpath.exists():
            log.error("%s has no time.npz -- the temporal crossover needs timestamps. "
                      "Skipping.", ds)
            continue
        tt = np.load(tpath)["transaction_dt"]
        window = float(args.window_frac * (np.nanmax(tt) - np.nanmin(tt)))
        rounded = bool(np.all(tt[np.isfinite(tt)] == np.round(tt[np.isfinite(tt)])))
        if rounded:
            # discrete time steps: a sub-step window would keep same-step members only
            window = float(max(1.0, np.ceil(window)))
        log.info("[%s] window = %g x span -> %g%s", ds, args.window_frac, window,
                 " (rounded up to whole steps)" if rounded else "")
        time_ordered = args.temporal_split or ds == "ieee_cis"
        arms = list(args.arms or ARMS) + (["E_temporal_gnn"] if args.with_temporal_gnn else [])
        diag: Dict = {}
        res = run(ds, args.seeds, args.levels, args.epochs, tt, window, args.lam_c,
                  time_ordered, arms, diag)
        out = config.results_dir(ds) / args.out_name
        _verdict(res)
        # run config stored with the results (09_make_tables.py skips "_" keys)
        res["_config"] = {"dataset": ds, "seeds": list(args.seeds),
                          "levels": list(args.levels), "epochs": args.epochs,
                          "window": window, "window_frac": args.window_frac,
                          "window_rounded_to_steps": rounded,
                          "lam_c": args.lam_c, **diag,
                          "temporal_split": time_ordered,
                          "with_temporal_gnn": bool(args.with_temporal_gnn)}
        out.write_text(json.dumps(res, indent=2))
        log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
