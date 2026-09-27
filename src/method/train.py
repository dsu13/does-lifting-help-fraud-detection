"""Train the camouflage-aware lifting on one split and report test metrics.

lambda = 0 is the task-loss-only lifting; lambda > 0 adds the fraud-conditional regularizer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from ..graph.schema import UNLABELED
from ..baselines.metrics import evaluate
from ..torch_utils import scipy_to_torch_sparse
from .model import CamoLiftNet
from .objectives import (
    build_struct_features, fraud_and_usable_mass, purity_from_mass, focal_bce,
    coverage_coupled_regularizer, density_penalty, selected_cell_fraud_purity, fraud_coverage,
)
from .refinement import refine_candidates, per_node_best_purity


def _lambda_schedule(epoch: int, lam: float, warmup: int, ramp: int) -> float:
    """0 during warm-up, then a linear ramp to lam over `ramp` epochs."""
    if lam <= 0 or epoch < warmup:
        return 0.0
    if epoch >= warmup + ramp:
        return lam
    return lam * (epoch - warmup) / max(ramp, 1)

logger = logging.getLogger(__name__)


@dataclass
class MethodConfig:
    hid: int = 128                  # wide enough that lambda=0 reaches the MLP floor
    n_layers: int = 2
    dropout: float = 0.4
    lr: float = 5e-3
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 50
    eval_every: int = 5
    focal_gamma: float = 2.0
    seed: int = 0
    # --- regularizer schedule and coverage ---
    warmup_epochs: int = 40         # lambda=0 first, to learn a useful lifting
    ramp_epochs: int = 80           # then ramp lambda linearly to target
    # checkpoints are selected from epoch warmup+ramp on, in every run
    cov_tau: float = 1.0            # target coverage per fraud node (relu(tau - tot_v))
    cov_gamma: float = 0.2          # coverage weight, small so it does not swamp purity
    target_density: float = 0.10    # budget: target active fraction of cells
    beta_density: float = 1e-2      # weight on the density (budget) penalty
    # --- arms and cell generation ---
    use_lifting: bool = True        # False: feature-only MLP on the same backbone (arms A/B)
    refine_mode: str = "label"      # cell generation: 'label' | 'unsup' | 'random' | 'none'
    refine_sim_percentile: float = 50.0   # keep co-members above this sim percentile
    refine_max_new: int = 50_000    # cap on refined cells added


def _select_from(cfg: "MethodConfig") -> int:
    """First epoch whose checkpoint may be selected (end of the lambda warm-up and ramp).

    Earlier checkpoints would have a weaker lambda. Same for every lambda and for arms A/B,
    so all runs share one early-stopping rule."""
    return cfg.warmup_epochs + cfg.ramp_epochs


def _fit_feature_only(feats, graph, split, lam, cfg, device) -> dict:
    """Arms A/B: feature-only MLP on the CamoLiftNet backbone (no cells)."""
    torch.manual_seed(cfg.seed)
    train_idx = split["train_idx"]; val_idx = split["val_idx"]; test_idx = split["test_idx"]
    X = torch.tensor(feats, dtype=torch.float32, device=device)
    y_np = (graph.labels == 1).astype(np.float32)
    y = torch.tensor(y_np, device=device)
    model = CamoLiftNet(X.shape[1], 1, cfg.hid, cfg.n_layers, cfg.dropout, use_lifting=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_pos = float(y_np[train_idx].sum()); n_neg = float((y_np[train_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    tr = torch.tensor(train_idx, dtype=torch.long, device=device)
    y_val = y_np[val_idx].astype(int)
    select_from = _select_from(cfg)
    best_ap, best_state, wait, best_epoch = float("nan"), None, 0, -1
    for epoch in range(cfg.epochs):
        model.train(); opt.zero_grad()
        logits, _ = model(X)
        focal_bce(logits[tr], y[tr], gamma=cfg.focal_gamma, pos_weight=pos_weight).backward()
        opt.step()
        if epoch < select_from and epoch != cfg.epochs - 1:
            continue                         # before the selection window: train only
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                proba = torch.sigmoid(model(X)[0]).cpu().numpy()
            ap = evaluate(y_val, proba[val_idx])["ap"]
            # '>=': on val-AP ties prefer the later checkpoint
            if best_state is None or ap >= best_ap:
                best_ap, wait, best_epoch = ap, 0, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += cfg.eval_every
                if wait >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(X)[0]).cpu().numpy()
    metrics = evaluate((graph.labels[test_idx] == 1).astype(int), proba[test_idx])
    metrics.update({"val_ap_best": best_ap, "lambda": lam,
                    "selected_epoch": best_epoch, "select_from": select_from})
    logger.info("[feature-only seed=%d] test AP=%.4f  (n_feat=%d)", cfg.seed, metrics["ap"], X.shape[1])
    return metrics


def fit_and_eval(graph, candidates, split, lam: float, cfg: MethodConfig) -> dict:
    """Train one (split, lambda, seed) run; return test metrics plus cell diagnostics."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = graph.num_nodes
    train_idx = split["train_idx"]; val_idx = split["val_idx"]; test_idx = split["test_idx"]

    # ---- standardize with train statistics ----
    feats = graph.features.astype(np.float64)
    mu = feats[train_idx].mean(axis=0)
    sd = feats[train_idx].std(axis=0); sd[sd == 0] = 1.0
    feats = ((feats - mu) / sd).astype(np.float32)

    # ---- feature-only arms (A/B): same backbone, no cells ----
    if not cfg.use_lifting:
        return _fit_feature_only(feats, graph, split, lam, cfg, device)

    # ---- candidate generation ----
    cells_added = 0
    if cfg.refine_mode != "none":
        if candidates.apex is not None or candidates.cell_features is not None:
            raise ValueError("cones and cell descriptors need refine_mode='none'")
        candidates, cells_added = refine_candidates(
            candidates, feats, graph.labels, train_idx, mode=cfg.refine_mode,
            sim_percentile=cfg.refine_sim_percentile, max_new=cfg.refine_max_new,
            seed=cfg.seed)
    B_cand = candidates.incidence.tocsr()
    # who receives each cell's message: every member, or only the apex of a cone
    if candidates.apex is None:
        B_recv = B_cand
    else:
        n_cells = candidates.num_cells
        B_recv = sp.csr_matrix(
            (np.ones(n_cells, dtype=np.float32), (candidates.apex, np.arange(n_cells))),
            shape=B_cand.shape)

    # ---- static tensors ----
    X = torch.tensor(feats, dtype=torch.float32, device=device)
    B = scipy_to_torch_sparse(B_recv, device)                 # (n, m) cell -> node
    B_t = scipy_to_torch_sparse(B_cand.T, device)             # (m, n) node -> cell
    cell_size = torch.tensor(candidates.cell_sizes(), dtype=torch.float32, device=device).clamp_(min=1.0)
    struct = torch.tensor(build_struct_features(candidates), dtype=torch.float32, device=device)

    y_np = (graph.labels == 1).astype(np.float32)
    y = torch.tensor(y_np, device=device)

    # regularizer masses from train labels only (no leakage)
    q_tr = np.zeros(n, dtype=np.float32); q_tr[train_idx] = y_np[train_idx]
    m_tr = np.zeros(n, dtype=np.float32); m_tr[train_idx] = 1.0
    f_tr, ell_tr = fraud_and_usable_mass(B_cand, q_tr, m_tr)
    f_tr = torch.tensor(f_tr, device=device); pi_tr = purity_from_mass(f_tr, torch.tensor(ell_tr, device=device))
    q_node_t = torch.tensor(q_tr, device=device)   # train fraud indicator per node

    # best available purity per node (regularizer target) and the purity ceiling
    best_pi_np = per_node_best_purity(B_recv, pi_tr.cpu().numpy())
    best_pi = torch.tensor(best_pi_np, dtype=torch.float32, device=device)
    fraud_mask = q_tr > 0
    purity_ceiling = float(best_pi_np[fraud_mask].mean()) if fraud_mask.any() else float("nan")
    logger.info("purity ceiling (train fraud, post-refine): %.4f  (+%d refined cells)",
                purity_ceiling, cells_added)

    # diagnostic masses from full labels (evaluation only)
    q_full = y_np.copy()
    m_full = (graph.labels != UNLABELED).astype(np.float32)
    f_full, ell_full = fraud_and_usable_mass(B_cand, q_full, m_full)
    f_full = torch.tensor(f_full, device=device); pi_full = purity_from_mass(f_full, torch.tensor(ell_full, device=device))

    # ---- model / opt ----
    model = CamoLiftNet(X.shape[1], struct.shape[1], cfg.hid, cfg.n_layers, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_pos = float(y_np[train_idx].sum()); n_neg = float((y_np[train_idx] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    tr = torch.tensor(train_idx, dtype=torch.long, device=device)
    y_val = y_np[val_idx].astype(int)

    best_ap, best_state, wait, best_epoch = float("nan"), None, 0, -1
    # same selection window for every run (see _select_from)
    eligible_from = _select_from(cfg)
    if eligible_from >= cfg.epochs:
        logger.warning("[refine=%s lam=%.3g] epochs=%d <= warmup+ramp=%d: only the final "
                       "state can be selected", cfg.refine_mode, lam, cfg.epochs, eligible_from)
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        logits, a = model(X, B, B_t, cell_size, struct)
        loss = focal_bce(logits[tr], y[tr], gamma=cfg.focal_gamma, pos_weight=pos_weight)
        eff_lambda = _lambda_schedule(epoch, lam, cfg.warmup_epochs, cfg.ramp_epochs)
        if eff_lambda > 0:
            loss = loss + eff_lambda * coverage_coupled_regularizer(
                a, pi_tr, B, q_node_t, cfg.cov_tau, cfg.cov_gamma, best_pi=best_pi)
        loss = loss + cfg.beta_density * density_penalty(a, cfg.target_density)
        loss.backward()
        opt.step()

        if epoch < eligible_from and epoch != cfg.epochs - 1:
            continue                         # before the selection window: train only
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                logits, a = model(X, B, B_t, cell_size, struct)
                proba = torch.sigmoid(logits).cpu().numpy()
            ap = evaluate(y_val, proba[val_idx])["ap"]
            # '>=': on val-AP ties prefer the later checkpoint
            if best_state is None or ap >= best_ap:
                best_ap, wait, best_epoch = ap, 0, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += cfg.eval_every
                if wait >= cfg.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final eval ----
    model.eval()
    with torch.no_grad():
        logits, a = model(X, B, B_t, cell_size, struct)
        proba = torch.sigmoid(logits).cpu().numpy()
    metrics = evaluate((graph.labels[test_idx] == 1).astype(int), proba[test_idx])
    metrics["sel_fraud_purity"] = selected_cell_fraud_purity(a, f_full, pi_full)
    metrics["active_cells"] = int((a > 0.5).sum().item())
    metrics["mean_soft_a"] = float(a.mean().item())   # soft mass: gates leak below the 0.5 threshold
    metrics["frac_fraud_covered"] = fraud_coverage(a, B, q_node_t)
    metrics["purity_ceiling"] = purity_ceiling
    metrics["cells_added"] = cells_added
    metrics["val_ap_best"] = best_ap
    metrics["lambda"] = lam
    metrics["selected_epoch"] = best_epoch if best_state is not None else cfg.epochs - 1
    metrics["select_from"] = eligible_from
    metrics["lambda_at_selected"] = _lambda_schedule(
        metrics["selected_epoch"], lam, cfg.warmup_epochs, cfg.ramp_epochs)
    logger.info("[refine=%s lam=%.3g seed=%d] test AP=%.4f  sel_purity=%.4f  cover=%.3f  active=%d/%d",
                cfg.refine_mode, lam, cfg.seed, metrics["ap"], metrics["sel_fraud_purity"],
                metrics["frac_fraud_covered"], metrics["active_cells"], candidates.num_cells)
    return metrics
