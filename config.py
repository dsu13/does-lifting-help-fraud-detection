"""Central configuration: paths, seeds, candidate cells, splits and IEEE-CIS settings.

Paths are relative to this file, so the scripts run from any directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# ---- paths ----
ROOT: Path = Path(__file__).resolve().parent
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
# result JSONs behind the paper's tables; tracked in git, unlike data/
RESULTS: Path = ROOT / "results"

IEEE_CIS_RAW: Path = DATA_RAW / "ieee_cis"
GAD_RAW: Path = DATA_RAW / "gad"                # Amazon and YelpChi
ELLIPTIC_RAW: Path = DATA_RAW / "elliptic"
DGRAPH_RAW: Path = DATA_RAW / "dgraph"
ELLIPTICPP_RAW: Path = DATA_RAW / "ellipticpp"
SFFSD_RAW: Path = DATA_RAW / "s_ffsd"
ETHEREUM_RAW: Path = DATA_RAW / "ethereum"
# raw source -> (folder, files its builder reads); README.md (Data) says where each comes from
RAW_FILES: Dict[str, Tuple[Path, Tuple[str, ...]]] = {
    "ieee_cis": (IEEE_CIS_RAW, ("train_transaction.csv", "train_identity.csv")),
    "gad": (GAD_RAW, ("Amazon.mat", "YelpChi.mat")),
    "elliptic": (ELLIPTIC_RAW, ("elliptic_txs_features.csv", "elliptic_txs_classes.csv",
                                "elliptic_txs_edgelist.csv")),
    "ellipticpp": (ELLIPTICPP_RAW, ("txs_features.csv", "txs_classes.csv", "txs_edgelist.csv")),
    "dgraph": (DGRAPH_RAW, ("dgraphfin.npz",)),
    "s_ffsd": (SFFSD_RAW, ("S-FFSD.csv",)),
    "ethereum": (ETHEREUM_RAW, ("MulDiGraph.pkl",)),
}


# ---- reproducibility ----
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)         # the 5 saved split seeds


# ---- candidate cells (sec:benchmark) ----
@dataclass(frozen=True)
class CandidateConfig:
    """Hyperparameters for C(G) generation (src/graph/candidate_cells.py)."""
    knn_k: int = 10                 # feature-kNN k (cells {v} U kNN_k(v))
    max_cell_size: int = 50         # max members per cell
    min_cell_size: int = 2          # no singleton cells
    time_window_seconds: int = 24 * 3600   # IEEE-CIS linking window, 24 h
    # scalability (IEEE-CIS has ~590k nodes)
    knn_max_nodes: int = 150_000    # skip brute-force feature-kNN above this size
    knn_n_jobs: int = -1
    max_total_cells: int = 300_000  # cap on |C(G)|, deterministic subsample beyond it
    subsample_seed: int = 0


# ---- splits ----
@dataclass(frozen=True)
class SplitConfig:
    """Split ratios; only labeled nodes are split."""
    train_ratio: float = 0.40
    val_ratio: float = 0.20               # the remaining 40% is test
    # chronological split by node time (IEEE-CIS; --temporal-split of 10)
    temporal_val_fraction: float = 0.15   # val: the 15% right before test
    temporal_test_fraction: float = 0.20  # test: the last 20% by time


# ---- IEEE-CIS graph construction ----
@dataclass(frozen=True)
class IeeeCisConfig:
    """Shared attributes that define relations and hyperedges, plus group caps.

    Each link column gives a relation (same value, same time window) and one hyperedge
    per window.
    """
    # card_combo is a key built from the card columns, a proxy for the physical card
    link_columns: Tuple[str, ...] = (
        "card_combo", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo",
    )
    build_card_combo_from: Tuple[str, ...] = ("card1", "card2", "card3", "card5", "card6")
    # each shared value is cut into time windows; a window keeps its earliest N events
    max_group_size: int = 200
    # numeric node features
    max_numeric_features: int = 200  # first N well-populated numeric columns
    na_fraction_drop: float = 0.90   # drop columns with more missing than this
    label_column: str = "isFraud"
    time_column: str = "TransactionDT"
    id_column: str = "TransactionID"


@dataclass(frozen=True)
class PipelineConfig:
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ieee_cis: IeeeCisConfig = field(default_factory=IeeeCisConfig)
    log_level: str = "INFO"


CONFIG = PipelineConfig()


def ensure_dirs() -> None:
    """Create the data directories (idempotent)."""
    for d in (DATA_PROCESSED, *(raw_dir for raw_dir, _ in RAW_FILES.values())):
        d.mkdir(parents=True, exist_ok=True)


def processed_dir(dataset: str) -> Path:
    d = DATA_PROCESSED / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "splits").mkdir(parents=True, exist_ok=True)
    return d


def results_dir(dataset: str) -> Path:
    """results/<dataset>/, created if missing."""
    d = RESULTS / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d
