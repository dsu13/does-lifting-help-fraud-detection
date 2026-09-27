"""Graph containers: MultiRelationGraph (paper: G = (V, {A_r}, X, y)) and CandidateCells (C(G)).

Both save to .npz files plus a JSON sidecar, so nothing depends on pickle.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# label of unlabeled nodes (paper: y_v = star)
UNLABELED: int = -1


@dataclass
class MultiRelationGraph:
    """Attributed multi-relation graph.

    features: (n, d) float32. labels: (n,) in {0, 1, UNLABELED}, 1 = fraud.
    relations: symmetric n x n adjacency per relation, no self-loops.
    """

    name: str
    features: np.ndarray
    labels: np.ndarray
    relations: Dict[str, sp.csr_matrix]
    meta: Dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def relation_names(self) -> List[str]:
        return list(self.relations.keys())

    def labeled_mask(self) -> np.ndarray:
        return self.labels != UNLABELED

    def homo_adjacency(self) -> sp.csr_matrix:
        """Binarized union of all relations (the homo graph)."""
        if not self.relations:
            return sp.csr_matrix((self.num_nodes, self.num_nodes))
        mats = list(self.relations.values())
        acc = mats[0].copy().tocsr()
        for A in mats[1:]:
            acc = acc + A.tocsr()
        acc = (acc > 0).astype(np.float32)
        acc = acc.tolil()
        acc.setdiag(0)
        acc = acc.tocsr()
        acc.eliminate_zeros()
        return acc

    def validate(self) -> None:
        n = self.num_nodes
        assert self.labels.shape == (n,), "labels must be shape (n,)"
        uniq = set(np.unique(self.labels).tolist())
        assert uniq.issubset({0, 1, UNLABELED}), f"unexpected labels {uniq}"
        for r, A in self.relations.items():
            assert A.shape == (n, n), f"relation {r} has shape {A.shape}, expected {(n, n)}"
            if (A != A.T).nnz != 0:
                logger.warning("relation %s is not symmetric; symmetrizing on save is recommended", r)
        logger.info(
            "[%s] n=%d, d=%d, relations=%s, fraud=%.3f%% (labeled=%d)",
            self.name, n, self.num_features, self.relation_names,
            100.0 * self.fraud_rate(), int(self.labeled_mask().sum()),
        )

    def fraud_rate(self) -> float:
        m = self.labeled_mask()
        if m.sum() == 0:
            return float("nan")
        return float((self.labels[m] == 1).mean())

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "features": self.features.astype(np.float32),
            "labels": self.labels.astype(np.int64),
        }
        np.savez_compressed(out_dir / "graph_arrays.npz", **arrays)

        for r, A in self.relations.items():
            sp.save_npz(out_dir / f"relation_{r}.npz", A.tocsr())

        meta = {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_features": self.num_features,
            "relations": self.relation_names,
            "fraud_rate": self.fraud_rate(),
            "extra": self.meta,
        }
        (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved MultiRelationGraph '%s' to %s", self.name, out_dir)

    @classmethod
    def load(cls, out_dir: Path) -> "MultiRelationGraph":
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "graph_meta.json").read_text())
        arr = np.load(out_dir / "graph_arrays.npz")
        relations = {
            r: sp.load_npz(out_dir / f"relation_{r}.npz").tocsr()
            for r in meta["relations"]
        }
        return cls(
            name=meta["name"],
            features=arr["features"],
            labels=arr["labels"],
            relations=relations,
            meta=meta.get("extra", {}),
        )


@dataclass
class CandidateCells:
    """Candidate cells C(G): the pool of node groups the lifting scores.

    incidence: (n, m) incidence B of C(G), one column per cell.
    provenance: (m,) template code of each cell (see PROVENANCE in candidate_cells).
    apex: (m,) optional; the only node that receives each cell's message (causal cones).
        None means every member receives.
    cell_features: (m, k) optional label-free descriptors appended to the scorer input.
    The two optional fields are built on the fly by the temporal variants and not saved.
    """

    incidence: sp.csr_matrix
    provenance: np.ndarray
    provenance_names: Dict[int, str] = field(default_factory=dict)
    apex: Optional[np.ndarray] = None
    cell_features: Optional[np.ndarray] = None

    @property
    def num_cells(self) -> int:
        return self.incidence.shape[1]

    @property
    def num_nodes(self) -> int:
        return self.incidence.shape[0]

    def cell_sizes(self) -> np.ndarray:
        return np.asarray(self.incidence.sum(axis=0)).ravel().astype(np.int64)

    def save(self, out_dir: Path) -> None:
        if self.apex is not None or self.cell_features is not None:
            raise ValueError("apex and cell_features are not saved; rebuild them instead")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(out_dir / "candidates_incidence.npz", self.incidence.tocsr())
        np.savez_compressed(
            out_dir / "candidates_meta.npz",
            provenance=self.provenance.astype(np.int64),
        )
        (out_dir / "candidates_provenance.json").write_text(
            json.dumps({str(k): v for k, v in self.provenance_names.items()}, indent=2)
        )
        logger.info("Saved %d candidate cells to %s", self.num_cells, out_dir)

    @classmethod
    def load(cls, out_dir: Path) -> "CandidateCells":
        out_dir = Path(out_dir)
        inc = sp.load_npz(out_dir / "candidates_incidence.npz").tocsr()
        m = np.load(out_dir / "candidates_meta.npz")
        prov_names = json.loads((out_dir / "candidates_provenance.json").read_text())
        return cls(
            incidence=inc,
            provenance=m["provenance"],
            provenance_names={int(k): v for k, v in prov_names.items()},
        )
