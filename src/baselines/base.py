"""Baseline interface and name registry.

A baseline implements fit(graph, train_idx, val_idx) and predict_proba(graph),
which returns P(fraud) for every node, shape (n,).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np


class BaselineModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, graph, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict_proba(self, graph) -> np.ndarray:
        """Return P(fraud) for every node (shape (n,))."""
        ...


# name -> constructor
REGISTRY: Dict[str, Callable[..., BaselineModel]] = {}


def register(name: str) -> Callable[[Callable[..., BaselineModel]], Callable[..., BaselineModel]]:
    def deco(ctor: Callable[..., BaselineModel]) -> Callable[..., BaselineModel]:
        REGISTRY[name] = ctor
        return ctor
    return deco


def get_model(name: str, **kwargs) -> BaselineModel:
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def available() -> List[str]:
    return sorted(REGISTRY)


# importing the modules fills the registry
def _autoregister() -> None:
    from . import tabular      # noqa: F401
    from . import gnn          # noqa: F401
    from . import hypergraph   # noqa: F401


_autoregister()
