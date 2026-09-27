"""Baseline models, their name registry and the metric suite."""
from .base import BaselineModel, REGISTRY, get_model, available   # noqa: F401
from .metrics import evaluate, METRIC_NAMES                        # noqa: F401
