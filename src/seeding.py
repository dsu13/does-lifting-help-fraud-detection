"""Seeding helpers; call set_seed at the top of every script and before each split."""
from __future__ import annotations

import os
import random
import logging

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (if installed), and set the deterministic cuDNN flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional for the data steps
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("Seeded torch with %d", seed)
    except ImportError:
        pass

    logger.debug("Seeded python/numpy with %d", seed)


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh NumPy Generator (preferred for split code)."""
    return np.random.default_rng(seed)
