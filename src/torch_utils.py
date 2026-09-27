"""scipy-to-torch sparse conversion, shared by the baselines and the lifting model."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def scipy_to_torch_sparse(A: sp.spmatrix, device) -> torch.Tensor:
    coo = A.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    val = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=coo.shape, device=device).coalesce()
