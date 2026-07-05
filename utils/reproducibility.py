"""
Reproducibility helpers.

Centralizes seeding of every RNG source (python's `random`, numpy, torch)
so experiments can be replayed exactly. Previously this logic was
duplicated inline in run_train.py / run_watch.py — now it lives here.
"""

from __future__ import annotations

import random
from typing import Optional


def set_seed(seed: Optional[int]) -> None:
    """Seed python, numpy and torch (if available) RNGs.

    Passing `None` is a no-op (useful for "unseeded" runs).
    """
    if seed is None:
        return

    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
