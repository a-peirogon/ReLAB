from __future__ import annotations
import random
from typing import Optional

def set_seed(seed: Optional[int]) -> None:
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
