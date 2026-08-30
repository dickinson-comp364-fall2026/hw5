"""Utility functions for setting and managing random seeds."""

import os
import random
from typing import Optional

DEFAULT_SEED = 1234
SEED_ENV_VAR = "PROJECT_SEED"


def get_seed(default: int = DEFAULT_SEED) -> int:
    """Get the random seed from the environment or use a default."""
    raw_value = os.environ.get(SEED_ENV_VAR)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def seed_everything(seed: Optional[int] = None) -> int:
    """Set the random seed for all relevant libraries."""
    resolved_seed = get_seed() if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(resolved_seed)
    random.seed(resolved_seed)

    try:
        import numpy as np

        np.random.seed(resolved_seed)
    except ImportError:
        pass

    return resolved_seed


def make_rng(seed: Optional[int] = None) -> random.Random:
    """Create a new random number generator with a specific seed."""
    resolved_seed = get_seed() if seed is None else seed
    return random.Random(resolved_seed)
