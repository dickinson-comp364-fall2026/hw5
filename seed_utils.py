import os
import random
from typing import Optional

DEFAULT_SEED = 1234
SEED_ENV_VAR = "PROJECT_SEED"


def get_seed(default: int = DEFAULT_SEED) -> int:
    raw_value = os.environ.get(SEED_ENV_VAR)
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def seed_everything(seed: Optional[int] = None) -> int:
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
    resolved_seed = get_seed() if seed is None else seed
    return random.Random(resolved_seed)
