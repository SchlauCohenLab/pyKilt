"""Input / Output helpers for Kilt."""

import numpy as np
import pickle
from typing import Tuple, Dict

def load_pickles(ut_path: str, mt_path: str) -> Tuple[Dict, Dict]:
    """Load two pickled photon dictionaries: micro_times (ut) and macro_times (mt)."""
    with open(ut_path, "rb") as f:
        ut = pickle.load(f)
    with open(mt_path, "rb") as f:
        mt = pickle.load(f)
    return ut, mt

def load_npz(npz_path: str) -> dict:
    return dict(np.load(npz_path, allow_pickle=True))

def save_results(path: str, result) -> None:
    """Save result dict (or dataclass) as numpy .npz. Accepts dataclasses with __dict__."""
    if hasattr(result, "__dict__"):
        np.savez(path, **result.__dict__)
    elif isinstance(result, dict):
        np.savez(path, **result)
    else:
        raise TypeError("result must be dataclass or dict")

def standardize_photon_arrays(macrotimes, microtimes, tau_min=0.0, tau_max=None):
    """Basic cleaning used in scripts: remove NaN, apply modulus if necessary."""
    import numpy as np
    macrotimes = np.nan_to_num(macrotimes, 0.0)
    macrotimes = macrotimes[macrotimes > 0.0]
    if tau_max is not None:
        mask = (microtimes > tau_min) & (microtimes < tau_max)
    else:
        mask = microtimes > tau_min
    microtimes = np.nan_to_num(microtimes, 0.0)
    microtimes = microtimes[mask]
    macrotimes = macrotimes[mask]
    return macrotimes.astype(np.float64), microtimes.astype(np.float64)
