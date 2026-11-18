"""Synthetic low-light degradation (report Section 3):

    I_low = clip( N( beta * I_gt ** gamma ) ),   beta in [0.2, 0.8],  gamma in [1.5, 3.0]

N(.) adds signal-dependent (Poisson) noise and Gaussian sensor noise. This is a
*training-data synthesis* model only; the network never sees beta, gamma or the
noise parameters and must infer the degradation from the image.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

BETA_RANGE = (0.2, 0.8)
GAMMA_RANGE = (1.5, 3.0)
PEAK_RANGE = (40.0, 300.0)       # Poisson "photon budget": smaller = noisier
SIGMA_RANGE = (0.005, 0.03)      # Gaussian read noise (std, in [0,1] units)


def synthesize_low_light(gt: np.ndarray, rng: np.random.Generator,
                         beta: Optional[float] = None, gamma: Optional[float] = None,
                         peak: Optional[float] = None, sigma: Optional[float] = None,
                         noise: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    """gt: float32 HxWx3 in [0,1]. Returns (low, params)."""
    beta = float(rng.uniform(*BETA_RANGE)) if beta is None else float(beta)
    gamma = float(rng.uniform(*GAMMA_RANGE)) if gamma is None else float(gamma)
    peak = float(rng.uniform(*PEAK_RANGE)) if peak is None else float(peak)
    sigma = float(rng.uniform(*SIGMA_RANGE)) if sigma is None else float(sigma)

    x = beta * np.power(np.clip(gt, 0.0, 1.0), gamma)
    if noise:
        x = rng.poisson(np.clip(x, 0, None) * peak).astype(np.float32) / peak     # signal-dependent
        x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)           # sensor noise
    low = np.clip(x, 0.0, 1.0).astype(np.float32)
    return low, dict(beta=beta, gamma=gamma, peak=peak, sigma=sigma)


def noise_level(params: Dict[str, float]) -> float:
    """Scalar proxy for how noisy a sample is (used to split 'noisy' vs 'clean' subsets)."""
    return params["sigma"] + 0.5 / np.sqrt(params["peak"])
