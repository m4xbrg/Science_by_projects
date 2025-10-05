"""
Resampling engines for bootstrap/permutation CIs and coverage experiments.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

Array = np.ndarray

# ---------------------- Distributions ----------------------
def sample_distribution(rng: np.random.Generator, family: str, n: int, params: Dict) -> Array:
    """
    Draws i.i.d. samples from a named family.
    Supported families: normal(mu, sigma), lognormal(mu, sigma), t(df), laplace(mu, b)
    """
    if family == "normal":
        mu = params.get("mu", 0.0); sigma = params.get("sigma", 1.0)
        return rng.normal(mu, sigma, size=n)
    if family == "lognormal":
        mu = params.get("mu", 0.0); sigma = params.get("sigma", 1.0)
        return rng.lognormal(mean=mu, sigma=sigma, size=n)
    if family == "t":
        df = params.get("df", 5.0)
        return rng.standard_t(df, size=n)
    if family == "laplace":
        mu = params.get("mu", 0.0); b = params.get("b", 1.0)
        return rng.laplace(mu, b, size=n)
    raise ValueError(f"Unknown family: {family}")


def true_parameter(family: str, params: Dict, statistic: str) -> float:
    """
    True parameter theta(F) for selected statistic.
    For lognormal, mean = exp(mu + 0.5*sigma^2), median = exp(mu).
    For t(df), mean=0 if df>1; median=0 (symmetric); for laplace(mu,b), mean=mu, median=mu.
    """
    if statistic == "mean":
        if family == "normal":
            return float(params.get("mu", 0.0))
        if family == "lognormal":
            mu = params.get("mu", 0.0); sigma = params.get("sigma", 1.0)
            return float(np.exp(mu + 0.5 * sigma**2))
        if family == "t":
            df = params.get("df", 5.0)
            return 0.0 if df > 1 else np.nan
        if family == "laplace":
            return float(params.get("mu", 0.0))
    if statistic == "median":
        if family in ("normal", "laplace", "t"):
            return float(params.get("mu", 0.0))
        if family == "lognormal":
            mu = params.get("mu", 0.0)
            return float(np.exp(mu))
    raise ValueError(f"No closed-form true parameter for (family={family}, statistic={statistic}).")


# ---------------------- Statistics ----------------------
def compute_statistic(x: Array, statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(x))
    if statistic == "median":
        return float(np.median(x))
    raise ValueError(f"Unknown statistic: {statistic}")


# ---------------------- Bootstrap CIs ----------------------
def bootstrap_statistics(rng: np.random.Generator, x: Array, B: int, statistic: str) -> Array:
    """Return bootstrap replicate statistics T* for B resamples with replacement."""
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    samples = x[idx]
    if statistic == "mean":
        return samples.mean(axis=1)
    if statistic == "median":
        return np.median(samples, axis=1)
    raise ValueError(f"Unknown statistic: {statistic}")


def ci_percentile(Tstar: Array, alpha: float) -> Tuple[float, float]:
    lo = float(np.quantile(Tstar, alpha/2, method="linear"))
    hi = float(np.quantile(Tstar, 1 - alpha/2, method="linear"))
    return lo, hi


def ci_basic(T: float, Tstar: Array, alpha: float) -> Tuple[float, float]:
    q_lo = float(np.quantile(Tstar, 1 - alpha/2, method="linear"))
    q_hi = float(np.quantile(Tstar, alpha/2, method="linear"))
    # Basic CI reflects around T: [2T - q_hi, 2T - q_lo]
    return 2*T - q_lo, 2*T - q_hi


def one_sample_ci(rng: np.random.Generator, x: Array, statistic: str, method: str, B: int, alpha: float):
    """Compute T, (lo,hi) for one-sample CI via bootstrap method."""
    T = compute_statistic(x, statistic)
    Tstar = bootstrap_statistics(rng, x, B, statistic)
    if method == "percentile":
        lo, hi = ci_percentile(Tstar, alpha)
    elif method == "basic":
        lo, hi = ci_basic(T, Tstar, alpha)
    else:
        raise ValueError(f"Unknown CI method: {method}")
    return T, lo, hi


# ---------------------- Permutation (extension hook) ----------------------
from dataclasses import dataclass

@dataclass
class TwoSampleSpec:
    enabled: bool = False
    n_b: int = 50
    effect: float = 0.0  # mean shift for group B
