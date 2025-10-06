"""
model.py — Core stochastic model: standardized sums of i.i.d. random variables.
API centers on `standardized_sum_samples`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np

# ---- Distribution registry ----


def _rv_bernoulli(rng: np.random.Generator, size: int, p: float = 0.5) -> np.ndarray:
    return rng.binomial(1, p, size=size).astype(float)


def _rv_uniform(
    rng: np.random.Generator, size: int, a: float = 0.0, b: float = 1.0
) -> np.ndarray:
    return rng.uniform(a, b, size=size)


def _rv_exponential(
    rng: np.random.Generator, size: int, lam: float = 1.0
) -> np.ndarray:
    return rng.exponential(1.0 / lam, size=size)


def _rv_pareto(rng: np.random.Generator, size: int, alpha: float = 3.0) -> np.ndarray:
    # Shifted/Scaled Pareto with mean  alpha/(alpha-1) if alpha>1; var finite if alpha>2
    return rng.pareto(alpha, size=size) + 1.0


_DIST: Dict[str, Callable] = {
    "bernoulli": _rv_bernoulli,
    "uniform": _rv_uniform,
    "exponential": _rv_exponential,
    "pareto": _rv_pareto,
}


def population_moments(name: str, params: Dict) -> Tuple[float, float]:
    """Return (mu, sigma2) for supported distributions. Raises if not finite."""
    if name == "bernoulli":
        p = params.get("p", 0.5)
        mu = p
        sigma2 = p * (1 - p)
        return mu, sigma2
    if name == "uniform":
        a = params.get("a", 0.0)
        b = params.get("b", 1.0)
        mu = 0.5 * (a + b)
        sigma2 = (b - a) ** 2 / 12.0
        return mu, sigma2
    if name == "exponential":
        lam = params.get("lam", 1.0)
        mu = 1.0 / lam
        sigma2 = 1.0 / (lam**2)
        return mu, sigma2
    if name == "pareto":
        alpha = params.get("alpha", 3.0)
        if alpha <= 2:
            raise ValueError(
                "Pareto variance is infinite for alpha<=2; CLT does not apply in classical form."
            )
        mu = alpha / (alpha - 1.0)
        sigma2 = (alpha) / ((alpha - 1.0) ** 2 * (alpha - 2.0))
        return mu, sigma2
    raise KeyError(f"Unknown distribution: {name}")


@dataclass
class IIDSpec:
    name: str
    params: Dict


def sample_iid(rng: np.random.Generator, spec: IIDSpec, size: int) -> np.ndarray:
    """Draw i.i.d. samples X_1,...,X_size from the specified distribution."""
    if spec.name not in _DIST:
        raise KeyError(f"Unknown distribution: {spec.name}")
    return _DIST[spec.name](rng, size=size, **spec.params)


def standardized_sum_samples(
    rng: np.random.Generator,
    spec: IIDSpec,
    n: int,
    trials: int,
    use_population_moments: bool = True,
) -> np.ndarray:
    r"""
    Return Z_n samples defined by:
        Z_n = (S_n - n μ) / (σ √n),   S_n = ∑_{i=1}^n X_i
    where {X_i} are i.i.d. with mean μ and variance σ^2.
    If use_population_moments=False, replace μ,σ with sample mean/var per-trial.
    """
    # draw an array of shape (trials, n)
    X = sample_iid(rng, spec, size=trials * n).reshape(trials, n)
    if use_population_moments:
        mu, sigma2 = population_moments(spec.name, spec.params)
        S = X.sum(axis=1)
        Z = (S - n * mu) / (np.sqrt(sigma2) * np.sqrt(n))
        return Z
    else:
        # Studentized variant: center/scale by within-trial estimates
        S = X.sum(axis=1)
        m = X.mean(axis=1)
        s = X.std(axis=1, ddof=1)
        Z = (S - n * m) / (s * np.sqrt(n))
        return Z
