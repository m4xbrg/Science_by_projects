"""
model.py — Conjugate Bayesian models: Beta–Binomial and Normal–Normal (known variance).

API goals:
- Clean, modular functions/classes with clear math-to-code mapping.
- No hardcoded params: all configurable via function args or config.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

# ------------------ Beta–Binomial ------------------


@dataclass
class BetaBinomialParams:
    alpha: float  # prior alpha > 0
    beta: float  # prior beta  > 0


def beta_posterior_params(
    prior: BetaBinomialParams, k: int, n: int
) -> BetaBinomialParams:
    """
    Closed-form conjugate update for Beta–Binomial.
    Args:
        prior: BetaBinomialParams (alpha, beta)
        k: number of successes
        n: number of trials
    Returns:
        BetaBinomialParams for posterior.
    """
    return BetaBinomialParams(alpha=prior.alpha + k, beta=prior.beta + n - k)


def beta_prior_pdf(p: np.ndarray, prior: BetaBinomialParams) -> np.ndarray:
    """Beta prior density evaluated at vector p in [0,1]."""
    return beta_dist.pdf(p, prior.alpha, prior.beta)


def beta_likelihood_curve(p: np.ndarray, k: int, n: int) -> np.ndarray:
    """
    Likelihood (up to normalization) as a function of p for Binomial(k|n,p) ∝ p^k (1-p)^(n-k).
    Returns unnormalized curve; use for comparative plotting vs posterior.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(p, k) * np.power(1.0 - p, n - k)


def beta_posterior_pdf(p: np.ndarray, posterior: BetaBinomialParams) -> np.ndarray:
    """Posterior Beta density evaluated at p."""
    return beta_dist.pdf(p, posterior.alpha, posterior.beta)


# ------------------ Normal–Normal (known variance) ------------------


@dataclass
class NormalNormalParams:
    mu0: float  # prior mean
    tau0_2: float  # prior variance tau0^2 > 0


def normal_posterior_params(
    prior: NormalNormalParams, sigma2: float, xbar: float, n: int
) -> Tuple[float, float]:
    """
    Closed-form Normal–Normal posterior for the mean with known variance sigma2.
    Returns (mu_n, tau_n2) where tau_n2 is posterior variance.
    """
    precision0 = 1.0 / prior.tau0_2
    precision_like = n / sigma2
    tau_n2 = 1.0 / (precision0 + precision_like)
    mu_n = tau_n2 * (precision0 * prior.mu0 + precision_like * xbar)
    return mu_n, tau_n2


def normal_prior_pdf(mu: np.ndarray, prior: NormalNormalParams) -> np.ndarray:
    """Gaussian prior density over mu."""
    return norm.pdf(mu, loc=prior.mu0, scale=np.sqrt(prior.tau0_2))


def normal_likelihood_curve(
    mu: np.ndarray, xbar: float, n: int, sigma2: float
) -> np.ndarray:
    """
    Likelihood (up to normalization) of mu given sample mean xbar and n, with known sigma2.
    For Gaussian observation model, the sufficient statistic is xbar.
    Curve is proportional to Normal(mu | xbar, sigma2/n).
    """
    return norm.pdf(mu, loc=xbar, scale=np.sqrt(sigma2 / n))


def normal_posterior_pdf(mu: np.ndarray, mu_n: float, tau_n2: float) -> np.ndarray:
    """Posterior density over mu."""
    return norm.pdf(mu, loc=mu_n, scale=np.sqrt(tau_n2))


# ------------------ Utilities ------------------


def grid01(n: int = 1001) -> np.ndarray:
    """Dense grid on [0,1] for Beta/Binomial plots."""
    return np.linspace(0.0, 1.0, n)


def gridR(lo: float, hi: float, n: int = 1001) -> np.ndarray:
    """Dense real line segment grid for Normal/Normal plots."""
    return np.linspace(lo, hi, n)
