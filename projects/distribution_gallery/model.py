"""
model.py — Probability distribution models and sampling utilities.
Clean API: DistributionModel implements pdf/pmf, cdf, rvs, and moments.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats

ArrayLike = Union[np.ndarray, float, int]

@dataclass(frozen=True)
class DistributionSpec:
    family: str           # "normal" | "exponential" | "poisson"
    kind: str             # "continuous" | "discrete"
    params: Dict[str, float]

class DistributionModel:
    """
    Unified interface to common distributions.
    Maps mathematical definitions to SciPy implementations.
    """
    def __init__(self, spec: DistributionSpec):
        self.spec = spec
        fam = spec.family.lower()
        if fam == "normal":
            self._dist = stats.norm(loc=spec.params.get("mu", 0.0),
                                     scale=spec.params.get("sigma", 1.0))
        elif fam == "exponential":
            # rate = λ, scale = 1/λ in SciPy's parameterization
            lam = spec.params.get("rate", 1.0)
            self._dist = stats.expon(scale=1.0/lam)
        elif fam == "poisson":
            lam = spec.params.get("lambda", 1.0)
            self._dist = stats.poisson(mu=lam)
        else:
            raise ValueError(f"Unsupported family: {fam}")

    # ---- Core math ↔ code mappings ----
    def pdf(self, x: ArrayLike) -> np.ndarray:
        """Return f_X(x) for continuous distributions; NaN for discrete."""
        if self.spec.kind != "continuous":
            return np.full_like(np.asarray(x, dtype=float), np.nan)
        return self._dist.pdf(x)

    def pmf(self, k: ArrayLike) -> np.ndarray:
        """Return P(X=k) for discrete distributions; NaN for continuous."""
        if self.spec.kind != "discrete":
            return np.full_like(np.asarray(k, dtype=float), np.nan)
        return self._dist.pmf(k)

    def cdf(self, x: ArrayLike) -> np.ndarray:
        """Return F_X(x) for continuous or discrete distributions."""
        return self._dist.cdf(x)

    def rvs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n iid samples according to the distribution."""
        return self._dist.rvs(size=n, random_state=rng)

    def theoretical_moments(self) -> Dict[str, float]:
        """Return mean, variance, skewness, kurtosis (excess) analytically where available."""
        fam = self.spec.family.lower()
        p = self.spec.params
        if fam == "normal":
            mu, sigma = p.get("mu", 0.0), p.get("sigma", 1.0)
            return {"mean": mu, "var": sigma**2, "skew": 0.0, "kurtosis_excess": 0.0}
        elif fam == "exponential":
            lam = p.get("rate", 1.0)
            return {"mean": 1/lam, "var": 1/lam**2, "skew": 2.0, "kurtosis_excess": 6.0}
        elif fam == "poisson":
            lam = p.get("lambda", 1.0)
            return {"mean": lam, "var": lam, "skew": 1/np.sqrt(lam), "kurtosis_excess": 1/lam}
        else:
            raise ValueError(f"Unsupported family: {fam}")

def make_model(family: str, kind: str, params: Dict[str, float]) -> DistributionModel:
    """Factory to build a DistributionModel from basic fields."""
    return DistributionModel(DistributionSpec(family=family, kind=kind, params=params))
