import numpy as np
from math import erf

def simulate_pvalues(m: int, pi1: float, mu: float, rng: np.random.Generator):
    """Generate m two-sided p-values under a Gaussian two-groups model."""
    is_alt = rng.random(m) < pi1
    z = rng.normal(0.0, 1.0, size=m)
    z[is_alt] = rng.normal(mu, 1.0, size=is_alt.sum())
    absz = np.abs(z)
    Phi = 0.5 * (1 + np.vectorize(erf)(absz / np.sqrt(2.0)))
    p = 2 * (1 - Phi)
    return p, ~is_alt

def bonferroni_reject(pvals: np.ndarray, alpha: float):
    """Bonferroni: reject p_i <= alpha/m."""
    m = pvals.size
    return pvals <= (alpha / m)

def bh_reject(pvals: np.ndarray, q: float):
    """Benjamini–Hochberg step-up at level q."""
    m = pvals.size
    idx = np.argsort(pvals)
    p_sorted = pvals[idx]
    thresh = q * (np.arange(1, m + 1) / m)
    hits = p_sorted <= thresh
    if not np.any(hits):
        return np.zeros(m, dtype=bool)
    k = np.max(np.where(hits)[0]) + 1
    cutoff = p_sorted[k - 1]
    return pvals <= cutoff

def evaluate_once(p: np.ndarray, is_null: np.ndarray, reject_mask: np.ndarray):
    """Return (R,V,S,FDR,FWER,TPR) for one replicate."""
    R = int(reject_mask.sum())
    V = int((reject_mask & is_null).sum())
    S = int((reject_mask & ~is_null).sum())
    fdr = (V / R) if R > 0 else 0.0
    fwer = 1.0 if V > 0 else 0.0
    tpr = S / max(1, (~is_null).sum())
    return R, V, S, fdr, fwer, tpr