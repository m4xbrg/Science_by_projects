import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class ZParams:
    tau: float = 3.0


@dataclass
class ZRobustParams:
    tau: float = 3.5
    mad_eps: float = 1e-9  # additive jitter to avoid divide-by-zero


@dataclass
class IQRParams:
    k: float = 1.5


def classical_z_scores(
    x: np.ndarray, params: ZParams
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute classical z-scores using sample mean and std.
    Returns (mask, stats).
    """
    mu = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    # Map to math: z_i = (x_i - mu) / s
    z = (x - mu) / s if s > 0 else np.zeros_like(x)
    mask = np.abs(z) > params.tau
    return mask, {"mu": mu, "std": s, "tau": params.tau}


def robust_z_scores(
    x: np.ndarray, params: ZRobustParams
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute robust z-scores using median and MAD scaled by 1.4826.
    Returns (mask, stats).
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad + params.mad_eps
    z = (x - med) / s
    mask = np.abs(z) > params.tau
    return mask, {"median": med, "mad": mad, "scale": s, "tau": params.tau}


def iqr_fences(x: np.ndarray, params: IQRParams) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute IQR fences (Tukey): [Q1 - k*IQR, Q3 + k*IQR].
    Returns (mask, stats).
    """
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - params.k * iqr
    hi = q3 + params.k * iqr
    mask = (x < lo) | (x > hi)
    return mask, {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "k": params.k,
        "lo": float(lo),
        "hi": float(hi),
    }


def simulate_contaminated_sample(
    n: int,
    epsilon: float,
    rng: np.random.Generator,
    base_mu: float,
    base_sigma: float,
    out_mu: float,
    out_sigma: float,
):
    """
    Draw n points from F_epsilon = (1-eps)N(base_mu, base_sigma^2) + eps N(out_mu, out_sigma^2).
    Returns (x, y), where y=1 denotes an outlier.
    """
    n_out = int(round(epsilon * n))
    n_in = n - n_out
    x_in = rng.normal(loc=base_mu, scale=base_sigma, size=n_in)
    x_out = rng.normal(loc=out_mu, scale=out_sigma, size=n_out)
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([np.zeros(n_in, dtype=int), np.ones(n_out, dtype=int)])
    # Shuffle to remove ordering
    perm = rng.permutation(n)
    return x[perm], y[perm]


def prf(mask: np.ndarray, y: np.ndarray):
    """
    Compute precision, recall, F1 for binary outlier mask vs true labels y in {0,1}.
    """
    tp = int(np.sum((mask == 1) & (y == 1)))
    fp = int(np.sum((mask == 1) & (y == 0)))
    fn = int(np.sum((mask == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return float(precision), float(recall), float(f1)
