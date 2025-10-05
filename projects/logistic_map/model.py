"""
model.py — Core logistic map definitions.

API targets portfolio consistency:
- rhs(n, x, params): one iteration step x_{n+1} = f(x_n; params)
- jacobian(x, params): derivative f'(x; params)
- iterate_map(...): helper to run many steps and statistics
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class LogisticParams:
    """Container for logistic map parameters."""
    r: float

def rhs(n: int, x: np.ndarray | float, params: LogisticParams) -> np.ndarray | float:
    """
    One-step map update for the logistic map.
    x_{n+1} = r * x_n * (1 - x_n)

    Parameters
    ----------
    n : int
        Discrete time index (unused; present for API consistency).
    x : array-like or float
        Current state(s) in [0,1].
    params : LogisticParams
        Parameters with control r.

    Returns
    -------
    array-like or float
        Next state(s).
    """
    r = params.r
    return r * np.asarray(x) * (1.0 - np.asarray(x))

def jacobian(x: np.ndarray | float, params: LogisticParams) -> np.ndarray | float:
    """
    Derivative of the logistic map with respect to x.
    f'(x; r) = r * (1 - 2x).

    Parameters
    ----------
    x : array-like or float
    params : LogisticParams

    Returns
    -------
    array-like or float
        Derivative evaluated at x.
    """
    r = params.r
    return r * (1.0 - 2.0 * np.asarray(x))

def iterate_map(r_values: np.ndarray, x0: float, n_transient: int, n_iter: int, n_keep: int, clip_eps: float = 1e-15):
    """
    Iterate the logistic map for multiple r values, returning bifurcation points and Lyapunov exponents.

    Parameters
    ----------
    r_values : np.ndarray
        1D array of r parameters.
    x0 : float
        Initial condition in [0,1].
    n_transient : int
        Number of burn-in iterations to discard.
    n_iter : int
        Number of iterations used for statistics.
    n_keep : int
        Number of final iterations to keep for bifurcation diagram (<= n_iter).
    clip_eps : float
        Small epsilon to clip |f'(x)| away from zero for log stability.

    Returns
    -------
    bifurcation_r : np.ndarray
        Repeated r values corresponding to stored x samples.
    bifurcation_x : np.ndarray
        Stored x samples from final n_keep iterations for each r.
    lyapunov : np.ndarray
        Lyapunov exponent estimate for each r.
    """
    r_values = np.asarray(r_values, dtype=float)
    m = r_values.size

    # Vectorized evolution over r grid
    x = np.full(m, x0, dtype=float)

    # Burn-in
    for _ in range(n_transient):
        x = r_values * x * (1.0 - x)

    # Statistics
    lyap_sum = np.zeros(m, dtype=float)
    keep_buffer = np.empty((n_keep, m), dtype=float)  # store last n_keep states

    for k in range(n_iter):
        x = r_values * x * (1.0 - x)
        deriv = np.abs(r_values * (1.0 - 2.0 * x))
        deriv = np.maximum(deriv, clip_eps)
        lyap_sum += np.log(deriv)
        # Rolling buffer for last n_keep steps
        if k >= n_iter - n_keep:
            keep_buffer[k - (n_iter - n_keep), :] = x

    lyapunov = lyap_sum / float(n_iter)

    # Flatten for scatter
    bifurcation_r = np.repeat(r_values[None, :], n_keep, axis=0).reshape(-1)
    bifurcation_x = keep_buffer.reshape(-1)

    return bifurcation_r, bifurcation_x, lyapunov
