"""
Core differential equation models and analytic utilities for Logistic and Riccati equations.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple

Array = np.ndarray


@dataclass
class LogisticParams:
    r: float
    K: float


@dataclass
class RiccatiParams:
    # Constant-coefficient Riccati: x' = a x^2 + b x + c
    a: float
    b: float
    c: float


def rhs_logistic(t: float, x: Array, params: Dict) -> Array:
    """
    Logistic ODE right-hand side: dx/dt = r x (1 - x/K)

    Parameters
    ----------
    t : float
        Time (unused; autonomous system)
    x : np.ndarray, shape (1,) or scalar array
        State variable
    params : dict with keys {"r", "K"}
        r : intrinsic growth rate
        K : carrying capacity

    Returns
    -------
    np.ndarray
        dx/dt evaluated at (t, x)
    """
    r = float(params["r"])
    K = float(params["K"])
    x_val = float(x[0]) if np.ndim(x) else float(x)
    return np.array([r * x_val * (1.0 - x_val / K)])


def rhs_riccati(t: float, x: Array, params: Dict) -> Array:
    """
    Constant-coefficient Riccati: dx/dt = a x^2 + b x + c
    """
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    x_val = float(x[0]) if np.ndim(x) else float(x)
    return np.array([a * x_val**2 + b * x_val + c])


def logistic_analytic(t: Array, x0: float, params: Dict) -> Array:
    """
    Closed-form solution to the logistic ODE with constant parameters.
    x(t) = K / (1 + ((K - x0)/x0) * exp(-r t))

    Parameters
    ----------
    t : array-like
        Times at which to evaluate the solution
    x0 : float
        Initial condition x(0)
    params : dict with keys {"r", "K"}

    Returns
    -------
    np.ndarray
        x(t) values
    """
    r = float(params["r"])
    K = float(params["K"])
    t = np.asarray(t, dtype=float)
    if x0 == 0.0:
        return np.zeros_like(t)
    denom = 1.0 + ((K - x0) / x0) * np.exp(-r * t)
    return K / denom


def jacobian_logistic(x: float, params: Dict) -> float:
    """
    ∂f/∂x for logistic f(x) = r x (1 - x/K) = r x - (r/K) x^2
    """
    r = float(params["r"])
    K = float(params["K"])
    return r * (1.0 - 2.0 * x / K)


def sensitivities_logistic_rhs(
    t: float, y: Array, params: Dict, wrt: Tuple[str, ...]
) -> Array:
    """
    Augmented ODE for logistic sensitivities using forward sensitivities.

    State: [x, s_r, s_K] for wrt = ("r","K") subset.
    General form: ds/dt = ∂f/∂x * s + ∂f/∂p

    Parameters
    ----------
    t : float
    y : np.ndarray, shape (1 + len(wrt),)
        [x, s_1, s_2, ...]
    params : dict with keys {"r","K"}
    wrt : tuple of parameter names to differentiate wrt (subset of {"r","K"})

    Returns
    -------
    np.ndarray
        Time derivative of augmented state.
    """
    x = y[0]
    r = float(params["r"])
    K = float(params["K"])
    # Base RHS
    fx = r * x * (1.0 - x / K)
    # Jacobian wrt x
    dfdx = r * (1.0 - 2.0 * x / K)
    # Parameter partials
    dfd_r = x * (1.0 - x / K)
    dfd_K = r * x * (x / (K**2))

    # Build derivative vector
    dy = np.empty_like(y)
    dy[0] = fx

    idx = 1
    for name in wrt:
        s = y[idx]
        if name == "r":
            dy[idx] = dfdx * s + dfd_r
        elif name == "K":
            dy[idx] = dfdx * s + dfd_K
        else:
            raise ValueError(f"Unsupported sensitivity parameter: {name}")
        idx += 1
    return dy
