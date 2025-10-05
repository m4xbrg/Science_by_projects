"""
Core model for the linear damped harmonic oscillator.

Equations:
    m * x_ddot + c * x_dot + k * x = 0
State:
    y = [x, v] with v = x_dot
API:
    - rhs(t, y, params): returns dy/dt
    - energy(x, v, params): mechanical energy
    - damping_ratio(params): zeta = c / (2*sqrt(k*m))
"""
from __future__ import annotations
from math import sqrt
from typing import Dict, Sequence

import numpy as np


def rhs(t: float, y: Sequence[float], params: Dict[str, float]) -> np.ndarray:
    """Right-hand side of the first-order system.

    Args:
        t: time [s] (unused; included for ODE solver API)
        y: state vector [x, v], where v = dx/dt
        params: dict with keys 'm', 'c', 'k'

    Returns:
        np.ndarray of shape (2,) with [dx/dt, dv/dt].
    """
    x, v = y  # x: displacement, v: velocity
    m = float(params["m"])
    c = float(params["c"])
    k = float(params["k"])

    dxdt = v
    dvdt = -(c/m) * v - (k/m) * x  # from m x'' + c x' + k x = 0

    return np.array([dxdt, dvdt], dtype=float)


def energy(x: np.ndarray, v: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Mechanical energy E = 0.5*m*v^2 + 0.5*k*x^2.

    Args:
        x: displacement array
        v: velocity array
        params: dict with 'm', 'k'

    Returns:
        Energy array E(t).
    """
    m = float(params["m"])
    k = float(params["k"])
    return 0.5 * m * np.asarray(v) ** 2 + 0.5 * k * np.asarray(x) ** 2


def damping_ratio(params: Dict[str, float]) -> float:
    """Return the non-dimensional damping ratio zeta = c/(2*sqrt(k*m))."""
    m = float(params["m"])
    c = float(params["c"])
    k = float(params["k"])
    return c / (2.0 * sqrt(k * m))