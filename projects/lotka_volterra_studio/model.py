"""
model.py — Lotka–Volterra model primitives

Clean API:
- rhs(t, x, params)
- jacobian(x, params)
- fixed_points(params)
- nullclines(grid, params)

All functions are pure where possible.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class LVParams:
    alpha: float  # prey growth rate
    beta: float  # predation rate
    gamma: float  # predator death rate
    delta: float  # predator reproduction per prey consumed


def rhs(t: float, x: np.ndarray, params: LVParams) -> np.ndarray:
    """
    Right-hand side of the LV ODEs.
    dx/dt = x * (alpha - beta*y)
    dy/dt = y * (-gamma + delta*x)

    Args:
        t: time (unused; included for ODE API compatibility)
        x: state vector [x, y] with x>=0, y>=0
        params: LVParams
    Returns:
        np.ndarray shape (2,) of time derivatives
    """
    X, Y = x
    a, b, g, d = params.alpha, params.beta, params.gamma, params.delta
    return np.array([X * (a - b * Y), Y * (-g + d * X)], dtype=float)


def jacobian(x: np.ndarray, params: LVParams) -> np.ndarray:
    """
    Jacobian matrix of rhs at state x.
    J = [[alpha - beta*y, -beta*x],
         [delta*y,       -gamma + delta*x]]
    """
    X, Y = x
    a, b, g, d = params.alpha, params.beta, params.gamma, params.delta
    return np.array([[a - b * Y, -b * X], [d * Y, -g + d * X]], dtype=float)


def fixed_points(params: LVParams) -> np.ndarray:
    """
    Compute equilibrium points for LV.
    Returns an array of shape (k,2):
        - (0,0)
        - (gamma/delta, alpha/beta) if parameters >0
    """
    fp = [np.array([0.0, 0.0])]
    if params.beta > 0 and params.delta > 0:
        fp.append(np.array([params.gamma / params.delta, params.alpha / params.beta]))
    return np.vstack(fp)


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    n: int


def nullclines(grid: GridSpec, params: LVParams) -> dict:
    """
    Compute nullclines in the phase plane:
        dx/dt = 0 -> x=0 or y = alpha/beta
        dy/dt = 0 -> y=0 or x = gamma/delta
    Returns dict with arrays for plotting convenience.
    """
    xs = np.linspace(grid.x_min, grid.x_max, grid.n)
    ys = np.linspace(grid.y_min, grid.y_max, grid.n)
    y_dx0 = np.full_like(xs, params.alpha / params.beta) if params.beta > 0 else np.nan
    x_dy0 = (
        np.full_like(ys, params.gamma / params.delta) if params.delta > 0 else np.nan
    )
    return {
        "x_axis": xs,
        "y_axis": ys,
        "dx0_y": y_dx0,  # horizontal line y = alpha/beta
        "dy0_x": x_dy0,  # vertical line x = gamma/delta
    }
