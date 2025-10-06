"""
Core model definitions for parameter identification on ODEs (pendulum case).
Provides the ODE right-hand side and helpers for simulation and sensitivities.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Callable, Tuple
from scipy.integrate import solve_ivp


def rhs(t: float, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    ODE right-hand side for a damped simple pendulum.
    Math:
        x1' = x2
        x2' = -theta1 * sin(x1) - theta2 * x2
    Args:
        t: time (unused, autonomous system)
        x: state vector [x1, x2]
        params: dict with keys 'theta1' (g/L), 'theta2' (damping)
    Returns:
        dxdt: np.ndarray of shape (2,)
    """
    theta1 = params["theta1"]
    theta2 = params["theta2"]
    x1, x2 = x
    return np.array([x2, -theta1 * np.sin(x1) - theta2 * x2])


def simulate_trajectory(
    t_grid: np.ndarray,
    x0: np.ndarray,
    params: Dict[str, float],
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> np.ndarray:
    """
    Integrate the ODE on a fixed grid using SciPy's solve_ivp (RK45).
    Args:
        t_grid: strictly increasing 1D array of times
        x0: initial condition array [x1_0, x2_0]
        params: parameter dict {'theta1':..., 'theta2':...}
        rtol, atol: solver tolerances
    Returns:
        X: array of shape (len(t_grid), 2) with states at t_grid points
    """

    def fun(t, x):  # bind params
        return rhs(t, x, params)

    sol = solve_ivp(
        fun, (t_grid[0], t_grid[-1]), x0, t_eval=t_grid, rtol=rtol, atol=atol
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.y.T


def add_observation_noise(
    x: np.ndarray, observe: str, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate noisy observations from the trajectory.
    Args:
        x: states (N,2)
        observe: 'x1' or 'x2'
        noise_std: standard deviation of Gaussian noise
        rng: np.random.Generator
    Returns:
        y: 1D array of length N (noisy observations)
    """
    idx = 0 if observe == "x1" else 1
    return x[:, idx] + rng.normal(0.0, noise_std, size=x.shape[0])


def integrate_with_params(
    theta: np.ndarray, t_grid: np.ndarray, x0: np.ndarray
) -> Callable[[Dict[str, float]], np.ndarray]:
    """
    Convenience wrapper that returns simulator bound to theta vector.
    Args:
        theta: array-like [theta1, theta2]
        t_grid: time grid
        x0: initial condition
    Returns:
        function(params_dict) -> states
    """
    params = {"theta1": float(theta[0]), "theta2": float(theta[1])}

    def _simulate(_: Dict[str, float] = params) -> np.ndarray:
        return simulate_trajectory(t_grid, x0, params)

    return _simulate
