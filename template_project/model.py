"""
Model equations -> code mapping.
Implement your core model here.

Expose a clean API so simulate.py can call:
    state_dot = rhs(t, state, params)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any


@dataclass
class ModelParams:
    theta1: float = 1.0
    theta2: float = 0.5


def rhs(t: float, x: np.ndarray, p: ModelParams) -> np.ndarray:
    """
    Right-hand side of the ODE system: dx/dt = f(t, x; p)

    Args:
        t: time
        x: state vector (n,)
        p: model parameters

    Returns:
        dxdt: derivative (n,)
    """
    # Example 2D linear system (replace with your equations)
    a, b = p.theta1, p.theta2
    dx1 = a * x[0] - b * x[1]
    dx2 = b * x[0] - a * x[1]
    return np.array([dx1, dx2], dtype=float)
