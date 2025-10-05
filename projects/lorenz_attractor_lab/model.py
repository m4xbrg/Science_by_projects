from __future__ import annotations
from typing import Dict
import numpy as np

def rhs(t: float, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Lorenz RHS: x' = [ sigma(y-x), x(rho - z) - y, xy - beta z ]"""
    sigma = params["sigma"]; rho = params["rho"]; beta = params["beta"]
    X, Y, Z = x
    return np.array([sigma*(Y - X), X*(rho - Z) - Y, X*Y - beta*Z], float)

def jacobian(t: float, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Analytic Jacobian of the Lorenz flow."""
    sigma = params["sigma"]; rho = params["rho"]; beta = params["beta"]
    X, Y, Z = x
    return np.array([[-sigma, sigma, 0.0], [rho - Z, -1.0, -X], [Y, X, -beta]], float)

def _plane_eval(x: np.ndarray, plane: str) -> float:
    if plane.startswith("x="): return x[0] - float(plane.split("=")[1])
    if plane.startswith("y="): return x[1] - float(plane.split("=")[1])
    if plane.startswith("z="): return x[2] - float(plane.split("=")[1])
    raise ValueError("Plane must be 'x=c', 'y=c', or 'z=c'.")

def poincare_crossings(ts, xs, plane="x=0", direction="increasing"):
    """Return points where trajectory crosses the plane (with direction filtering)."""
    vals = np.apply_along_axis(lambda u: _plane_eval(u, plane), 1, xs)
    hits = []
    for i in range(len(ts)-1):
        a, b = vals[i], vals[i+1]
        if direction == "increasing" and not (a < 0 <= b): continue
        if direction == "decreasing" and not (a > 0 >= b): continue
        if a == b: lam = 0.0
        else: lam = -a/(b - a)
        if 0.0 <= lam <= 1.0:
            hits.append(xs[i] + lam*(xs[i+1]-xs[i]))
    return np.array(hits)
