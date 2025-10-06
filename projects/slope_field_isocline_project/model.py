"""
Model utilities for user-defined scalar ODE: x'(t) = f(t, x; params).
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple
import math
import numpy as np

_SAFE_NS = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": math.pi,
    "e": math.e,
}


def parse_rhs(expr: str) -> Callable[[float, float, Dict[str, float]], float]:
    code = compile(expr, "<rhs_expr>", "eval")

    def f(t: float, x: float, params: Dict[str, float]) -> float:
        ns = dict(_SAFE_NS)
        ns["t"] = t
        ns["x"] = x
        ns.update(params or {})
        return float(eval(code, {"__builtins__": {}}, ns))

    return f


def euler_step(f: Callable, t: float, x: float, h: float, params: Dict):
    return t + h, x + h * f(t, x, params)


def rk4_step(f: Callable, t: float, x: float, h: float, params: Dict):
    k1 = f(t, x, params)
    k2 = f(t + 0.5 * h, x + 0.5 * h * k1, params)
    k3 = f(t + 0.5 * h, x + 0.5 * h * k2, params)
    k4 = f(t + h, x + h * k3, params)
    return t + h, x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    f: Callable,
    t0: float,
    x0: float,
    t_final: float,
    h: float,
    method: str,
    params: Dict,
):
    step = euler_step if method.lower() == "euler" else rk4_step
    import numpy as np

    n = int(np.ceil((t_final - t0) / h))
    t = t0
    x = x0
    t_vec = np.empty(n + 1)
    t_vec[0] = t0
    x_vec = np.empty(n + 1)
    x_vec[0] = x0
    for i in range(1, n + 1):
        t, x = step(f, t, x, h, params)
        t_vec[i] = t
        x_vec[i] = x
    return t_vec, x_vec


def integrate_reference(f, t0, x0, t_final, h_ref, params):
    return integrate(f, t0, x0, t_final, h_ref, "rk4", params)
