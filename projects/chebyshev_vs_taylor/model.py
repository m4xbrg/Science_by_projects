"""
Core approximation routines for Chebyshev vs Taylor on [a,b].
Public API highlights:
- to_cheb_domain, from_cheb_domain
- chebyshev_fit
- taylor_coeffs_fd
- eval_taylor
- compare_approximations

All functions are deterministic given (f, a, b, n, c, ...).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List
import numpy as np
from numpy.polynomial import Chebyshev as Cheb


# ----------------------------
# Utilities: domain mapping
# ----------------------------
def to_cheb_domain(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Map x in [a,b] to t in [-1,1]."""
    return (2.0 * x - (a + b)) / (b - a)


def from_cheb_domain(a: float, b: float, t: np.ndarray) -> np.ndarray:
    """Map t in [-1,1] to x in [a,b]."""
    return ((b - a) * t + (a + b)) / 2.0


# ----------------------------
# Finite differences for derivatives at a point
# (central differences + Richardson extrapolation)
# ----------------------------
def _derivative_central(
    f: Callable[[float], float], c: float, order: int, h: float
) -> float:
    """
    Approximate the order-th derivative of f at c using 7-point central stencil.
    For order in {1,2,3,4,5,6}. Raises NotImplementedError for higher order.
    """
    if order == 0:
        return f(c)
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float) * h
    fx = np.array([f(c + xi) for xi in x])
    if order == 1:
        coeff = np.array([1 / 60, -3 / 20, 3 / 4, 0, -3 / 4, 3 / 20, -1 / 60])
        return float(np.dot(coeff, fx) / h)
    elif order == 2:
        coeff = np.array([-1 / 90, 3 / 20, -3 / 2, 49 / 18, -3 / 2, 3 / 20, -1 / 90])
        return float(np.dot(coeff, fx) / (h**2))
    elif order == 3:
        coeff = np.array([-1 / 8, 1, -13 / 8, 0, 13 / 8, -1, 1 / 8])
        return float(np.dot(coeff, fx) / (h**3))
    elif order == 4:
        coeff = np.array(
            [7 / 240, -3 / 10, 169 / 60, -122 / 15, 169 / 60, -3 / 10, 7 / 240]
        )
        return float(np.dot(coeff, fx) / (h**4))
    elif order == 5:
        coeff = np.array([-1 / 6, 2, -13 / 2, 0, 13 / 2, -2, 1 / 6])
        return float(np.dot(coeff, fx) / (h**5))
    elif order == 6:
        coeff = np.array([-1 / 90, 3 / 10, -3, 49 / 6, -3, 3 / 10, -1 / 90])
        return float(np.dot(coeff, fx) / (h**6))
    else:
        raise NotImplementedError("Order > 6 not implemented in this stencil.")


def _derivative_fd_richardson(
    f: Callable[[float], float], c: float, k: int, h0: float = 1e-2, levels: int = 3
) -> float:
    """
    Richardson extrapolation on central-difference estimate to reduce truncation error.
    """
    hs = [h0 / (2**j) for j in range(levels)]
    est = [_derivative_central(f, c, k, h) for h in hs]
    p = 2.0  # assume leading error ~ h^2
    for m in range(1, levels):
        for j in range(levels - 1, m - 1, -1):
            factor = 2 ** (p * m)
            est[j] = (factor * est[j] - est[j - 1]) / (factor - 1.0)
    return float(est[levels - 1])


def taylor_coeffs_fd(
    f_scalar: Callable[[float], float],
    c: float,
    n: int,
    h0: float = 1e-2,
    richardson_levels: int = 3,
) -> np.ndarray:
    """
    Compute Taylor coefficients up to degree n about c using FD+Richardson.
    Returns array coeff[k] = f^{(k)}(c)/k!.
    """
    coeff = np.zeros(n + 1, dtype=float)
    coeff[0] = f_scalar(c)
    for k in range(1, n + 1):
        dk = _derivative_fd_richardson(f_scalar, c, k, h0=h0, levels=richardson_levels)
        coeff[k] = dk / np.math.factorial(k)
    return coeff


# ----------------------------
# Chebyshev fitting
# ----------------------------
def chebyshev_fit(
    f_vec: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    N_fit: int | None = None,
) -> Cheb:
    """
    Fit a Chebyshev polynomial of degree n to f on [a,b] using numpy's Chebyshev.fit.
    Returns a Chebyshev object with domain [a,b].
    """
    if N_fit is None:
        N_fit = max(2 * n + 10, 64)
    xs = np.cos((np.pi * (np.arange(N_fit) + 0.5)) / N_fit)  # Chebyshev nodes in [-1,1]
    x_fit = from_cheb_domain(a, b, xs)
    y_fit = f_vec(x_fit)
    cheb = Cheb.fit(x_fit, y_fit, deg=n, domain=[a, b])
    return cheb


# ----------------------------
# Evaluation
# ----------------------------
def eval_taylor(coeff: np.ndarray, c: float, x: np.ndarray) -> np.ndarray:
    """
    Evaluate Taylor series with coefficients around c via Horner about (x-c).
    coeff[k] = f^{(k)}(c)/k!.
    """
    p = np.zeros_like(x, dtype=float)
    for k in range(len(coeff) - 1, -1, -1):
        p = coeff[k] + (x - c) * p
    return p


@dataclass
class ApproxConfig:
    a: float
    b: float
    n: int
    c: float
    N_eval: int = 2001
    h0: float = 1e-2
    richardson_levels: int = 3
    N_fit: int | None = None


def compare_approximations(
    f_vec: Callable[[np.ndarray], np.ndarray], cfg: ApproxConfig
) -> Dict[str, Any]:
    """
    Compute Chebyshev and Taylor approximations and error diagnostics on a dense grid.
    Returns a dict with x, y, p_cheb, p_taylor, err_cheb, err_taylor, Emax_cheb, Emax_taylor.
    """
    cheb = chebyshev_fit(f_vec, cfg.a, cfg.b, cfg.n, N_fit=cfg.N_fit)
    # scalar wrapper for derivatives
    def f_scalar(x):
    return float(f_vec(np.array([x]))[0])
    coeff_taylor = taylor_coeffs_fd(
        f_scalar, cfg.c, cfg.n, h0=cfg.h0, richardson_levels=cfg.richardson_levels
    )

    x = np.linspace(cfg.a, cfg.b, cfg.N_eval)
    y = f_vec(x)
    p_cheb = cheb(x)
    p_taylor = eval_taylor(coeff_taylor, cfg.c, x)
    err_cheb = np.abs(y - p_cheb)
    err_taylor = np.abs(y - p_taylor)

    return {
        "x": x,
        "y": y,
        "p_cheb": p_cheb,
        "p_taylor": p_taylor,
        "err_cheb": err_cheb,
        "err_taylor": err_taylor,
        "Emax_cheb": float(np.max(err_cheb)),
        "Emax_taylor": float(np.max(err_taylor)),
        "coeff_taylor": coeff_taylor,
        "cheb_coeffs": cheb.coef,
    }


def sweep_degree(
    f_vec: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    c: float,
    n_list: List[int],
    N_eval: int = 2001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute max errors for Chebyshev and Taylor across degrees in n_list.
    """
    Echeb, Etaylor = [], []
    for n in n_list:
        cfg = ApproxConfig(a=a, b=b, n=n, c=c, N_eval=N_eval)
        res = compare_approximations(f_vec, cfg)
        Echeb.append(res["Emax_cheb"])
        Etaylor.append(res["Emax_taylor"])
    return np.array(Echeb), np.array(Etaylor)
