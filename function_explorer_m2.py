
# function_explorer_m2.py
from typing import Callable, Tuple, List, Dict, Optional
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol("x", real=True)

def _parse(expr_str: str) -> sp.Expr:
    expr = sp.sympify(expr_str, locals={
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
        "sqrt": sp.sqrt, "abs": sp.Abs
    })
    return expr

def _realify_solutions(solutions) -> List[float]:
    reals = []
    try:
        iterable = list(solutions)
    except TypeError:
        iterable = [solutions]
    for s in iterable:
        s = sp.N(s)
        if s.is_real:
            try:
                reals.append(float(s))
            except TypeError:
                pass
    return sorted(set(reals))

def _lambdify(expr: sp.Expr):
    return sp.lambdify(x, expr, modules=["numpy"])

def analyze_expression(expr_str: str) -> Dict[str, object]:
    expr = _parse(expr_str)
    den = sp.denom(expr)
    disc_candidates = []
    if den != 1:
        try:
            sol_den = sp.solveset(sp.Eq(den, 0), x, domain=sp.S.Reals)
        except Exception:
            sol_den = sp.solveset(den, 0, domain=sp.S.Reals)
        disc_candidates = _realify_solutions(sol_den)

    try:
        sol_zero = sp.solveset(sp.Eq(expr, 0), x, domain=sp.S.Reals)
    except Exception:
        sol_zero = sp.solveset(expr, 0, domain=sp.S.Reals)
    zeros = _realify_solutions(sol_zero)

    y_intercept = None
    try:
        val0 = sp.simplify(expr.subs(x, 0))
        if val0.is_real and val0.is_finite:
            y_intercept = float(sp.N(val0))
        else:
            raise Exception("non-finite value at 0")
    except Exception:
        try:
            lim0 = sp.limit(expr, x, 0)
            if lim0.is_real and lim0.is_finite:
                y_intercept = float(sp.N(lim0))
        except Exception:
            y_intercept = None

    vertical_asymptotes = []
    for a in disc_candidates:
        try:
            lim_left = sp.limit(expr, x, a, dir='-')
            lim_right = sp.limit(expr, x, a, dir='+')
            if (lim_left in (sp.oo, -sp.oo)) or (lim_right in (sp.oo, -sp.oo)):
                vertical_asymptotes.append(a)
        except Exception:
            pass

    return {
        "expr_str": expr_str,
        "expr_sympy": sp.sstr(expr),
        "denominator": sp.sstr(den),
        "discontinuities": disc_candidates,
        "zeros": zeros,
        "y_intercept": y_intercept,
        "vertical_asymptotes": sorted(set(vertical_asymptotes)),
    }

def plot_with_annotations(expr_str: str, x_min: float=-10.0, x_max: float=10.0, samples: int=2000) -> Dict[str, object]:
    assert x_max > x_min
    assert samples >= 2
    expr = _parse(expr_str)
    f_np = _lambdify(expr)

    xs = np.linspace(x_min, x_max, samples)
    ys = f_np(xs)
    ys = np.array(ys, dtype=np.complex128)
    real_mask = np.isfinite(ys.real) & (np.abs(ys.imag) < 1e-12)
    y_real = np.where(real_mask, ys.real, np.nan)

    analysis = analyze_expression(expr_str)

    plt.figure()
    plt.plot(xs, y_real, linewidth=2, label="f(x)")

    for z in analysis["zeros"]:
        if x_min <= z <= x_max:
            plt.scatter([z], [0.0], s=40, marker='o', zorder=5, label="zero" if "zero" not in plt.gca().get_legend_handles_labels()[1] else None)

    yi = analysis["y_intercept"]
    if yi is not None and x_min <= 0 <= x_max:
        plt.scatter([0.0], [yi], s=50, marker='s', zorder=5, label="y-intercept")

    for a in analysis["vertical_asymptotes"]:
        if x_min <= a <= x_max:
            plt.axvline(a, linestyle="--", linewidth=1.5, alpha=0.75, label="vertical asymptote" if "vertical asymptote" not in plt.gca().get_legend_handles_labels()[1] else None)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"f(x) = {sp.sstr(expr)}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.show()

    return analysis
