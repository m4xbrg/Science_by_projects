# function_explorer.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import sympy as sp


@dataclass
class DomainInterval:
    a: float
    b: float

    def __repr__(self):
        return f"[{self.a}, {self.b}]"


@dataclass
class AnalysisResult:
    expr_str: str
    sym_x: sp.Symbol
    sym_f: sp.Expr
    domain_intervals: List[DomainInterval]
    singularities: List[float]
    x_intercepts: List[float]
    y_intercept: Optional[float]
    critical_points: List[float]
    range_estimate: Tuple[Optional[float], Optional[float], str]
    samples: Dict[str, Any]


def _parse_expr(expr_str: str):
    x = sp.symbols("x", real=True)
    allowed = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    f = sp.sympify(expr_str, locals=allowed)
    f = sp.simplify(f)
    return x, f


def _denominators(expr: sp.Expr):
    dens = []
    for term in sp.preorder_traversal(expr):
        if isinstance(term, sp.Pow) and term.exp.is_Number and term.exp < 0:
            dens.append(term.base)
    num, den = sp.fraction(sp.together(expr))
    if den != 1:
        dens.append(den)
    return list({sp.simplify(d) for d in dens})


def _log_args(expr: sp.Expr):
    args = []
    for term in sp.preorder_traversal(expr):
        if isinstance(term, sp.log):
            args.append(sp.simplify(term.args[0]))
    return args


def _even_root_args(expr: sp.Expr):
    args = []
    for term in sp.preorder_traversal(expr):
        if isinstance(term, sp.Pow) and term.exp.is_Rational and (term.exp.q % 2 == 0):
            args.append(sp.simplify(term.base))
    return args


def _solve_real_eq(eq: sp.Eq, x: sp.Symbol):
    sol = sp.solveset(eq, x, domain=sp.S.Reals)
    out = []
    if isinstance(sol, sp.Interval):
        return out
    try:
        for s in sol:
            if getattr(s, "is_real", False):
                out.append(float(sp.N(s)))
    except TypeError:
        pass
    return sorted(set(out))


def _solve_ineq_ge0(expr: sp.Expr, x: sp.Symbol):
    try:
        return sp.solveset(expr >= 0, x, domain=sp.S.Reals)
    except Exception:
        return sp.S.Reals


def _solve_ineq_gt0(expr: sp.Expr, x: sp.Symbol):
    try:
        return sp.solveset(expr > 0, x, domain=sp.S.Reals)
    except Exception:
        return sp.S.Reals


def _to_intervals(sset: sp.Set):
    intervals = []
    if sset == sp.S.Reals:
        return [DomainInterval(float(-np.inf), float(np.inf))]
    if isinstance(sset, sp.Interval):
        intervals.append(DomainInterval(float(sset.start), float(sset.end)))
    elif isinstance(sset, sp.Union):
        for part in sset.args:
            if isinstance(part, sp.Interval):
                intervals.append(DomainInterval(float(part.start), float(part.end)))
    elif isinstance(sset, sp.FiniteSet):
        pass
    return intervals


def _intersect_with_window(intervals, window):
    (a, b) = window
    out = []
    for I in intervals:
        aa = max(I.a, a)
        bb = min(I.b, b)
        if aa < bb:
            out.append(DomainInterval(aa, bb))
    return out


class FunctionExplorer:
    def __init__(self, expr_str: str):
        self.expr_str = expr_str
        self.x, self.f = _parse_expr(expr_str)

    def analyze(self, window=(-10.0, 10.0), samples=2000, tol=1e-9) -> AnalysisResult:
        x, f = self.x, self.f
        dens = _denominators(f)
        logs = _log_args(f)
        evens = _even_root_args(f)

        domain_set = sp.S.Reals
        for d in dens:
            zeros = _solve_real_eq(sp.Eq(d, 0), x)
            for z in zeros:
                domain_set = domain_set - sp.FiniteSet(z)
        for a in logs:
            domain_set = domain_set.intersect(_solve_ineq_gt0(a, x))
        for a in evens:
            domain_set = domain_set.intersect(_solve_ineq_ge0(a, x))

        singularities = []
        for d in dens:
            singularities.extend(_solve_real_eq(sp.Eq(d, 0), x))
        singularities = sorted(set(singularities))

        x_intercepts = _solve_real_eq(sp.Eq(f, 0), x)

        try:
            y_intercept = float(sp.N(f.subs(x, 0))) if 0 not in singularities else None
        except Exception:
            y_intercept = None

        try:
            fp = sp.diff(f, x)
            cps_all = _solve_real_eq(sp.Eq(fp, 0), x)
            cps = [c for c in cps_all if window[0] <= c <= window[1]]
        except Exception:
            cps = []

        domain_intervals = _to_intervals(domain_set)
        domain_intervals_window = _intersect_with_window(domain_intervals, window)

        Xs, Ys, segments = [], [], []
        if domain_intervals_window and samples > 0:
            total_len = sum(
                I.b - I.a
                for I in domain_intervals_window
                if np.isfinite(I.a) and np.isfinite(I.b)
            )
            for I in domain_intervals_window:
                length = I.b - I.a
                nI = (
                    max(10, int(samples * (length / max(total_len, 1e-9))))
                    if np.isfinite(length)
                    else max(10, samples // len(domain_intervals_window))
                )
                xi = np.linspace(I.a, I.b, nI)
                f_lmbd = sp.lambdify(x, f, modules=["numpy"])
                with np.errstate(all="ignore"):
                    yi = f_lmbd(xi).astype(float)
                yi[~np.isfinite(yi)] = np.nan
                start_idx = sum(len(arr) for arr in Xs)
                Xs.append(xi)
                Ys.append(yi)
                segments.append(slice(start_idx, start_idx + len(xi)))

        import numpy as _np

        X = _np.concatenate(Xs) if Xs else _np.array([])
        Y = _np.concatenate(Ys) if Ys else _np.array([])

        if len(Y) > 0 and _np.isfinite(Y).any():
            finiteY = Y[_np.isfinite(Y)]
            ymin, ymax = float(_np.nanmin(finiteY)), float(_np.nanmax(finiteY))
            note = "Range estimated from samples within window."
        else:
            ymin = ymax = None
            note = "No finite samples; range unavailable in given window."

        return AnalysisResult(
            expr_str=self.expr_str,
            sym_x=x,
            sym_f=f,
            domain_intervals=domain_intervals,
            singularities=singularities,
            x_intercepts=x_intercepts,
            y_intercept=y_intercept,
            critical_points=cps,
            range_estimate=(ymin, ymax, note),
            samples={"X": X, "Y": Y, "segments": segments},
        )

    @staticmethod
    def format_domain(intervals: List[DomainInterval]) -> str:
        import numpy as _np

        if not intervals:
            return "∅"
        parts = []
        for I in intervals:
            a = "-∞" if _np.isneginf(I.a) else f"{I.a:g}"
            b = "∞" if _np.isposinf(I.b) else f"{I.b:g}"
            parts.append(f"({a}, {b})")
        return " ∪ ".join(parts)
