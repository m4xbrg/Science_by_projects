
# function_explorer_m3.py
from typing import Dict, List, Tuple, Optional
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

x = sp.Symbol("x", real=True)

def parse_expr(expr_str: str) -> sp.Expr:
    return sp.sympify(expr_str, locals={
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
        "sqrt": sp.sqrt, "abs": sp.Abs
    })

def domain_intervals(expr: sp.Expr) -> List[Tuple[Optional[float], Optional[float]]]:
    '''Return real-line domain intervals as (a,b) tuples where a/b may be None for infinities.'''
    dom = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
    intervals = []
    if isinstance(dom, sp.Interval):
        a = None if dom.start is sp.S.NegativeInfinity else float(sp.N(dom.start))
        b = None if dom.end   is sp.S.Infinity        else float(sp.N(dom.end))
        if dom.left_open or dom.right_open:
            # We keep openness info conceptually; plotting won't show closed vs open
            intervals.append((a, b))
        else:
            intervals.append((a, b))
    elif isinstance(dom, sp.Union):
        for part in dom.args:
            if isinstance(part, sp.Interval):
                a = None if part.start is sp.S.NegativeInfinity else float(sp.N(part.start))
                b = None if part.end   is sp.S.Infinity        else float(sp.N(part.end))
                intervals.append((a, b))
    elif dom == sp.S.Reals:
        intervals.append((None, None))
    return sorted(intervals, key=lambda t: (float('-inf') if t[0] is None else t[0]))

def approx_range(expr: sp.Expr, x_min: float, x_max: float, samples: int=2000) -> Tuple[float, float]:
    '''Approximate range by sampling over [x_min, x_max]'''
    f = sp.lambdify(x, expr, modules=["numpy"])
    xs = np.linspace(x_min, x_max, samples)
    ys = f(xs)
    ys = np.array(ys, dtype=np.complex128)
    real = np.isfinite(ys.real) & (np.abs(ys.imag) < 1e-12)
    y_real = np.where(real, ys.real, np.nan)
    # ignore NaNs
    y_finite = y_real[np.isfinite(y_real)]
    if y_finite.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(y_finite)), float(np.nanmax(y_finite))

def plot_with_domain(expr_str: str, x_min: float=-10.0, x_max: float=10.0, samples: int=2000) -> Dict[str, object]:
    expr = parse_expr(expr_str)
    f = sp.lambdify(x, expr, modules=["numpy"])
    xs = np.linspace(x_min, x_max, samples)
    ys = f(xs)
    ys = np.array(ys, dtype=np.complex128)
    real = np.isfinite(ys.real) & (np.abs(ys.imag) < 1e-12)
    y_real = np.where(real, ys.real, np.nan)

    # Plot
    plt.figure()
    plt.plot(xs, y_real, linewidth=2)
    plt.xlabel("x"); plt.ylabel("f(x)")
    plt.title(f"f(x) = {sp.sstr(expr)}")
    plt.grid(True, alpha=0.3)

    # Domain intervals (draw as light background spans within [x_min,x_max])
    dom_ints = domain_intervals(expr)
    clipped = []
    for (a, b) in dom_ints:
        aa = x_min if a is None else max(a, x_min)
        bb = x_max if b is None else min(b, x_max)
        if bb > aa:
            plt.axvspan(aa, bb, alpha=0.06)
            clipped.append((aa, bb))

    # Approx range
    y_min, y_max = approx_range(expr, x_min, x_max, samples)
    if np.isfinite(y_min) and np.isfinite(y_max):
        plt.axhline(y_min, linestyle=":", linewidth=1.2, alpha=0.7)
        plt.axhline(y_max, linestyle=":", linewidth=1.2, alpha=0.7)

    plt.show()
    return {
        "expr_sympy": sp.sstr(expr),
        "domain_intervals": dom_ints,
        "domain_intervals_clipped": clipped,
        "range_approx": (y_min, y_max),
        "plot_window": (x_min, x_max),
    }

def widget(expr_init: str="sin(x)/x", x_min_init: float=-10.0, x_max_init: float=10.0, samples_init: int=1200):
    '''Interactive widget: sliders for x_min/x_max/samples and textbox for expression.'''
    expr = parse_expr(expr_init)
    f = sp.lambdify(x, expr, modules=["numpy"])

    # Base figure and main axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.12, bottom=0.28)  # leave room for widgets

    # Initial data
    xs = np.linspace(x_min_init, x_max_init, samples_init)
    ys = np.array(f(xs), dtype=np.complex128)
    real = np.isfinite(ys.real) & (np.abs(ys.imag) < 1e-12)
    y_real = np.where(real, ys.real, np.nan)
    [line] = ax.plot(xs, y_real, linewidth=2)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title(f"f(x) = {sp.sstr(expr)}")
    ax.grid(True, alpha=0.3)

    # Draw initial domain spans and range hints
    dom_ints = domain_intervals(expr)
    span_patches = []
    for (a, b) in dom_ints:
        aa = x_min_init if a is None else max(a, x_min_init)
        bb = x_max_init if b is None else min(b, x_max_init)
        if bb > aa:
            p = ax.axvspan(aa, bb, alpha=0.06)
            span_patches.append(p)
    y_min, y_max = approx_range(expr, x_min_init, x_max_init, samples_init)
    hmin = ax.axhline(y_min, linestyle=":", linewidth=1.2, alpha=0.7) if np.isfinite(y_min) else None
    hmax = ax.axhline(y_max, linestyle=":", linewidth=1.2, alpha=0.7) if np.isfinite(y_max) else None

    # Widgets: Sliders for x_min, x_max, samples; TextBox for expression; Button to reset
    ax_xmin = plt.axes([0.12, 0.19, 0.35, 0.03])
    ax_xmax = plt.axes([0.12, 0.14, 0.35, 0.03])
    ax_samples = plt.axes([0.12, 0.09, 0.35, 0.03])
    ax_expr = plt.axes([0.53, 0.14, 0.35, 0.08])
    ax_reset = plt.axes([0.53, 0.09, 0.12, 0.035])

    s_xmin = Slider(ax=ax_xmin, label="x_min", valmin=-100.0, valmax=0.0, valinit=x_min_init, valstep=0.1)
    s_xmax = Slider(ax=ax_xmax, label="x_max", valmin=0.0, valmax=100.0, valinit=x_max_init, valstep=0.1)
    s_samples = Slider(ax=ax_samples, label="samples", valmin=200, valmax=10000, valinit=samples_init, valstep=100)
    t_expr = TextBox(ax=ax_expr, label="f(x) = ", initial=expr_init)
    b_reset = Button(ax=ax_reset, label="Reset")

    def recompute(_=None):
        # parse expression string
        expr_str = t_expr.text
        try:
            expr_local = parse_expr(expr_str)
        except Exception:
            return  # invalid expression: ignore
        f_local = sp.lambdify(x, expr_local, modules=["numpy"])

        xmin = float(s_xmin.val); xmax = float(s_xmax.val)
        if xmax <= xmin:
            return
        samples = int(s_samples.val)

        xs = np.linspace(xmin, xmax, samples)
        ys = np.array(f_local(xs), dtype=np.complex128)
        real = np.isfinite(ys.real) & (np.abs(ys.imag) < 1e-12)
        y_real = np.where(real, ys.real, np.nan)

        # update line
        line.set_data(xs, y_real)
        ax.set_xlim(xmin, xmax)
        # rescale y to visible finite values if any
        if np.isfinite(y_real).any():
            y_finite = y_real[np.isfinite(y_real)]
            if y_finite.size > 0:
                yr = float(np.nanmax(np.abs(y_finite)))
                if np.isfinite(yr) and yr > 0:
                    ax.set_ylim(-1.05*yr, 1.05*yr)

        ax.set_title(f"f(x) = {sp.sstr(expr_local)}")

        # update domain spans
        nonlocal span_patches, hmin, hmax
        for p in span_patches:
            p.remove()
        span_patches = []
        for (a, b) in domain_intervals(expr_local):
            aa = xmin if a is None else max(a, xmin)
            bb = xmax if b is None else min(b, xmax)
            if bb > aa:
                p = ax.axvspan(aa, bb, alpha=0.06)
                span_patches.append(p)

        # update range hints
        ymin, ymax = approx_range(expr_local, xmin, xmax, samples)
        if hmin is not None:
            hmin.remove()
            hmin = None
        if hmax is not None:
            hmax.remove()
            hmax = None
        if np.isfinite(ymin):
            hmin = ax.axhline(ymin, linestyle=":", linewidth=1.2, alpha=0.7)
        if np.isfinite(ymax):
            hmax = ax.axhline(ymax, linestyle=":", linewidth=1.2, alpha=0.7)

        fig.canvas.draw_idle()

    # Wire callbacks
    s_xmin.on_changed(recompute)
    s_xmax.on_changed(recompute)
    s_samples.on_changed(recompute)
    t_expr.on_submit(lambda _: recompute())
    b_reset.on_clicked(lambda _: (
        s_xmin.reset(), s_xmax.reset(), s_samples.reset(), t_expr.set_val(expr_init), recompute()
    ))

    recompute()
    plt.show()
