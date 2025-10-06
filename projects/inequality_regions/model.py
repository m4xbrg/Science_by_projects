import itertools
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def normalize_ineq(a: Tuple[float, float], b: float, sense: str):
    """Convert inequality to canonical <= form. sense in {'<=','>='}."""
    ax = np.array(a, dtype=float)
    if sense.strip() == "<=":
        return ax, float(b)
    elif sense.strip() == ">=":
        return -ax, float(-b)
    raise ValueError("sense must be '<=' or '>='")


def line_intersection(a1, b1, a2, b2, eps: float = 1e-12):
    """Intersection of a1*x=b1 and a2*x=b2 for x=(x,y). Returns None if parallel."""
    A = np.vstack([a1, a2])
    det = np.linalg.det(A)
    if abs(det) < eps:
        return None
    return np.linalg.solve(A, np.array([b1, b2]))


def satisfies_all(xy, A, b, tol: float = 1e-9) -> bool:
    """Check A @ xy <= b + tol componentwise."""
    return np.all(A @ xy <= b + tol)


def unique_points(points, tol: float = 1e-8):
    """Deduplicate by Euclidean distance tolerance."""
    uniq = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= tol for q in uniq):
            uniq.append(p)
    return uniq


def monotone_chain_convex_hull(points, tol: float = 1e-12):
    """Andrew's monotone chain. Returns vertices CCW without repeating first point."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= tol:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= tol:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def feasible_region_2d(inequalities: List[Dict], add_bbox=None, tol: float = 1e-9):
    """Compute feasible polygon (if bounded) from linear inequalities in 2D.
    Returns dict with A, b, vertices, raw_points, bounded.
    """
    cons = [normalize_ineq(tuple(c["a"]), c["b"], c["sense"]) for c in inequalities]

    if add_bbox is not None:
        xmin, xmax, ymin, ymax = add_bbox
        cons += [normalize_ineq((1, 0), xmax, "<=")]
        cons += [normalize_ineq((1, 0), xmin, ">=")]
        cons += [normalize_ineq((0, 1), ymax, "<=")]
        cons += [normalize_ineq((0, 1), ymin, ">=")]

    A = np.vstack([c[0] for c in cons])
    b = np.array([c[1] for c in cons])

    candidates = []
    for i, j in itertools.combinations(range(len(cons)), 2):
        p = line_intersection(cons[i][0], cons[i][1], cons[j][0], cons[j][1])
        if p is not None and np.all(np.isfinite(p)) and satisfies_all(p, A, b, tol=tol):
            candidates.append(p)

    candidates = unique_points(candidates, tol=1e-8)
    pts = [(float(p[0]), float(p[1])) for p in candidates]

    if len(pts) == 0:
        hull = []
        bounded = False
    elif len(pts) <= 2:
        hull = pts
        bounded = len(pts) > 0
    else:
        hull = monotone_chain_convex_hull(pts)
        if len(hull) >= 3:
            x = np.array([p[0] for p in hull])
            y = np.array([p[1] for p in hull])
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            bounded = area > 1e-12
        else:
            bounded = False

    return {"A": A, "b": b, "vertices": hull, "raw_points": pts, "bounded": bounded}


def plot_feasible_region(
    result, inequalities, view, title="Feasible Region", fname=None
):
    """Plot boundary lines, feasible polygon, and labeled vertices."""
    xmin, xmax, ymin, ymax = view
    xs = np.linspace(xmin, xmax, 500)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    for c in inequalities:
        a, b, sense = np.array(c["a"], dtype=float), float(c["b"]), c["sense"]
        if abs(a[1]) > 1e-12:
            ys = (b - a[0] * xs) / a[1]
            ax.plot(
                xs,
                ys,
                linewidth=1.5,
                label=f"{a[0]:.2g}x + {a[1]:.2g}y {sense} {b:.2g}",
            )
        else:
            xv = b / a[0] if abs(a[0]) > 1e-12 else np.nan
            ax.plot(
                [xv, xv],
                [ymin, ymax],
                linewidth=1.5,
                label=f"{a[0]:.2g}x + {a[1]:.2g}y {sense} {b:.2g}",
            )

    if result["bounded"] and len(result["vertices"]) >= 3:
        vx = [p[0] for p in result["vertices"]]
        vy = [p[1] for p in result["vertices"]]
        ax.fill(vx, vy, alpha=0.2, linewidth=0, label="Feasible polygon")

    for idx, (xv, yv) in enumerate(result["vertices"]):
        ax.plot([xv], [yv], marker="o")
        ax.text(xv, yv, f" V{idx}({xv:.3g},{yv:.3g})", fontsize=9)

    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.3)
    if fname:
        fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_feasibility_heatmap(
    inequalities, view, grid_n: int = 200, title="Feasibility Count Heatmap", fname=None
):
    """Heatmap of how many inequalities are satisfied across the viewport grid."""
    xmin, xmax, ymin, ymax = view
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    X, Y = np.meshgrid(xs, ys)
    count = np.zeros_like(X, dtype=int)
    for c in inequalities:
        a_can, b_can = normalize_ineq(tuple(c["a"]), c["b"], c["sense"])
        count += ((a_can[0] * X + a_can[1] * Y) <= (b_can + 1e-9)).astype(int)

    fig = plt.figure(figsize=(6, 5.5))
    ax = plt.gca()
    ax.imshow(count, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title + " (darker = more constraints satisfied)")
    ax.grid(False)
    if fname:
        fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
