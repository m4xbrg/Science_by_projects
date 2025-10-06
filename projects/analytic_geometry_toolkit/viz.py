import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import Line, Circle
from model import DEFAULT_EPS
from model import point_distance
from matplotlib.patches import Circle as CirclePatch
from matplotlib.lines import Line2D


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_scene(scene_cfg, epsilon: float, normalize_lines: bool):
    points = [tuple(p) for p in scene_cfg.get("points", [])]
    lines = []
    for L in scene_cfg.get("lines", []):
        if "from_points" in L:
            p1, p2 = map(tuple, L["from_points"])
            lines.append(
                Line.from_points(p1, p2, normalize=normalize_lines, eps=epsilon)
            )
        elif "coeffs" in L:
            a, b, c = L["coeffs"]
            lines.append(
                Line.from_coeffs(a, b, c, normalize=normalize_lines, eps=epsilon)
            )
    circles = []
    for C in scene_cfg.get("circles", []):
        (x0, y0) = tuple(C["center"])
        r = float(C["radius"])
        circles.append(Circle(x0, y0, r))
    return points, lines, circles


def plot_scene(points, lines, circles, viewport=None, out_path="figs/scene.png"):
    fig, ax = plt.subplots()

    # Points
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, marker="o", label="Points")
        for i, (x, y) in enumerate(points):
            ax.annotate(f"P{i}", (x, y), xytext=(5, 5), textcoords="offset points")

    # Circles
    for i, c in enumerate(circles):
        patch = CirclePatch((c.x0, c.y0), c.r, fill=False)
        ax.add_patch(patch)
        ax.annotate(f"C{i}", (c.x0, c.y0), xytext=(5, 5), textcoords="offset points")

    # Viewport
    if viewport is None:
        xs = (
            [x for x, y in points]
            + [c.x0 - c.r for c in circles]
            + [c.x0 + c.r for c in circles]
        )
        ys = (
            [y for x, y in points]
            + [c.y0 - c.r for c in circles]
            + [c.y0 + c.r for c in circles]
        )
        if xs and ys:
            margin = 0.1 * max((max(xs) - min(xs) + max(ys) - min(ys)), 1.0)
            viewport = (
                min(xs) - margin,
                max(xs) + margin,
                min(ys) - margin,
                max(ys) + margin,
            )
        else:
            viewport = (-5, 5, -5, 5)
    xmin, xmax, ymin, ymax = viewport
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Lines
    for i, L in enumerate(lines):
        candidates = []
        if abs(L.b) > DEFAULT_EPS:
            for xv in (xmin, xmax):
                yv = (-L.a * xv - L.c) / L.b
                if ymin - 1e-9 <= yv <= ymax + 1e-9:
                    candidates.append((xv, yv))
        if abs(L.a) > DEFAULT_EPS:
            for yv in (ymin, ymax):
                xv = (-L.b * yv - L.c) / L.a
                if xmin - 1e-9 <= xv <= xmax + 1e-9:
                    candidates.append((xv, yv))
        if len(candidates) >= 2:
            candidates = sorted(candidates)
            p_start, p_end = candidates[0], candidates[-1]
            ax.add_line(
                Line2D(
                    [p_start[0], p_end[0]],
                    [p_start[1], p_end[1]],
                    label="Line" if i == 0 else None,
                )
            )
        ax.annotate(
            f"L{i}",
            ((xmin + xmax) / 2, (ymin + ymax) / 2),
            xytext=(0, 0),
            textcoords="offset points",
        )

    # Pairwise intersections (simple marker only)
    inter_pts = []
    lines_list = list(lines)
    for i in range(len(lines_list)):
        for j in range(i + 1, len(lines_list)):
            pts, rel = lines_list[i].intersect_line(lines_list[j])
            inter_pts.extend(pts)
    for i, L in enumerate(lines_list):
        for j, C in enumerate(circles):
            pts, rel = C.intersect_line(L)
            inter_pts.extend(pts)
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            pts, rel = circles[i].intersect_circle(circles[j])
            inter_pts.extend(pts)
    if inter_pts:
        xi, yi = zip(*inter_pts)
        ax.scatter(xi, yi, marker="x", label="Intersections")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_matrix(points, out_path="figs/distance_matrix.png"):
    import matplotlib.pyplot as plt

    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = point_distance(points[i], points[j])
    fig, ax = plt.subplots()
    im = ax.imshow(D, origin="lower")
    ax.set_title("Point–Point Distance Matrix")
    ax.set_xlabel("Index")
    ax.set_ylabel("Index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    epsilon = float(cfg.get("epsilon", DEFAULT_EPS))
    normalize_lines = bool(cfg.get("normalize_lines", True))
    points, lines, circles = build_scene(cfg["scene"], epsilon, normalize_lines)

    fig_dir = Path(cfg["output"]["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_scene(points, lines, circles, out_path=fig_dir / cfg["output"]["fig_scene"])
    plot_distance_matrix(
        points, out_path=fig_dir / cfg["output"]["fig_distance_matrix"]
    )
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()


# --- AUTO-ADDED STUBS: uniform visualization entrypoints ---
def plot_primary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # simple line of first numeric column or index
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c
                break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index")
    plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_secondary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # histogram on first numeric column
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c
                break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    try:
        plt.hist(df[col], bins=20)
    except Exception:
        plt.plot(range(len(df[col])), df[col])
    plt.title("Secondary Plot (stub)")
    plt.xlabel(str(col))
    plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out
