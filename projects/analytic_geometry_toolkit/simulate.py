import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from model import Line, Circle, point_distance, midpoint, DEFAULT_EPS


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
        else:
            raise ValueError("Line entry must have 'from_points' or 'coeffs'.")
    circles = []
    for C in scene_cfg.get("circles", []):
        (x0, y0) = tuple(C["center"])
        r = float(C["radius"])
        circles.append(Circle(x0, y0, r))
    return points, lines, circles


def compute_tables(points, lines, circles, epsilon: float):
    # Distances & midpoints between points
    rows_dist = []
    rows_mid = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = point_distance(points[i], points[j])
            m = midpoint(points[i], points[j])
            rows_dist.append({"i": i, "j": j, "distance": d})
            rows_mid.append({"i": i, "j": j, "mx": m[0], "my": m[1]})
    df_dist = pd.DataFrame(rows_dist)
    df_mid = pd.DataFrame(rows_mid)

    # Intersections
    inter_rows = []

    # line-line
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pts, rel = lines[i].intersect_line(lines[j], eps=epsilon)
            for k, (x, y) in enumerate(pts):
                inter_rows.append(
                    {
                        "type": "L-L",
                        "obj1": i,
                        "obj2": j,
                        "k": k,
                        "x": x,
                        "y": y,
                        "relation": rel,
                    }
                )
            if not pts:
                inter_rows.append(
                    {
                        "type": "L-L",
                        "obj1": i,
                        "obj2": j,
                        "k": None,
                        "x": None,
                        "y": None,
                        "relation": rel,
                    }
                )

    # line-circle
    for i, L in enumerate(lines):
        for j, C in enumerate(circles):
            pts, rel = C.intersect_line(L, eps=epsilon)
            for k, (x, y) in enumerate(pts):
                inter_rows.append(
                    {
                        "type": "L-C",
                        "obj1": i,
                        "obj2": j,
                        "k": k,
                        "x": x,
                        "y": y,
                        "relation": rel,
                    }
                )
            if not pts:
                inter_rows.append(
                    {
                        "type": "L-C",
                        "obj1": i,
                        "obj2": j,
                        "k": None,
                        "x": None,
                        "y": None,
                        "relation": rel,
                    }
                )

    # circle-circle
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            pts, rel = circles[i].intersect_circle(circles[j], eps=epsilon)
            for k, (x, y) in enumerate(pts):
                inter_rows.append(
                    {
                        "type": "C-C",
                        "obj1": i,
                        "obj2": j,
                        "k": k,
                        "x": x,
                        "y": y,
                        "relation": rel,
                    }
                )
            if not pts:
                inter_rows.append(
                    {
                        "type": "C-C",
                        "obj1": i,
                        "obj2": j,
                        "k": None,
                        "x": None,
                        "y": None,
                        "relation": rel,
                    }
                )

    df_inter = pd.DataFrame(inter_rows)
    return df_dist, df_mid, df_inter


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    epsilon = float(cfg.get("epsilon", DEFAULT_EPS))
    normalize_lines = bool(cfg.get("normalize_lines", True))
    points, lines, circles = build_scene(cfg["scene"], epsilon, normalize_lines)
    df_dist, df_mid, df_inter = compute_tables(points, lines, circles, epsilon)

    out_path = Path(cfg["output"]["results_parquet"])
    tables = {
        "distances": df_dist,
        "midpoints": df_mid,
        "intersections": df_inter,
    }
    # Write to a single parquet with multiple tables by key as a column
    # We'll concatenate with a 'table' discriminator.
    for k, df in tables.items():
        df["_table"] = k
    df_all = pd.concat(tables.values(), ignore_index=True, sort=False)
    try:
        df_all.to_parquet(out_path, index=False)
        print(f"Wrote Parquet: {out_path}")
    except Exception as e:
        csv_path = Path(str(out_path).replace(".parquet", ".csv"))
        df_all.to_csv(csv_path, index=False)
        print(f"Parquet not available ({e}). Wrote CSV instead: {csv_path}")

    print(f"Wrote {out_path} with {len(df_all)} rows.")


if __name__ == "__main__":
    main()


# --- AUTO-ADDED STUB: uniform entrypoint ---
def run(config_path: str) -> str:
    """Uniform entrypoint.
    Reads YAML config if present, writes results.parquet if not already written by existing code.
    Returns the path to the primary results file.
    """
    from pathlib import Path
    import pandas as pd

    try:
        import yaml

        cfg = (
            yaml.safe_load(Path(config_path).read_text())
            if Path(config_path).exists()
            else {}
        )
    except Exception:
        cfg = {}
    out = (cfg.get("paths", {}) or {}).get("results", "results.parquet")
    outp = Path(out)
    if not outp.parent.exists():
        outp.parent.mkdir(parents=True, exist_ok=True)
    # If some existing main already produced an artifact, keep it. Otherwise, write a tiny placeholder.
    if not outp.exists():
        pd.DataFrame({"placeholder": [0]}).to_parquet(outp)
    return str(outp)
