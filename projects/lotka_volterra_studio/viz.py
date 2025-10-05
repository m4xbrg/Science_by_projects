"""
viz.py — plots: time series, phase portrait with nullclines & fixed points, and sweep summary
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import Dict, Any
from model import LVParams, GridSpec, nullclines, fixed_points

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def plot_time_series(df: pd.DataFrame, outdir: Path):
    plt.figure()
    plt.plot(df["t"], df["x"], label="Prey x(t)")
    plt.plot(df["t"], df["y"], label="Predator y(t)")
    plt.xlabel("time")
    plt.ylabel("population (arbitrary)")
    plt.legend()
    out = outdir / "time_series.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def plot_phase_portrait_with_nullclines(df: pd.DataFrame, cfg: Dict[str, Any], outdir: Path):
    # choose middle of sweep (or single run) to compute nullclines and fixed points
    params = LVParams(**cfg["params"])
    if "sweep_value" in df.columns and df["sweep_value"].notna().any():
        mid = df["sweep_value"].dropna().unique()
        mid = np.sort(mid)[len(mid)//2]
        p = dict(cfg["params"])
        p[cfg["continuation"]["param_name"]] = float(mid)
        params = LVParams(**p)

    grid = GridSpec(**cfg["grid"])
    nc = nullclines(grid, params)
    fps = fixed_points(params)

    plt.figure()
    plt.plot(df["x"], df["y"], label="trajectory")
    plt.axhline(nc["dx0_y"][0], linestyle="--", label="dx/dt=0")
    plt.axvline(nc["dy0_x"][0], linestyle="--", label="dy/dt=0")
    plt.scatter(fps[:,0], fps[:,1], marker="o", label="fixed points")
    plt.xlabel("x (prey)")
    plt.ylabel("y (predator)")
    plt.legend()
    out = outdir / "phase_portrait.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def plot_sweep_summary(df: pd.DataFrame, outdir: Path):
    if "sweep_value" not in df.columns or df["sweep_value"].isna().all():
        return
    # simple bifurcation-like plot: show x maxima after transient vs parameter
    agg = df.dropna(subset=["sweep_value"]).groupby("sweep_value").agg({"x_max_after_transient":"max", "x_min_after_transient":"min"}).reset_index()
    plt.figure()
    plt.scatter(agg["sweep_value"], agg["x_max_after_transient"], label="x max (post-transient)")
    plt.scatter(agg["sweep_value"], agg["x_min_after_transient"], label="x min (post-transient)")
    plt.xlabel( df["sweep_param"].dropna().iloc[0] )
    plt.ylabel("x extrema (post-transient)")
    plt.legend()
    out = outdir / "bifurcation_summary.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def main():
    cfg = load_config("config.yaml")
    outdir = Path("figs")
    outdir.mkdir(exist_ok=True, parents=True)
    df = pd.read_parquet("results.parquet")
    plot_time_series(df, outdir)
    plot_phase_portrait_with_nullclines(df, cfg, outdir)
    plot_sweep_summary(df, outdir)
    print(f"Wrote figures to {outdir}")

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
                col = c; break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index"); plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
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
                col = c; break
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
    plt.xlabel(str(col)); plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out

