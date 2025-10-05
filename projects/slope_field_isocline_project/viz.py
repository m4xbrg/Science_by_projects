"""
Visualization: slope field + isoclines; error vs step size plot.
"""
from __future__ import annotations
import os, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import parse_rhs, integrate

def slope_field_and_isoclines(cfg, save_path="figs/slope_isoclines.png"):
    f = parse_rhs(cfg["expr"]); P = cfg.get("params", {}); g = cfg["grid"]
    t = np.linspace(g["t_min"], g["t_max"], g["n_t"])
    x = np.linspace(g["x_min"], g["x_max"], g["n_x"])
    T, X = np.meshgrid(t, x)
    F = np.vectorize(lambda tt, xx: f(tt, xx, P))(T, X)
    U = np.ones_like(F); V = F; L = np.hypot(U, V); U2, V2 = U/L, V/L
    fig = plt.figure()
    plt.quiver(T, X, U2, V2, angles="xy", scale=20, width=0.002)
    CS = plt.contour(T, X, F, levels=cfg.get("isocline_levels", [-2,-1,0,1,2]), linestyles="--")
    plt.clabel(CS, inline=True, fontsize=8)
    dt = sorted(cfg["dt_list"])[-1]
    t_vec, x_vec = integrate(f, cfg["t0"], cfg["x0"], cfg["t_final"], dt, method="rk4", params=P)
    plt.plot(t_vec, x_vec, label="RK4 sample trajectory")
    plt.xlabel("t"); plt.ylabel("x"); plt.legend(loc="best")
    plt.title("Slope Field with Isoclines and Sample Trajectory")
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=180, bbox_inches="tight"); plt.close(fig)

def error_vs_stepsize(cfg, df_path="results.parquet", save_path="figs/error_vs_stepsize.png"):
    # Load parquet or csv fallback
    if os.path.exists(df_path):
        df = pd.read_parquet(df_path)
    elif os.path.exists(df_path.replace(".parquet", ".csv")):
        df = pd.read_csv(df_path.replace(".parquet", ".csv"))
    else:
        raise FileNotFoundError("results parquet/csv not found. Run simulate.py first.")
    fig = plt.figure()
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("dt"); plt.loglog(sub["dt"], sub["global_error"], marker="o", label=method)
    xs = np.array(sorted(df["dt"].unique()))
    plt.loglog(xs, xs/xs.max(), linestyle=":", label="slope 1 (guide)")
    plt.loglog(xs, (xs/xs.max())**4, linestyle=":", label="slope 4 (guide)")
    plt.xlabel("Δt (step size)"); plt.ylabel("Global error at t_final"); plt.legend(loc="best")
    plt.title("Global Error vs Step Size")
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=180, bbox_inches="tight"); plt.close(fig)

def load_cfg(path="config.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_cfg(); slope_field_and_isoclines(cfg); error_vs_stepsize(cfg)

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

