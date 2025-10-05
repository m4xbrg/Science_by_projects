"""
Visualization script for Logistic/Riccati results.
Generates:
1) Time evolution plot: x_numeric (and x_analytic if present)
2) 1D "phase portrait proxy": x vs f(x) to show fixed points and flow direction

Usage:
    python viz.py results.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_time_series(df: pd.DataFrame, outdir: Path):
    # One figure per run_id
    for rid, g in df.groupby("run_id"):
        fig, ax = plt.subplots()
        ax.plot(g["t"], g["x_numeric"], label="numeric")
        if "x_analytic" in g and np.isfinite(g["x_analytic"]).any():
            ax.plot(g["t"], g["x_analytic"], linestyle="--", label="analytic")
        ax.set_xlabel("t")
        ax.set_ylabel("x(t)")
        title = f"Time Evolution ({g['model'].iloc[0]}) – {rid}"
        ax.set_title(title)
        ax.legend()
        fig.savefig(outdir / f"time_series_{rid}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def plot_phase_proxy(df: pd.DataFrame, outdir: Path):
    # Plot x vs f(x) at representative parameters (use first row of each group)
    for rid, g in df.groupby("run_id"):
        model = g["model"].iloc[0]
        # Construct vector field f(x)
        xgrid = np.linspace(min(g["x_numeric"].min(), 0.0), max(g["x_numeric"].max(), 1.0), 400)
        if model == "logistic":
            r = float(g["r"].iloc[0]); K = float(g["K"].iloc[0])
            f = r * xgrid * (1.0 - xgrid / K)
        else:
            a = float(g["a"].iloc[0]); b = float(g["b"].iloc[0]); c = float(g["c"].iloc[0])
            f = a * xgrid**2 + b * xgrid + c

        fig, ax = plt.subplots()
        ax.plot(xgrid, f, label="f(x)")
        ax.axhline(0.0)
        ax.set_xlabel("x")
        ax.set_ylabel("dx/dt = f(x)")
        ax.set_title(f"Phase Portrait Proxy – {model} – {rid}")
        ax.legend()
        fig.savefig(outdir / f"phase_proxy_{rid}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def main(argv=None):
    argv = sys.argv if argv is None else argv
    inpath = Path(argv[1]) if len(argv) > 1 else Path("results.parquet")
    outdir = Path("figs")
    outdir.mkdir(exist_ok=True, parents=True)

    df = pd.read_parquet(inpath)
    plot_time_series(df, outdir)
    plot_phase_proxy(df, outdir)
    print(f"Saved figures to {outdir}")

if __name__ == "__main__":
    main()
