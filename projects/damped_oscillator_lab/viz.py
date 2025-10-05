"""
Visualization script: time series, phase portrait, and energy decay.

Usage:
    python viz.py --results results.parquet --outdir figs
"""
from __future__ import annotations
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries(df: pd.DataFrame, outdir: str) -> str:
    """Plot displacement x(t)."""
    fig, ax = plt.subplots()
    ax.plot(df["t"], df["x"], label="x(t) [m]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("displacement x [m]")
    ax.legend()
    path = os.path.join(outdir, "timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_phase(df: pd.DataFrame, outdir: str) -> str:
    """Plot phase portrait (x, v)."""
    fig, ax = plt.subplots()
    ax.plot(df["x"], df["v"], label="trajectory")
    ax.set_xlabel("displacement x [m]")
    ax.set_ylabel("velocity v [m/s]")
    ax.legend()
    path = os.path.join(outdir, "phase.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_energy(df: pd.DataFrame, outdir: str) -> str:
    """Plot energy decay E(t)."""
    fig, ax = plt.subplots()
    ax.plot(df["t"], df["E"], label="E(t) [J]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("energy E [J]")
    ax.legend()
    path = os.path.join(outdir, "energy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results.parquet")
    ap.add_argument("--outdir", type=str, default="figs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_parquet(args.results)

    p1 = plot_timeseries(df, args.outdir)
    p2 = plot_phase(df, args.outdir)
    p3 = plot_energy(df, args.outdir)
    print(f"Saved plots:\\n{p1}\\n{p2}\\n{p3}")


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

