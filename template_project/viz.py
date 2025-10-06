"""
Visualization utilities.
- Reads Parquet results
- Saves figures to figs/
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_time_series(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for col in df.columns:
        if col != "t":
            ax.plot(df["t"], df[col], label=col)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("State")
    ax.legend()
    ax.set_title("Time Series")
    fig.savefig(outdir / "time_series.png", dpi=150)
    plt.close(fig)


def plot_phase(df: pd.DataFrame, outdir: Path) -> None:
    cols = [c for c in df.columns if c != "t"]
    if len(cols) >= 2:
        fig, ax = plt.subplots()
        ax.plot(df[cols[0]], df[cols[1]])
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_title("Phase Portrait")
        fig.savefig(outdir / "phase.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results.parquet")
    ap.add_argument("--outdir", type=str, default="figs")
    args = ap.parse_args()
    df = pd.read_parquet(args.input)
    outdir = Path(args.outdir)
    plot_time_series(df, outdir)
    plot_phase(df, outdir)
    print(f"Saved figures to {outdir}")


if __name__ == "__main__":
    main()
