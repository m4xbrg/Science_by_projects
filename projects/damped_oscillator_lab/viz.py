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