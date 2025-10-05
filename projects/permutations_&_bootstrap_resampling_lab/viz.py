"""
Visualization utilities: plots coverage vs n and width distributions.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_coverage_vs_n(df: pd.DataFrame, outdir: Path) -> Path:
    cov = df.groupby(["n","statistic","method"], as_index=False)["covers"].mean()
    fig, ax = plt.subplots()
    for (stat, method), g in cov.groupby(["statistic","method"]):
        g = g.sort_values("n")
        ax.plot(g["n"], g["covers"], marker="o", label=f"{stat}-{method}")
    ax.axhline(1 - df['alpha'].iloc[0], linestyle="--")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage vs n")
    ax.legend()
    out = outdir / "coverage_vs_n.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def plot_width_distribution(df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots()
    nmax = df["n"].max()
    sub = df[df["n"] == nmax]
    labels = []
    data = []
    for (stat, method), g in sub.groupby(["statistic","method"]):
        labels.append(f"{stat}-{method}")
        data.append(g["width"].values)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xlabel("Statistic-Method @ largest n")
    ax.set_ylabel("CI width")
    ax.set_title(f"CI width distribution (n={nmax})")
    out = outdir / "width_boxplot.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="figs")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.results)
    p1 = plot_coverage_vs_n(df, outdir)
    p2 = plot_width_distribution(df, outdir)
    print(f"Wrote {p1} and {p2}")

if __name__ == "__main__":
    main()
