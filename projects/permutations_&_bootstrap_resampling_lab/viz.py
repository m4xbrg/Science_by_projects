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

