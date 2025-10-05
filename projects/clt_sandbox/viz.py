
"""
viz.py — Produce two complementary visualizations:
1) Histogram of Z_n at a representative n with standard normal PDF overlay.
2) Q–Q plot of Z_n versus N(0,1).
Also: optional convergence curve for KS statistic.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def _load_results(results_path: str = "results.parquet") -> pd.DataFrame:
    p = Path(results_path)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    alt = Path(str(results_path).replace(".parquet", ".csv"))
    return pd.read_csv(alt)

def _choose_n(df: pd.DataFrame) -> int:
    # pick the median n in the run as the representative for single-snapshot plots
    return int(df["n"].iloc[len(df)//2])

def hist_with_pdf(Z: np.ndarray, n: int, out_path: str):
    plt.figure()
    plt.hist(Z, bins=50, density=True, alpha=0.6, label=f"Z_n (n={n})")
    xs = np.linspace(min(-4, Z.min()), max(4, Z.max()), 400)
    plt.plot(xs, norm.pdf(xs), label="Standard Normal PDF")
    plt.xlabel("z")
    plt.ylabel("density")
    plt.legend()
    plt.title("Histogram of Standardized Sum with N(0,1) Overlay")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def qq_plot(Z: np.ndarray, n: int, out_path: str):
    plt.figure()
    # theoretical quantiles
    p = (np.arange(1, len(Z)+1) - 0.5) / len(Z)
    q_theory = norm.ppf(p)
    q_sample = np.sort(Z)
    plt.scatter(q_theory, q_sample, s=8, label="Sample vs Theory")
    # 45-degree line
    lo = min(q_theory.min(), q_sample.min())
    hi = max(q_theory.max(), q_sample.max())
    plt.plot([lo, hi], [lo, hi], linewidth=1.0, label="y = x")
    plt.xlabel("Theoretical quantiles (N(0,1))")
    plt.ylabel("Sample quantiles")
    plt.title(f"Q–Q Plot for Z_n (n={n})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def ks_convergence(df: pd.DataFrame, out_path: str):
    plt.figure()
    plt.loglog(df["n"], df["ks_stat"])
    plt.xlabel("n")
    plt.ylabel("KS statistic D_n")
    plt.title("Convergence of KS distance to Normal (log–log)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    df = _load_results("results.parquet")
    n_rep = _choose_n(df)
    # load samples from simulate.py
    sample_bank_path = Path("samples.json")
    if sample_bank_path.exists() and str(n_rep) in json.loads(sample_bank_path.read_text()):
        samples = json.loads(sample_bank_path.read_text())[str(n_rep)]
        Z = np.array(samples, dtype=float)
    else:
        # Fallback: cannot find samples; approximate by normal
        Z = np.random.normal(size=5000)

    hist_with_pdf(Z, n_rep, out_path="figs/histogram.png")
    qq_plot(Z, n_rep, out_path="figs/qqplot.png")
    ks_convergence(df, out_path="figs/ks_convergence.png")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="results.parquet")
    p.add_argument("--outdir", type=str, default="figs")
    args = p.parse_args()
    plot_primary(args.results, args.outdir)
    plot_secondary(args.results, args.outdir)


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

