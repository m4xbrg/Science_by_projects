import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_results(path_dir: str = ".") -> pd.DataFrame:
    """Load results from parquet if available else csv."""
    pq = os.path.join(path_dir, "results.parquet")
    cs = os.path.join(path_dir, "results.csv")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    return pd.read_csv(cs)

def performance_plot(df: pd.DataFrame, out_path: str):
    """Make F1 vs contamination plot (one line per method)."""
    fig = plt.figure()
    for m in sorted(df["method"].unique()):
        sub = df[df["method"] == m].sort_values("epsilon")
        plt.plot(sub["epsilon"], sub["f1"], marker="o", label=m)
    plt.xlabel("contamination ε")
    plt.ylabel("F1 score")
    plt.legend()
    plt.title("Outlier detection performance vs contamination")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def annotated_sample_plot(sample_csv: str, out_path: str, tau: float = 3.0, k: float = 1.5):
    """Show data distribution and flags for classical z and IQR on a sample."""
    import numpy as np
    import pandas as pd
    from model import ZParams, IQRParams, classical_z_scores, iqr_fences

    df = pd.read_csv(sample_csv)
    x = df["x"].to_numpy()
    # Classical z
    mz, stats_z = classical_z_scores(x, ZParams(tau=tau))
    # IQR
    mi, stats_i = iqr_fences(x, IQRParams(k=k))

    fig = plt.figure()
    # Rug plot-like scatter with jitter on y for visibility
    y0 = np.zeros_like(x, dtype=float)
    plt.scatter(x[mz==0], y0[mz==0], s=8, alpha=0.6, label="inlier (z)")
    plt.scatter(x[mz==1], y0[mz==1]+0.02, s=12, alpha=0.9, label="flagged by z")
    plt.scatter(x[mi==1], y0[mi==1]-0.02, s=12, alpha=0.9, label="flagged by IQR")

    # Show fences / thresholds
    mu, s = stats_z["mu"], stats_z["std"]
    plt.axvline(mu - stats_z["tau"]*s, linestyle="--")
    plt.axvline(mu + stats_z["tau"]*s, linestyle="--")
    plt.axvline(stats_i["lo"], linestyle=":")
    plt.axvline(stats_i["hi"], linestyle=":")

    plt.yticks([])
    plt.xlabel("x")
    plt.title("Sample with classical z vs IQR flags")
    plt.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def build_all(fig_dir: str = "figs"):
    df = load_results(".")
    os.makedirs(fig_dir, exist_ok=True)
    performance_plot(df, os.path.join(fig_dir, "performance.png"))

    # Find sample file
    art = json.load(open("ARTIFACTS.json"))
    annotated_sample_plot(art["sample"], os.path.join(fig_dir, "sample_flags.png"))

if __name__ == "__main__":
    build_all("figs")

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

