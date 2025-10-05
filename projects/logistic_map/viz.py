"""
viz.py — Create bifurcation diagram and Lyapunov exponent plot from Parquet/CSV outputs.
"""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os

def _read_any(path_parquet: str, path_csv: str) -> pd.DataFrame:
    if os.path.exists(path_parquet):
        try:
            return pd.read_parquet(path_parquet)
        except Exception:
            pass
    return pd.read_csv(path_csv)

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    bif_pq = cfg["output_bifurcation"]
    lya_pq = cfg["output_lyapunov"]
    df_bif = _read_any(bif_pq, "bifurcation.csv")
    df_lya = _read_any(lya_pq, "lyapunov.csv")

    # --- Bifurcation diagram ---
    plt.figure()
    plt.scatter(df_bif["r"], df_bif["x"], s=0.1)
    plt.xlabel("r (control parameter)")
    plt.ylabel("x (state)")
    plt.title("Logistic Map Bifurcation Diagram")
    plt.tight_layout()
    plt.savefig("figs/bifurcation.png", dpi=300)
    plt.close()

    # --- Lyapunov exponent curve ---
    plt.figure()
    plt.plot(df_lya["r"], df_lya["lambda"], linewidth=1.0)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("r (control parameter)")
    plt.ylabel("λ_max(r)")
    plt.title("Maximal Lyapunov Exponent vs r (Logistic Map)")
    plt.tight_layout()
    plt.savefig("figs/lyapunov.png", dpi=300)
    plt.close()

    print("Saved figs/bifurcation.png and figs/lyapunov.png")

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

