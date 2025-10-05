import argparse, os, pandas as pd, matplotlib.pyplot as plt

def plot_error_rates(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for method in ["Bonferroni", "BH"]:
        sub = df[df["method"] == method]
        plt.plot(sub["alpha"], sub["mean_FDR"], marker="o", label=f"{method} FDR")
        plt.plot(sub["alpha"], sub["FWER"], marker="x", label=f"{method} FWER")
    plt.xlabel("Nominal level (alpha)")
    plt.ylabel("Rate")
    plt.title("Error Rates vs Alpha (Monte Carlo)")
    plt.legend(); plt.tight_layout()
    p = os.path.join(outdir, "error_rates_vs_alpha.png")
    plt.savefig(p, dpi=160); plt.close()
    return p

def plot_power(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    for method in ["Bonferroni", "BH"]:
        sub = df[df["method"] == method]
        plt.plot(sub["alpha"], sub["TPR"], marker="o", label=f"{method} TPR")
    plt.xlabel("Nominal level (alpha)")
    plt.ylabel("True Positive Rate (Power)")
    plt.title("Power vs Alpha (Monte Carlo)")
    plt.legend(); plt.tight_layout()
    p = os.path.join(outdir, "power_vs_alpha.png")
    plt.savefig(p, dpi=160); plt.close()
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results.csv")
    ap.add_argument("--outdir", type=str, default="figs")
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    print(plot_error_rates(df, args.outdir))
    print(plot_power(df, args.outdir))

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

