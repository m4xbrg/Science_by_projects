"""
Visualization utilities: learning curves, MSE vs degree, bias–variance curves.
"""

import argparse, yaml, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_mse_vs_degree(results_path: str, out_path: str, which: str = "val"):
    df = pd.read_parquet(results_path)
    key = "mse_val" if which == "val" else "mse_train"
    g = df.groupby(["degree", "lambda"], as_index=False)[key].mean()
    lambdas = sorted(g["lambda"].unique())
    plt.figure()
    for lam in lambdas:
        sub = g[g["lambda"] == lam].sort_values("degree")
        plt.plot(sub["degree"], sub[key], marker="o", label=f"λ={lam}")
    plt.xlabel("Polynomial degree")
    plt.ylabel(f"Mean {key.replace('_', ' ').upper()}")
    plt.title(f"{key.replace('_', ' ').title()} vs Degree (averaged over folds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def plot_learning_curve(results_path: str, out_path: str):
    df = pd.read_parquet(results_path)
    tr = df.groupby(["degree", "lambda"], as_index=False)["mse_train"].mean()
    va = df.groupby(["degree", "lambda"], as_index=False)["mse_val"].mean()
    # Choose the lambda that minimizes avg val MSE at each degree and plot train vs val
    best = (
        va.sort_values(["degree", "mse_val"])
        .groupby("degree", as_index=False)
        .first()[["degree", "lambda", "mse_val"]]
    )
    merged = best.merge(tr, on=["degree", "lambda"], how="left")
    plt.figure()
    plt.plot(merged["degree"], merged["mse_train"], marker="o", label="Train MSE")
    plt.plot(
        merged["degree"],
        merged["mse_val"],
        marker="o",
        label="Val MSE (best λ per degree)",
    )
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean MSE")
    plt.title("Learning Curve: Train vs Validation Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def plot_bias_variance(bv_path: str, out_path_prefix: str):
    df = pd.read_parquet(bv_path)
    agg = df.groupby("degree", as_index=False)[["bias2", "variance", "noise"]].mean()
    plt.figure()
    plt.plot(agg["degree"], agg["bias2"], marker="o", label="Bias$^2$")
    plt.plot(agg["degree"], agg["variance"], marker="o", label="Variance")
    plt.plot(
        agg["degree"],
        agg["bias2"] + agg["variance"],
        marker="o",
        label="Bias$^2$ + Variance",
    )
    plt.xlabel("Polynomial degree")
    plt.ylabel("Average across x")
    plt.title("Bias–Variance Decomposition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path_prefix}_aggregate.png", dpi=180)


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    figs = Path(cfg["paths"]["figs_dir"])
    figs.mkdir(parents=True, exist_ok=True)
    results_path = cfg["paths"]["results"]
    bv_path = cfg["paths"]["bias_variance"]
    plot_mse_vs_degree(results_path, str(figs / "mse_vs_degree_val.png"), which="val")
    plot_learning_curve(results_path, str(figs / "learning_curve.png"))
    if Path(bv_path).exists():
        plot_bias_variance(bv_path, str(figs / "bias_variance"))
    print("Figures written to", figs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)


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
                col = c
                break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index")
    plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
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
                col = c
                break
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
    plt.xlabel(str(col))
    plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out
