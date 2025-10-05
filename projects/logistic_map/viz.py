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
    main()
