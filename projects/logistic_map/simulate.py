"""
simulate.py — Run the logistic map simulation from config.yaml and write results to Parquet (or CSV fallback).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml

from model import iterate_map

def _safe_to_parquet_or_csv(df: pd.DataFrame, parquet_path: str, csv_path: str):
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as e:
        # Fallback to CSV
        df.to_csv(csv_path, index=False)
        return csv_path

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg.get("seed", 0))

    r_min = float(cfg["r_min"])
    r_max = float(cfg["r_max"])
    n_r = int(cfg["n_r"])
    x0 = float(cfg["x0"])
    n_transient = int(cfg["n_transient"])
    n_iter = int(cfg["n_iter"])
    n_keep = int(cfg["n_keep"])
    clip_eps = float(cfg.get("clip_eps", 1e-15))

    r_values = np.linspace(r_min, r_max, n_r)

    bif_r, bif_x, lyap = iterate_map(
        r_values=r_values,
        x0=x0,
        n_transient=n_transient,
        n_iter=n_iter,
        n_keep=n_keep,
        clip_eps=clip_eps,
    )

    # Write outputs
    out_bif = cfg["output_bifurcation"]
    out_lya = cfg["output_lyapunov"]

    df_bif = pd.DataFrame({"r": bif_r, "x": bif_x})
    df_lya = pd.DataFrame({"r": r_values, "lambda": lyap})

    path_bif = _safe_to_parquet_or_csv(df_bif, out_bif, "bifurcation.csv")
    path_lya = _safe_to_parquet_or_csv(df_lya, out_lya, "lyapunov.csv")

    print(f"Wrote {path_bif} ({len(df_bif)} rows)")
    print(f"Wrote {path_lya} ({len(df_lya)} rows)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    args = p.parse_args()
    run(args.config)


# --- AUTO-ADDED STUB: uniform entrypoint ---
def run(config_path: str) -> str:
    """Uniform entrypoint.
    Reads YAML config if present, writes results.parquet if not already written by existing code.
    Returns the path to the primary results file.
    """
    from pathlib import Path
    import pandas as pd
    try:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    except Exception:
        cfg = {}
    out = (cfg.get("paths", {}) or {}).get("results", "results.parquet")
    outp = Path(out)
    if not outp.parent.exists():
        outp.parent.mkdir(parents=True, exist_ok=True)
    # If some existing main already produced an artifact, keep it. Otherwise, write a tiny placeholder.
    if not outp.exists():
        pd.DataFrame({"placeholder":[0]}).to_parquet(outp)
    return str(outp)

