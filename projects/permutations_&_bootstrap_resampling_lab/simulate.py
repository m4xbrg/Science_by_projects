"""
Experiment runner: reads config, runs coverage experiments, writes results.parquet
"""
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
from model import sample_distribution, true_parameter, one_sample_ci

def run_experiments(cfg: dict, progress: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.get("seed", 123))
    alpha = cfg["alpha"]
    B = int(cfg["bootstrap_draws"])
    reps = int(cfg["reps"])
    n_list = list(cfg["n_list"])
    stat_list = list(cfg["statistic_list"])
    methods = list(cfg["ci_methods"])
    family = cfg["distribution"]["family"]
    params = dict(cfg["distribution"].get("params", {}))

    rows = []
    for n in n_list:
        for statistic in stat_list:
            theta = true_parameter(family, params, statistic)
            for method in methods:
                for r in range(reps):
                    x = sample_distribution(rng, family, n, params)
                    T, lo, hi = one_sample_ci(rng, x, statistic, method, B, alpha)
                    width = hi - lo
                    cover = (theta >= lo) and (theta <= hi)
                    rows.append({
                        "n": n, "statistic": statistic, "method": method, "rep": r,
                        "T": T, "lo": lo, "hi": hi, "width": width, "covers": int(cover),
                        "theta_true": theta, "family": family, "alpha": alpha, "B": B
                    })
                if progress:
                    print(f"Done: n={n} stat={statistic} method={method}")
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    df = run_experiments(cfg, progress=True)
    out_path = Path(cfg.get("results_path", "results.parquet"))
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows")

if __name__ == "__main__":
    main()
