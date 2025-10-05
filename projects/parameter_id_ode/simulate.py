"""
Experiment runner for generating synthetic datasets: integrate ODE and add noise.
Writes results to Parquet with columns: t, x1, x2, y.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from model import simulate_trajectory, add_observation_noise

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    t = np.arange(0.0, cfg["t_final"] + 1e-12, cfg["dt"])
    x0 = np.array(cfg["x0"], dtype=float)
    params = cfg["true_params"]
    X = simulate_trajectory(t, x0, params)
    y = add_observation_noise(X, cfg["obs"]["observe"], cfg["obs"]["noise_std"], rng)

    df = pd.DataFrame({"t": t, "x1": X[:,0], "x2": X[:,1], "y": y})
    out_path = Path(cfg["paths"]["results_parquet"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path}")
    except Exception as e:
        print(f"Parquet unavailable ({e}); writing CSV fallback.")
    csv_path = Path(cfg["paths"]["results_csv"])
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    main(args.config)
