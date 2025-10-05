"""
Experiment runner: load config, generate or load data, run k-fold CV, write results parquet.
Optionally compute bias–variance curves on synthetic ground truth.
"""
import argparse, json, yaml, pandas as pd, numpy as np
from pathlib import Path
from model import cross_validate, synthetic_data, bias_variance_curves

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    rng_seed = int(cfg.get("seed", 0))
    data_cfg = cfg["data"]
    paths = cfg["paths"]
    degrees = cfg["degrees"]
    lambdas = cfg["lambdas"]
    k = int(cfg["k_folds"])
    scale = bool(cfg["scale_features"])
    include_bias = bool(cfg["include_bias"])

    # Data
    if data_cfg.get("use_synthetic", True):
        X, y, ftrue = synthetic_data(
            n=int(data_cfg["n_samples"]),
            x_range=tuple(data_cfg.get("x_range", (-1,1))),
            noise_sigma=float(data_cfg["noise_sigma"]),
            seed=rng_seed,
            func=data_cfg.get("ground_truth", "sin(2*pi*x)")
        )
    else:
        raise NotImplementedError("Only synthetic data is wired in by default.")

    # CV
    results = cross_validate(
        X, y,
        degrees=degrees, lambdas=lambdas, k=k,
        scale=scale, include_bias=include_bias, seed=rng_seed
    )
    pd.DataFrame(results).to_parquet(paths["results"])

    # Bias–variance (synthetic only)
    bv_cfg = cfg.get("bias_variance", {})
    if data_cfg.get("use_synthetic", True) and bv_cfg:
        G = int(bv_cfg.get("test_grid_n", 200))
        x_grid = np.linspace(data_cfg.get("x_range", (-1,1))[0], data_cfg.get("x_range", (-1,1))[1], G)
        # Use a representative lambda (small ridge) for stability
        lam = min(lambdas) if min(lambdas) > 0 else (lambdas[1] if len(lambdas) > 1 else 1e-3)
        bv = bias_variance_curves(
            ftrue=ftrue, x_grid=x_grid, degrees=degrees, lam=lam,
            n_trials=int(bv_cfg.get("n_trials", 50)),
            n_train=int(data_cfg["n_samples"]),
            noise_sigma=float(data_cfg["noise_sigma"]),
            include_bias=include_bias, scale=scale, seed=rng_seed
        )
        bv.to_parquet(paths["bias_variance"])

    print(f"Wrote {paths['results']} and (optional) {paths['bias_variance']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
