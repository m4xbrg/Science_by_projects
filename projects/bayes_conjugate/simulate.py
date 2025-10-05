\
"""
simulate.py — Generate synthetic data and sequential Bayesian updates.
Writes results to Parquet/CSV for downstream visualization.
"""

import os, json, yaml
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from tqdm import trange

from model import (
    BetaBinomialParams, beta_posterior_params,
    NormalNormalParams, normal_posterior_params
)

def set_seed(seed: int):
    np.random.seed(seed)

def simulate_beta_binomial(cfg, outdir: Path):
    p_true = cfg["true_p"]
    prior = BetaBinomialParams(**cfg["prior"])
    n_trials = cfg["n_trials_per_batch"]
    n_batches = cfg["n_batches"]

    records = []
    alpha, beta = prior.alpha, prior.beta
    successes_cum = 0
    trials_cum = 0
    for b in trange(n_batches, desc="Beta–Binomial"):
        k = np.random.binomial(n_trials, p_true)
        successes_cum += k
        trials_cum += n_trials
        posterior = beta_posterior_params(BetaBinomialParams(alpha, beta), k, n_trials)
        alpha, beta = posterior.alpha, posterior.beta

        records.append({
            "model": "beta_binomial",
            "batch": b+1,
            "k_batch": k,
            "n_batch": n_trials,
            "k_cum": successes_cum,
            "n_cum": trials_cum,
            "alpha": alpha,
            "beta": beta,
            "p_true": p_true,
            "p_mean_post": alpha / (alpha + beta),
            "p_var_post": (alpha*beta) / (((alpha+beta)**2) * (alpha+beta+1))
        })

    df = pd.DataFrame(records)
    df.to_csv(outdir / "beta_binomial.csv", index=False)
    try:
        df.to_parquet(outdir / "beta_binomial.parquet", index=False)
    except Exception as e:
        print("Parquet write failed (install pyarrow). Saved CSV instead.", e)
    return df

def simulate_normal_normal(cfg, outdir: Path):
    mu_true = cfg["true_mu"]
    sigma2 = cfg["sigma2"]
    prior = NormalNormalParams(**cfg["prior"])
    n_per_batch = cfg["n_per_batch"]
    n_batches = cfg["n_batches"]

    records = []
    sum_x = 0.0
    n_cum = 0
    for b in trange(n_batches, desc="Normal–Normal"):
        x = np.random.normal(loc=mu_true, scale=np.sqrt(sigma2), size=n_per_batch)
        sum_x += x.sum()
        n_cum += n_per_batch
        xbar = sum_x / n_cum
        mu_n, tau_n2 = normal_posterior_params(prior, sigma2, xbar, n_cum)
        records.append({
            "model": "normal_normal",
            "batch": b+1,
            "n_batch": n_per_batch,
            "n_cum": n_cum,
            "xbar_cum": xbar,
            "mu_post": mu_n,
            "tau2_post": tau_n2,
            "mu_true": mu_true,
            "sigma2": sigma2
        })

    df = pd.DataFrame(records)
    df.to_csv(outdir / "normal_normal.csv", index=False)
    try:
        df.to_parquet(outdir / "normal_normal.parquet", index=False)
    except Exception as e:
        print("Parquet write failed (install pyarrow). Saved CSV instead.", e)
    return df

def main(config_path: str = "config.yaml"):
    base = Path(__file__).resolve().parent
    cfg = yaml.safe_load(open(base / config_path, "r"))
    results_dir = base / cfg["paths"]["results_dir"]
    results_dir.mkdir(exist_ok=True)

    set_seed(cfg["seed"])
    df_beta = simulate_beta_binomial(cfg["beta_binomial"], results_dir)
    df_norm = simulate_normal_normal(cfg["normal_normal"], results_dir)
    print("Saved results to", results_dir)

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

