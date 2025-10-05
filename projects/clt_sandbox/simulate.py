
"""
simulate.py — Generate standardized sum samples across n_list and summarize convergence.
Outputs results.parquet with summary metrics and a few diagnostic quantiles.
"""
import argparse, json
import numpy as np, pandas as pd, yaml
from pathlib import Path
from scipy import stats
from model import IIDSpec, standardized_sum_samples

def run_sim(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    rng = np.random.default_rng(cfg.get("seed", None))
    spec = IIDSpec(name=cfg["distribution"]["name"], params=cfg["distribution"]["params"])
    n_list = cfg["n_list"]
    trials = int(cfg["trials_per_n"])
    use_pop = bool(cfg.get("use_population_moments", True))

    rows = []
    # Optionally store a small subsample of Z for quick plotting at a representative n
    sample_bank = {}

    for n in n_list:
        Z = standardized_sum_samples(rng, spec, n=n, trials=trials, use_population_moments=use_pop)
        # Kolmogorov-Smirnov distance against N(0,1)
        ks = stats.kstest(Z, 'norm')
        # Anderson-Darling statistic for additional sensitivity in tails
        ad = stats.anderson(Z, dist='norm')
        rows.append({
            "n": n,
            "trials": trials,
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "ad_stat": float(ad.statistic),
            "mean": float(np.mean(Z)),
            "std": float(np.std(Z, ddof=1)),
            "q05": float(np.quantile(Z, 0.05)),
            "q50": float(np.quantile(Z, 0.50)),
            "q95": float(np.quantile(Z, 0.95)),
        })
        if n in (n_list[min(3, len(n_list)-1)], n_list[-2] if len(n_list)>1 else n_list[0]):
            # store up to 5000 for later hist/QQ
            sample_bank[str(n)] = Z[: min(5000, len(Z))].astype(float).tolist()

    df = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    out_path = Path(cfg["results_file"]).resolve() if Path(cfg["results_file"]).is_absolute() \
               else Path.cwd() / cfg["results_file"]
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        out_csv = Path(str(out_path).replace(".parquet", ".csv"))
        df.to_csv(out_csv, index=False)
        out_path = out_csv
    return df, sample_bank, cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    df, sample_bank, cfg = run_sim(args.config)
    # write a small JSON with stored samples for quick plotting
    Path("samples.json").write_text(json.dumps(sample_bank))
    print(f"Wrote {cfg['results_file']} with {len(df)} rows and samples.json")
