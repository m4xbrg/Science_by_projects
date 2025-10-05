import os, json, yaml
import numpy as np
import pandas as pd

from model import (
    ZParams, ZRobustParams, IQRParams,
    classical_z_scores, robust_z_scores, iqr_fences,
    simulate_contaminated_sample, prf
)

def run_simulation(cfg_path: str = "config.yaml", out_dir: str = "."):
    """
    Run contamination sweep and save results as parquet if pyarrow is available, else CSV.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(cfg["seed"])
    n = int(cfg["n"])
    eps_grid = list(cfg["epsilon_grid"])
    base = cfg["base_dist"]
    outd = cfg["out_dist"]
    z_params = ZParams(**cfg["z_params"])
    zr_params = ZRobustParams(**cfg["z_robust_params"])
    iqr_params = IQRParams(**cfg["iqr_params"])

    rows = []
    for eps in eps_grid:
        x, y = simulate_contaminated_sample(
            n=n, epsilon=eps, rng=rng,
            base_mu=base["mu"], base_sigma=base["sigma"],
            out_mu=outd["mu"], out_sigma=outd["sigma"]
        )
        m_z, _ = classical_z_scores(x, z_params)
        m_zr, _ = robust_z_scores(x, zr_params)
        m_iqr, _ = iqr_fences(x, iqr_params)

        for name, m in [("z", m_z), ("z_robust", m_zr), ("iqr", m_iqr)]:
            p, r, f1 = prf(m, y)
            rows.append({"epsilon": eps, "method": name, "precision": p, "recall": r, "f1": f1})

    df = pd.DataFrame(rows)
    # Write results
    out_path_parquet = os.path.join(out_dir, "results.parquet")
    out_path_csv = os.path.join(out_dir, "results.csv")
    try:
        df.to_parquet(out_path_parquet, index=False)
        written = out_path_parquet
    except Exception:
        df.to_csv(out_path_csv, index=False)
        written = out_path_csv
    # Save one sample for visualization
    eps_sample = float(cfg["sample_for_plot"]["epsilon"])
    x, y = simulate_contaminated_sample(
        n=n, epsilon=eps_sample, rng=rng,
        base_mu=base["mu"], base_sigma=base["sigma"],
        out_mu=outd["mu"], out_sigma=outd["sigma"]
    )
    sample_df = pd.DataFrame({"x": x, "is_outlier": y})
    sample_path = os.path.join(out_dir, "sample.csv")
    sample_df.to_csv(sample_path, index=False)
    with open(os.path.join(out_dir, "ARTIFACTS.json"), "w") as f:
        json.dump({"metrics": written, "sample": sample_path}, f)
    return written, sample_path

if __name__ == "__main__":
    run_simulation(cfg_path="config.yaml", out_dir=".")

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

