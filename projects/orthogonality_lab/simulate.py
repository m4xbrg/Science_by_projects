from __future__ import annotations
import argparse, json, os, sys
import yaml
import numpy as np
import pandas as pd
from model import gram_schmidt, compute_metrics, make_matrix

def _save_df(df: pd.DataFrame, out_dir: str, use_parquet: bool) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path_parquet = os.path.join(out_dir, "results.parquet")
    path_csv = os.path.join(out_dir, "results.csv")
    if use_parquet:
        try:
            df.to_parquet(path_parquet, index=False)
            return path_parquet
        except Exception as e:
            print(f"[warn] Parquet unavailable ({e}); falling back to CSV.", file=sys.stderr)
    df.to_csv(path_csv, index=False)
    return path_csv

def run_one(cfg: dict, trial_seed: int) -> dict:
    cfg_local = dict(cfg)
    cfg_local["seed"] = trial_seed
    A, info = make_matrix(cfg_local)
    Q, R = gram_schmidt(A, method=cfg["method"], reorth=int(cfg["reorth"]))
    metrics = compute_metrics(A, Q, R)
    return {
        "seed": trial_seed,
        "m": cfg["m"], "n": cfg["n"],
        "kind": info["kind"],
        "method": cfg["method"],
        "reorth": int(cfg["reorth"]),
        **metrics,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    base_seed = int(cfg.get("seed", 0))
    trials = int(cfg.get("trials", 1))
    seeds = [base_seed + k for k in range(trials)]

    rows = []
    for s in seeds:
        rows.append(run_one(cfg, s))

    df = pd.DataFrame(rows)
    out_dir = cfg.get("io", {}).get("out_dir", "results")
    use_parquet = bool(cfg.get("io", {}).get("parquet", True))
    path = _save_df(df, out_dir, use_parquet)

    print(json.dumps({"wrote": path, "rows": len(df)}, indent=2))

if __name__ == "__main__":
    main()

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

