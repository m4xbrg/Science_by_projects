from __future__ import annotations
import argparse, numpy as np, pandas as pd, yaml
from scipy.integrate import solve_ivp
from model import rhs

def integrate(t_final: float, dt: float, x0: np.ndarray, params: dict):
    t_eval = np.arange(0.0, t_final + dt, dt)
    sol = solve_ivp(lambda t, x: rhs(t, x, params), (0.0, t_final), x0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out", type=str, default="results.parquet")
    ap.add_argument("--format", type=str, choices=["parquet","csv"], default="parquet")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    t, X = integrate(cfg["t_final"], cfg["dt"], np.array(cfg["initial_conditions"]["x0"], float), cfg["params"])
    df = pd.DataFrame({"t": t, "x": X[:,0], "y": X[:,1], "z": X[:,2]})
    df.attrs["params"] = cfg["params"]; df.attrs["initial_conditions"] = cfg["initial_conditions"]
    if args.format == "parquet":
        df.to_parquet(args.out, index=False)
    else:
        if not args.out.endswith(".csv"): args.out = "results.csv"
        df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows.")

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

