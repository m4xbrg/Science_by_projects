"""
Simulation runner.
- Loads config.yaml
- Integrates ODE using SciPy
- Saves results to Parquet for downstream visualization.
"""
import argparse, json, yaml, numpy as np, pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from model import ModelParams, rhs

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--output", type=str, default="results.parquet")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    t_final = float(cfg.get("t_final", 10.0))
    dt = float(cfg.get("dt", 0.01))
    t_eval = np.arange(0.0, t_final + 1e-12, dt)

    p = ModelParams(**cfg.get("params", {}))
    x0 = np.array(cfg.get("initial_conditions", {}).get("x0", [1.0, 0.0]), dtype=float)

    sol = solve_ivp(fun=lambda t, x: rhs(t, x, p),
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=x0, t_eval=t_eval, rtol=1e-8, atol=1e-10, method="RK45")

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    df = pd.DataFrame({"t": sol.t, **{f"x{i}": sol.y[i] for i in range(sol.y.shape[0])}})
    df.to_parquet(args.output, index=False)
    print(f"Wrote {args.output} with shape {df.shape}")

if __name__ == "__main__":
    main()