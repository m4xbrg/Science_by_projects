"""
Run error-vs-stepsize sweeps for Euler and RK4 on a user-defined ODE.
Outputs results.parquet (requires pyarrow).
"""
from __future__ import annotations
import os, yaml
import numpy as np
import pandas as pd
from model import parse_rhs, integrate, integrate_reference

def run(config_path: str = "config.yaml", out_dir: str = "."):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    f_rhs = parse_rhs(cfg["expr"])
    params = cfg.get("params", {})
    t0 = float(cfg.get("t0", 0.0)); t_final = float(cfg.get("t_final", 1.0))
    x0 = float(cfg.get("x0", 0.0))
    dt_list = list(cfg.get("dt_list", [0.1, 0.05, 0.025]))
    dt_ref = float(cfg.get("dt_ref", 1e-3))
    out_parquet = os.path.join(out_dir, cfg.get("output_parquet", "results.parquet"))

    t_ref, x_ref = integrate_reference(f_rhs, t0, x0, t_final, dt_ref, params)
    x_ref_final = float(x_ref[-1])

    rows = []
    for method in ["euler", "rk4"]:
        for dt in dt_list:
            t, x = integrate(f_rhs, t0, x0, t_final, dt, method, params)
            rows.append({"method": method.upper(), "dt": dt, "t_final": t_final,
                         "x_final": float(x[-1]), "x_ref_final": x_ref_final,
                         "global_error": abs(float(x[-1]) - x_ref_final)})
    df = pd.DataFrame(rows).sort_values(["method","dt"])
    try:
        df.to_parquet(out_parquet)
    except Exception as e:
        # Fallback to CSV if parquet engine missing
        df.to_csv(out_parquet.replace(".parquet", ".csv"), index=False)
    return df

if __name__ == "__main__":
    run()
