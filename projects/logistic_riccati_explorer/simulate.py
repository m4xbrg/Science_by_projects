"""
Simulation driver for Logistic/Riccati models.
- Reads YAML config
- Integrates with scipy.integrate.solve_ivp
- Optionally augments system for logistic sensitivities
- Writes a tidy results.parquet for downstream viz
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from model import rhs_logistic, rhs_riccati, logistic_analytic, sensitivities_logistic_rhs

def _load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _t_eval_from_span(span, num=500):
    t0, tf = float(span[0]), float(span[1])
    return np.linspace(t0, tf, num=num)

def _integrate_logistic(cfg: dict, tag: dict) -> pd.DataFrame:
    r = cfg["logistic"]["r"]
    K = cfg["logistic"]["K"]
    x0 = float(cfg["x0"])
    t_span = cfg["t_span"]
    t_eval = cfg["t_eval"] if cfg["t_eval"] is not None else _t_eval_from_span(t_span)

    params = {"r": r, "K": K}
    # Base integration
    sol = solve_ivp(lambda t, x: rhs_logistic(t, x, params),
                    t_span, [x0], method=cfg["solver"], t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"])

    # Analytic solution at same grid
    x_analytic = logistic_analytic(sol.t, x0, params)

    df = pd.DataFrame({
        "t": sol.t,
        "x_numeric": sol.y[0],
        "x_analytic": x_analytic,
        "model": "logistic",
        "r": r, "K": K,
        **tag
    })

    # Sensitivities (forward) if enabled
    if cfg["sensitivities"]["enabled"]:
        wrt = tuple(cfg["sensitivities"]["with_respect_to"])
        y0 = [x0] + [0.0] * len(wrt)  # ∂x/∂p(0)=0
        sol_s = solve_ivp(lambda t, y: sensitivities_logistic_rhs(t, y, params, wrt),
                          t_span, y0, method=cfg["solver"], t_eval=t_eval,
                          rtol=cfg["rtol"], atol=cfg["atol"])
        for i, name in enumerate(wrt, start=1):
            df[f"s_{name}"] = sol_s.y[i]

    return df

def _integrate_riccati(cfg: dict, tag: dict) -> pd.DataFrame:
    a = cfg["riccati"]["a"]
    b = cfg["riccati"]["b"]
    c = cfg["riccati"]["c"]
    x0 = float(cfg["x0"])
    t_span = cfg["t_span"]
    t_eval = cfg["t_eval"] if cfg["t_eval"] is not None else _t_eval_from_span(t_span)

    params = {"a": a, "b": b, "c": c}
    sol = solve_ivp(lambda t, x: rhs_riccati(t, x, params),
                    t_span, [x0], method=cfg["solver"], t_eval=t_eval,
                    rtol=cfg["rtol"], atol=cfg["atol"])

    # No general closed form included here
    df = pd.DataFrame({
        "t": sol.t,
        "x_numeric": sol.y[0],
        "x_analytic": np.nan,
        "model": "riccati",
        "a": a, "b": b, "c": c,
        **tag
    })
    return df

def main(config_path: str | Path = "config.yaml", out_path: str | Path = "results.parquet"):
    cfg = _load_config(config_path)

    rng = np.random.default_rng(cfg["seed"])
    sweep_r = cfg["sweep"].get("r", [])
    sweep_K = cfg["sweep"].get("K", [])

    dfs = []
    if cfg["model"] == "logistic":
        if sweep_r or sweep_K:
            r_vals = sweep_r if sweep_r else [cfg["logistic"]["r"]]
            K_vals = sweep_K if sweep_K else [cfg["logistic"]["K"]]
            for r in r_vals:
                for K in K_vals:
                    # override params for this run
                    cfg_run = json.loads(json.dumps(cfg))
                    cfg_run["logistic"]["r"] = float(r)
                    cfg_run["logistic"]["K"] = float(K)
                    tag = {"run_id": f"r={r:.4g}_K={K:.4g}"}
                    dfs.append(_integrate_logistic(cfg_run, tag))
        else:
            dfs.append(_integrate_logistic(cfg, {"run_id": "default"}))
    elif cfg["model"] == "riccati":
        dfs.append(_integrate_riccati(cfg, {"run_id": "default"}))
    else:
        raise ValueError("Unknown model")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with shape {df_all.shape}")

if __name__ == "__main__":
    main()
