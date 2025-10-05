"""
Simulation script: integrates the damped oscillator and writes results.parquet.

Usage:
    python simulate.py --config config.yaml
"""
from __future__ import annotations
import argparse
from typing import Dict

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import yaml

from model import rhs, energy, damping_ratio


def load_config(path: str) -> Dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_time_grid(t_final: float, dt: float) -> np.ndarray:
    """Create a uniform time grid including endpoints."""
    n = int(np.ceil(t_final / dt)) + 1
    return np.linspace(0.0, t_final, n)


def simulate(params: Dict[str, float], IC: Dict[str, float], t_final: float, dt: float,
             solver: Dict, paths: Dict) -> pd.DataFrame:
    """Run the simulation and return a DataFrame with t, x, v, E.

    Args:
        params: {'m','c','k'}
        IC: {'x0','v0'}
        t_final: end time [s]
        dt: output interval [s]
        solver: {'rtol','atol','method','max_step'}
        paths: {'results', 'figs'} (figs optional here)

    Returns:
        DataFrame with columns ['t','x','v','E'].
    """
    t_eval = make_time_grid(t_final, dt)
    y0 = np.array([IC["x0"], IC["v0"]], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=(0.0, t_final),
        y0=y0,
        t_eval=t_eval,
        method=solver.get("method", "RK45"),
        rtol=solver.get("rtol", 1e-8),
        atol=solver.get("atol", 1e-10),
        max_step=solver.get("max_step", np.inf),
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    x = sol.y[0]
    v = sol.y[1]
    E = energy(x, v, params)

    df = pd.DataFrame({"t": sol.t, "x": x, "v": v, "E": E})
    out = paths.get("results", "results.parquet")
    df.to_parquet(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    params = cfg["params"]
    IC = cfg["IC"]
    t_final = float(cfg["t_final"])
    dt = float(cfg["dt"])
    solver = cfg.get("solver", {})
    paths = cfg.get("paths", {"results": "results.parquet"})

    zeta = damping_ratio(params)
    regime = "under" if zeta < 1 else ("critical" if abs(zeta-1) < 1e-12 else "over")
    print(f"Damping ratio zeta = {zeta:.4f} ({regime}-damped)")

    df = simulate(params, IC, t_final, dt, solver, paths)
    print(f"Wrote {len(df)} rows to {paths.get('results', 'results.parquet')}")


if __name__ == "__main__":
    main()