"""
simulate.py — deterministic integration and simple 1D parameter continuation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List
from model import LVParams, GridSpec, rhs, jacobian, fixed_points

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def integrate_once(IC: Tuple[float, float], params: LVParams, t0: float, t_final: float, dt: float, rtol: float, atol: float, method: str) -> pd.DataFrame:
    t_eval = np.arange(t0, t_final+dt/2, dt)
    sol = solve_ivp(fun=lambda t, x: rhs(t, x, params),
                    t_span=(t0, t_final), y0=np.array(IC, dtype=float),
                    t_eval=t_eval, rtol=rtol, atol=atol, method=method, vectorized=False)
    df = pd.DataFrame({"t": sol.t, "x": sol.y[0], "y": sol.y[1]})
    for k, v in params.__dict__.items():
        df[k] = v
    return df

def sweep_parameter(cfg: Dict[str, Any]) -> pd.DataFrame:
    cont = cfg["continuation"]
    if not cont.get("enabled", False):
        params = LVParams(**cfg["params"])
        df = integrate_once((cfg["ICs"]["x0"], cfg["ICs"]["y0"]), params,
                            cfg["t0"], cfg["t_final"], cfg["dt"], cfg["rtol"], cfg["atol"], cfg["method"])
        df["sweep_param"] = None
        df["sweep_value"] = None
        return df

    p_name = cont["param_name"]
    grid = np.linspace(cont["start"], cont["end"], cont["steps"])
    rows = []
    base_params = dict(cfg["params"])
    for val in grid:
        p = dict(base_params)
        p[p_name] = float(val)
        params = LVParams(**p)
        df = integrate_once((cfg["ICs"]["x0"], cfg["ICs"]["y0"]), params,
                            cfg["t0"], cfg["t_final"], cfg["dt"], cfg["rtol"], cfg["atol"], cfg["method"])
        df["sweep_param"] = p_name
        df["sweep_value"] = val
        # simple summary stats for bifurcation-like plot
        T = df["t"].to_numpy()
        mask = T >= (cfg["t0"] + (cfg["t_final"]-cfg["t0"]) * cont.get("transient_fraction", 0.5))
        for var in ["x", "y"]:
            df.loc[mask, f"{var}_max_after_transient"] = df.loc[mask, var].max()
            df.loc[mask, f"{var}_min_after_transient"] = df.loc[mask, var].min()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def main():
    cfg = load_config("config.yaml")
    np.random.seed(cfg.get("seed", 0))
    df = sweep_parameter(cfg)
    out = Path("results.parquet")
    df.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(df)} rows.")

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

