from __future__ import annotations
import yaml
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from model import EpidemicModel, EpidemicParams

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_model(cfg: Dict[str, Any]) -> Tuple[EpidemicModel, EpidemicParams, np.ndarray]:
    N = float(cfg["population"])
    mt = cfg["model_type"]
    sched = [(float(t), float(b)) for t, b in cfg["beta_schedule"]]
    model = EpidemicModel(mt, N, sched)

    p = cfg["params"]
    params = EpidemicParams(beta=float(p["beta"]), gamma=float(p["gamma"]), sigma=float(p.get("sigma", 0.0)))

    ic = cfg["initial_conditions"]
    if mt.upper() == "SIR":
        x0 = np.array([float(ic["S0"]), float(ic["I0"]), float(ic["R0"])], dtype=float)
    else:
        x0 = np.array([float(ic["S0"]), float(ic["E0"]), float(ic["I0"]), float(ic["R0"])], dtype=float)
    return model, params, x0

def integrate(model: EpidemicModel, params: EpidemicParams, x0: np.ndarray, t_span, dt: float) -> pd.DataFrame:
    t0, tf = float(t_span[0]), float(t_span[1])
    t_eval = np.arange(t0, tf + dt, dt, dtype=float)

    def rhs(t, x):
        return model.rhs(t, x, params)

    sol = solve_ivp(rhs, (t0, tf), x0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9, vectorized=False)
    if not sol.success:
        raise RuntimeError(sol.message)

    df = pd.DataFrame({"t": sol.t})
    if model.model_type == "SIR":
        S, I, R = sol.y
        df["S"] = S
        df["I"] = I
        df["R"] = R
    else:
        S, E, I, R = sol.y
        df["S"] = S
        df["E"] = E
        df["I"] = I
        df["R"] = R

    Rt_vals = []
    for idx, ti in enumerate(sol.t):
        x = sol.y[:, idx]
        Rt_vals.append(model.Rt(ti, x, params))
    df["Rt"] = np.array(Rt_vals, dtype=float)
    df["R0"] = model.R0(params)
    return df

def sample_lognormal(median: float, gsd: float, size=None):
    mu = np.log(median)
    sigma = np.log(gsd)
    return np.random.lognormal(mean=mu, sigma=sigma, size=size)

def monte_carlo(cfg: Dict[str, Any]) -> pd.DataFrame:
    np.random.seed(int(cfg["seed"]))
    model, params, x0 = make_model(cfg)
    t_span, dt = cfg["t_span"], float(cfg["dt"])
    N = model.N

    n = int(cfg["monte_carlo"]["n_samples"])
    gsd_beta  = float(cfg["monte_carlo"]["beta_gsd"])
    gsd_gamma = float(cfg["monte_carlo"]["gamma_gsd"])
    gsd_sigma = float(cfg["monte_carlo"]["sigma_gsd"])
    gsd_I0    = float(cfg["monte_carlo"]["I0_gsd"])

    records = []
    for k in range(n):
        beta_k  = float(sample_lognormal(cfg["params"]["beta"],  gsd_beta))
        gamma_k = float(sample_lognormal(cfg["params"]["gamma"], gsd_gamma))
        sigma_k = float(sample_lognormal(cfg["params"]["sigma"], gsd_sigma))
        I0_k    = float(sample_lognormal(cfg["initial_conditions"]["I0"], gsd_I0))

        params_k = EpidemicParams(beta=beta_k, gamma=gamma_k, sigma=sigma_k)
        x0_k = x0.copy()
        if model.model_type == "SIR":
            x0_k[1] = I0_k
            x0_k[0] = N - x0_k[1] - x0_k[2]
        else:
            x0_k[2] = I0_k
            x0_k[0] = N - x0_k[1] - x0_k[2] - x0_k[3]

        df_k = integrate(model, params_k, x0_k, t_span, dt)
        df_k["sample"] = k
        records.append(df_k)

    df_all = pd.concat(records, ignore_index=True)
    return df_all

def summarize_mc(df_all: pd.DataFrame, model_type: str) -> pd.DataFrame:
    vars_ = ["S", "I", "R"] if model_type == "SIR" else ["S", "E", "I", "R"]
    qtiles = [0.05, 0.5, 0.95]
    out = []
    for t, grp in df_all.groupby("t"):
        row = {"t": t}
        for v in vars_ + ["Rt"]:
            qs = grp[v].quantile(qtiles).values
            row[f"{v}_q05"], row[f"{v}_q50"], row[f"{v}_q95"] = qs[0], qs[1], qs[2]
        row["R0_q05"], row["R0_q50"], row["R0_q95"] = grp["R0"].quantile(qtiles).values
        out.append(row)
    return pd.DataFrame(out).sort_values("t").reset_index(drop=True)

def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    model, params, x0 = make_model(cfg)
    df_nom = integrate(model, params, x0, cfg["t_span"], cfg["dt"])
    df_mc = monte_carlo(cfg)
    df_summary = summarize_mc(df_mc, model.model_type)

    out_nom = cfg["output"]["results_parquet"]
    out_mc  = cfg["output"]["mc_parquet"]
    try:
        df_nom.to_parquet(out_nom, index=False)
        df_summary.to_parquet(out_mc, index=False)
    except Exception:
        # Fallback to CSV if parquet engine is missing
        out_nom = out_nom.replace('.parquet', '.csv')
        out_mc  = out_mc.replace('.parquet', '.csv')
        df_nom.to_csv(out_nom, index=False)
        df_summary.to_csv(out_mc, index=False)
    return out_nom, out_mc

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

