def run(config_path: str) -> str:
    from pathlib import Path
    import yaml, pandas as pd, numpy as np
    from scipy.integrate import solve_ivp
    cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    p = cfg.get("params", {})
    omega = float(p.get("omega", 2.0))
    zeta  = float(p.get("zeta", 0.1))
    t_final = float(p.get("t_final", 20.0))
    n = int(p.get("n_points", 2000))
    ic = cfg.get("ics", {"x0":1.0,"v0":0.0})
    x0, v0 = float(ic["x0"]), float(ic["v0"])
    def rhs(t, y):
        x, v = y
        return [v, -2*zeta*omega*v - omega**2*x]
    t = np.linspace(0, t_final, n)
    sol = solve_ivp(rhs, [0, t_final], [x0, v0], t_eval=t, rtol=1e-7, atol=1e-9)
    out = (cfg.get("paths") or {}).get("results", "results.parquet")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": sol.t, "x": sol.y[0], "v": sol.y[1]}).to_parquet(out)
    return str(out)
