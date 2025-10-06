def run(config_path: str) -> str:
    """
    Integrate damped oscillator:
        x'' + 2*zeta*omega*x' + omega^2*x = 0
    State-space:
        dx/dt = v
        dv/dt = -2*zeta*omega*v - omega^2*x
    Writes Parquet with columns: t, x, v; returns the path.
    """
    from pathlib import Path
    import yaml
    import pandas as pd
    import numpy as np
    from scipy.integrate import solve_ivp

    cfg = (
        yaml.safe_load(Path(config_path).read_text())
        if Path(config_path).exists()
        else {}
    )
    paths = cfg.get("paths") or {}
    out = paths.get("results", "results.parquet")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    p = cfg.get("params", {})
    omega = float(p.get("omega", 2.0))
    zeta = float(p.get("zeta", 0.1))
    t_final = float(cfg.get("t_final", 20.0))
    n_points = int(cfg.get("n_points", 2000))
    t = np.linspace(0.0, t_final, n_points)

    ics = cfg.get("ics", {"x0": 1.0, "v0": 0.0})
    x0 = float(ics.get("x0", 1.0))
    v0 = float(ics.get("v0", 0.0))

    def rhs(_t, y):
        x, v = y
        return [v, -2.0 * zeta * omega * v - (omega**2) * x]

    sol = solve_ivp(rhs, [0.0, t_final], [x0, v0], t_eval=t, rtol=1e-8, atol=1e-10)

    df = pd.DataFrame({"t": sol.t, "x": sol.y[0], "v": sol.y[1]})
    df.to_parquet(out)
    return str(out)
