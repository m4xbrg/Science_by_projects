def run(config_path: str) -> str:
    """
    Integrate classical Lotka–Volterra ODEs:
        dx/dt =  alpha*x - beta*x*y
        dy/dt = -gamma*y + delta*x*y
    Writes a Parquet with columns: t, x (prey), y (predator), and returns its path.
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

    p = cfg.get("params", {})  # defaults follow standard textbook scales
    alpha = float(p.get("alpha", 1.0))  # prey birth
    beta = float(p.get("beta", 0.1))  # predation
    gamma = float(p.get("gamma", 1.5))  # predator death
    delta = float(p.get("delta", 0.075))  # predator growth from prey

    t_final = float(cfg.get("t_final", 60.0))
    n_points = int(cfg.get("n_points", 2000))
    t_eval = np.linspace(0.0, t_final, n_points)

    ics = cfg.get("ics", {})
    x0 = float(ics.get("x0", 10.0))
    y0 = float(ics.get("y0", 5.0))

    def rhs(_t, z):
        x, y = z
        return [alpha * x - beta * x * y, -gamma * y + delta * x * y]

    sol = solve_ivp(rhs, [0.0, t_final], [x0, y0], t_eval=t_eval, rtol=1e-9, atol=1e-11)

    df = pd.DataFrame({"t": sol.t, "x": sol.y[0], "y": sol.y[1]})
    # Optional invariant surrogate (not exact const due to numerics)
    C = (
        delta * np.log(np.clip(df["x"], 1e-12, None))
        - beta * np.log(np.clip(df["y"], 1e-12, None))
        - gamma * df["x"]
        - alpha * df["y"]
    )
    df["invariant"] = C.to_numpy()
    df.to_parquet(out)
    return str(out)
