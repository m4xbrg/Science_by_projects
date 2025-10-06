def run(config_path: str) -> str:
    """
    Uniform entrypoint.
    Reads YAML config, integrates SIR or SEIR ODEs, writes results.parquet, returns its path.
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

    mode = (cfg.get("mode") or "SIR").upper()  # "SIR" or "SEIR"
    p = cfg.get("params", {})  # infection params
    N = float(p.get("N", 1_000.0))
    beta = float(p.get("beta", 0.3))  # transmission rate
    gamma = float(p.get("gamma", 0.1))  # recovery rate
    sigma = float(p.get("sigma", 0.2))  # incubation (SEIR only)

    t_final = float(cfg.get("t_final", 160.0))
    n_points = int(cfg.get("n_points", 1600))
    t_eval = np.linspace(0.0, t_final, n_points)

    # Initial conditions
    ics = cfg.get("ics", {})
    I0 = float(ics.get("I0", 1.0))
    E0 = float(ics.get("E0", 0.0))
    R0 = float(ics.get("R0", 0.0))
    S0 = float(N - I0 - R0 - E0)

    if mode == "SEIR":
        y0 = [S0, E0, I0, R0]

        def rhs(t, y):
            S_comp, E_comp, I_comp, R_comp = y
            dS = -beta * S_comp * I_comp / N
            dE = beta * S_comp * I_comp / N - sigma * E_comp
            dI = sigma * E_comp - gamma * I_comp
            dR = gamma * I_comp
            return [dS, dE, dI, dR]

        labels = ["S", "E", "I", "R"]
    else:
        y0 = [S0, I0, R0]

        def rhs(t, y):
            S_comp, I_comp, R_comp = y
            dS = -beta * S_comp * I_comp / N
            dI = beta * S_comp * I_comp / N - gamma * I_comp
            dR = gamma * I_comp
            return [dS, dI, dR]

        labels = ["S", "I", "R"]

    sol = solve_ivp(rhs, [0.0, t_final], y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    data = {"t": sol.t}
    for i, name in enumerate(labels):
        data[name] = sol.y[i]
    # Derived metrics (basic reproduction number R0, effective Rt)
    data["N"] = N
    R0_basic = beta / gamma if gamma > 0 else float("inf")
    data["R0_basic"] = np.full_like(sol.t, R0_basic, dtype=float)
    # Rt = R0 * S/N (for SIR; in SEIR same approximation with S compartment)
    S_series = data["S"]
    data["Rt"] = (
        (beta / gamma) * (S_series / N)
        if gamma > 0
        else np.full_like(S_series, float("inf"))
    )

    df = pd.DataFrame(data)
    df.to_parquet(out)
    return str(out)
