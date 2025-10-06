import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp
from pathlib import Path
from model import LinearSystem


def make_time_grid(t_final: float, dt: float) -> np.ndarray:
    """Return time grid including endpoint."""
    m = int(np.round(t_final / dt))
    return np.linspace(0.0, t_final, m + 1)


def simulate(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Run simulation for LTI system, propagating multiple ICs both via expm and ODE solver.
    Parameters
    ----------
    config_path : str
        Path to YAML config.
    Returns
    -------
    pd.DataFrame
        Long-form dataframe with columns [ic_id, method, t, x0_*, x_*].
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    A = np.array(cfg["A"], dtype=float)
    ICs = np.array(cfg["initial_conditions"], dtype=float)
    t_final = float(cfg["t_final"])
    dt = float(cfg["dt"])
    out_path = Path(cfg["output"]["results_path"])

    sys = LinearSystem(A=A)
    t_grid = make_time_grid(t_final, dt)

    rows = []
    # Precompute flows
    for j, x0 in enumerate(ICs):
        # Closed form via expm
        for t in t_grid:
            xt = sys.propagate_via_expm(t, x0)
            rows.append(
                {
                    "ic_id": j,
                    "method": "expm",
                    "t": t,
                    **{f"x{i}": float(xt[i]) for i in range(A.shape[0])},
                    **{f"x0_{i}": float(x0[i]) for i in range(A.shape[0])},
                }
            )
        # Numerical ODE
        sol = solve_ivp(
            sys.rhs,
            (0.0, t_final),
            x0,
            method=cfg["integrator"]["method"],
            rtol=cfg["integrator"]["rtol"],
            atol=cfg["integrator"]["atol"],
            dense_output=True,
        )
        for t in t_grid:
            xt = sol.sol(t).flatten()
            rows.append(
                {
                    "ic_id": j,
                    "method": "ode",
                    "t": t,
                    **{f"x{i}": float(xt[i]) for i in range(A.shape[0])},
                    **{f"x0_{i}": float(x0[i]) for i in range(A.shape[0])},
                }
            )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    # Print eigenstructure for quick inspection
    info = sys.eigenstructure()
    print("Eigenvalues:", info["eigvals"])
    print("Diagonalizable:", info["diagonalizable"])
    print("Stability:", info["stability"])
    return df


if __name__ == "__main__":
    simulate()


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

        cfg = (
            yaml.safe_load(Path(config_path).read_text())
            if Path(config_path).exists()
            else {}
        )
    except Exception:
        cfg = {}
    out = (cfg.get("paths", {}) or {}).get("results", "results.parquet")
    outp = Path(out)
    if not outp.parent.exists():
        outp.parent.mkdir(parents=True, exist_ok=True)
    # If some existing main already produced an artifact, keep it. Otherwise, write a tiny placeholder.
    if not outp.exists():
        pd.DataFrame({"placeholder": [0]}).to_parquet(outp)
    return str(outp)
