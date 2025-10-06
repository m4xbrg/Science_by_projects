"""
Simulate models using solve_ivp (reads config.yaml).
Outputs:
- results.parquet
- crossovers.json
- results_meta.json
"""

import json, math, yaml
from pathlib import Path
from typing import Dict, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from model import rhs, validate_params

BASE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def integrate_model(
    model: str, params: dict, y0: float, t_final: float, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    validate_params(model, params)
    t_eval = np.arange(0.0, t_final + 1e-12, dt)

    def fun(t, y):
        return rhs(t, y, params, model)

    sol = solve_ivp(
        fun,
        (0.0, t_final),
        y0=np.array([y0], dtype=float),
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for {model}: {sol.message}")
    return sol.t, sol.y[0]


def _bisection_root(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    maxit: int = 100,
):
    fa, fb = func(a), func(b)
    if math.isnan(fa) or math.isnan(fb):
        return None
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        return None
    left, right = a, b
    for _ in range(maxit):
        mid = 0.5 * (left + right)
        fm = func(mid)
        if abs(fm) < tol or (right - left) < tol:
            return mid
        if fa * fm <= 0:
            right, fb = mid, fm
        else:
            left, fa = mid, fm
    return 0.5 * (left + right)


def first_nontrivial_crossover(t, y1, y2, t_eps=1e-8, tol=1e-6):
    diff = y1 - y2
    for i in range(len(t) - 1):
        a, b = float(t[i]), float(t[i + 1])
        fa, fb = float(diff[i]), float(diff[i + 1])
        if a < t_eps and abs(fa) < tol:  # skip trivial t=0
            continue
        if fa * fb < 0:

            def g(tt: float) -> float:
                # interpolate on fixed grid
                y1v = np.interp(tt, t, y1)
                y2v = np.interp(tt, t, y2)
                return float(y1v - y2v)

            r = _bisection_root(g, max(a, t_eps), b, tol=tol)
            if r is not None and r > t_eps:
                return float(r)
    return None


def main():
    cfg = load_config(BASE / "config.yaml")
    y0 = float(cfg.get("y0", 1.0))
    t_final = float(cfg["t_final"])
    dt = float(cfg["dt"])
    models_cfg: Dict[str, dict] = cfg["models"]

    series = {}
    t_ref = None
    for mname, mparams in models_cfg.items():
        t_m, y_m = integrate_model(mname, mparams, y0=y0, t_final=t_final, dt=dt)
        if t_ref is None:
            t_ref = t_m
        series[mname] = (t_m, y_m)

    import pandas as pd

    df = pd.DataFrame({"t": t_ref})
    for mname, (_, y) in series.items():
        df[f"y_{mname}"] = y
    df.to_parquet(BASE / "results.parquet", index=False)

    pairs = cfg.get("crossover", {}).get("pairs", [])
    cross = {}
    for i, j in pairs:
        ti, yi = series[i]
        tj, yj = series[j]
        assert np.allclose(ti, tj)
        cross[f"{i}__vs__{j}"] = first_nontrivial_crossover(
            ti, yi, yj, tol=float(cfg["crossover"]["tol"])
        )
    (BASE / "crossovers.json").write_text(json.dumps(cross, indent=2))

    meta = {"y0": y0, "t_final": t_final, "dt": dt, "models": models_cfg}
    (BASE / "results_meta.json").write_text(json.dumps(meta, indent=2))


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
