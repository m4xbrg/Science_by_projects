"""
Run approximations based on config.yaml and dump parquet outputs.
- Single run: results_parquet with x, y, p_cheb, p_taylor, err_* and Emax in attrs.
- Optional sweep: sweep_parquet with columns [n, Emax_cheb, Emax_taylor].
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from model import ApproxConfig, compare_approximations, sweep_degree

PRESETS = {
    "exp_cos3": (lambda x: np.exp(x) * np.cos(3.0*x)),
    "runge": (lambda x: 1.0/(1.0 + 25.0*x**2)),
    "abs": (lambda x: np.abs(x)),
    "logistic": (lambda x: 1.0/(1.0 + np.exp(-5.0*x))),
}

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_function(spec: dict):
    kind = spec.get("kind", "preset")
    if kind == "preset":
        name = spec.get("name", "exp_cos3")
        if name not in PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Valid: {list(PRESETS.keys())}")
        return PRESETS[name], name
    elif kind == "expr":
        expr = spec.get("expr", "").strip()
        if not expr:
            raise ValueError("function.kind='expr' requires non-empty 'expr' field.")
        # Build a vectorized lambda x: <expr>
        def f_vec(x):
            return eval(expr, {"np": np, "__builtins__": {}}, {"x": x})
        return f_vec, f"expr:{expr}"
    else:
        raise ValueError("function.kind must be 'preset' or 'expr'.")

def main():
    cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)

    a, b = map(float, cfg["interval"])
    c = float(cfg["taylor_center"])
    n = int(cfg["degree"])
    n_list = cfg.get("degree_list", None)

    f_vec, f_label = get_function(cfg["function"])

    N_eval = int(cfg["N_eval"])
    N_fit = cfg.get("N_fit", None)
    h0 = float(cfg["fd_h0"])
    levels = int(cfg["fd_richardson_levels"])

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / cfg.get("results_parquet", "results.parquet")
    sweep_path = out_dir / cfg.get("sweep_parquet", "sweep.parquet")

    # Single run
    acfg = ApproxConfig(a=a, b=b, n=n, c=c, N_eval=N_eval, h0=h0, richardson_levels=levels, N_fit=N_fit)
    res = compare_approximations(f_vec, acfg)

    df = pd.DataFrame({
        "x": res["x"],
        "f": res["y"],
        "p_cheb": res["p_cheb"],
        "p_taylor": res["p_taylor"],
        "err_cheb": res["err_cheb"],
        "err_taylor": res["err_taylor"],
    })
    # Store metadata in a sidecar JSON
    meta = {
        "function": f_label,
        "interval": [a, b],
        "center": c,
        "degree": n,
        "N_eval": N_eval,
        "N_fit": N_fit,
        "fd_h0": h0,
        "fd_richardson_levels": levels,
        "Emax_cheb": res["Emax_cheb"],
        "Emax_taylor": res["Emax_taylor"],
    }
    df.to_parquet(results_path, index=False)
    with open(out_dir / "results_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Optional sweep
    if n_list:
        Echeb, Etaylor = sweep_degree(f_vec, a, b, c, n_list, N_eval=N_eval)
        df_sweep = pd.DataFrame({"n": n_list, "Emax_cheb": Echeb, "Emax_taylor": Etaylor})
        df_sweep.to_parquet(sweep_path, index=False)

    print(f"Wrote {results_path}")
    if n_list:
        print(f"Wrote {sweep_path}")

if __name__ == "__main__":
    main()
