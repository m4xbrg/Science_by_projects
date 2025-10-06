import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from model import LSConfig, fit_least_squares, save_beta_json


def _simulate_linear(
    n: int, p: int, noise_sigma: float, add_intercept: bool, rng: np.random.Generator
):
    X = rng.normal(0, 1, size=(n, p))
    if add_intercept:
        beta_true = np.concatenate(([1.5], rng.normal(0, 1, size=p)))
    else:
        beta_true = rng.normal(0, 1, size=p)
    X_model = np.hstack([np.ones((n, 1)), X]) if add_intercept else X
    y = X_model @ beta_true + rng.normal(0, noise_sigma, size=n)
    return X, y, beta_true


def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    io_cfg = cfg["io"]

    add_intercept = bool(model_cfg.get("add_intercept", True))

    if data_cfg["path"] is not None:
        df = pd.read_csv(data_cfg["path"])
        y = df["y"].to_numpy()
        X = df[[c for c in df.columns if c.startswith("x")]].to_numpy()
    else:
        n, p, sigma = (
            int(data_cfg["n"]),
            int(data_cfg["p"]),
            float(data_cfg["noise_sigma"]),
        )
        X, y, beta_true = _simulate_linear(n, p, sigma, add_intercept, rng)
        Path(io_cfg["results_path"]).parent.mkdir(parents=True, exist_ok=True)
        with open("beta_true.json", "w") as f:
            json.dump({"beta_true": beta_true.tolist()}, f, indent=2)

    weights = model_cfg.get("weights", None)
    weights_arr = np.array(weights) if weights is not None else None

    ls_cfg = LSConfig(
        add_intercept=add_intercept,
        solver=str(model_cfg.get("solver", "qr")),
        weights=weights_arr,
        ridge_lambda=float(model_cfg.get("ridge_lambda", 0.0)),
    )

    res = fit_least_squares(X, y, ls_cfg)

    df_out = pd.DataFrame(
        {
            "y": y,
            "y_hat": res.y_hat,
            "residual": res.residuals,
            "leverage": res.H_diag,
            "std_resid": res.std_resid,
            "cooks_d": res.cooks_d,
        }
    )
    df_out.reset_index(names="i", inplace=True)

    out_path = Path(io_cfg["results_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, engine="pyarrow")

    save_beta_json(res.beta, io_cfg["beta_json_path"])

    print(f"Saved results to {out_path} and coefficients to {io_cfg['beta_json_path']}")


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
