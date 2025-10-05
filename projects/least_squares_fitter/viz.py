import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _load_meta(beta_json_path: str):
    try:
        with open(beta_json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def residuals_vs_fitted(df: pd.DataFrame, figs_dir: Path, thr_std: float):
    plt.figure()
    plt.scatter(df["y_hat"], df["residual"], alpha=0.8)
    plt.axhline(0.0)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    flag = np.abs(df["std_resid"].to_numpy()) >= thr_std
    if flag.any():
        plt.scatter(df.loc[flag, "y_hat"], df.loc[flag, "residual"])
    out = figs_dir / "residuals_vs_fitted.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out

def leverage_plot(df: pd.DataFrame, p: int, figs_dir: Path):
    plt.figure()
    x = df["leverage"].to_numpy()
    y = (df["std_resid"].to_numpy())**2
    plt.scatter(x, y, alpha=0.8)
    plt.xlabel("Leverage h_ii")
    plt.ylabel("Standardized residual^2")
    plt.title("Leverage Plot")
    n = len(df)
    for k in (2, 3):
        thr = k * p / n
        plt.axvline(thr)
    out = figs_dir / "leverage_plot.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    res_path = cfg["io"]["results_path"]
    figs_dir = Path(cfg["plots"]["figs_dir"])
    thr_std = float(cfg["plots"]["residual_flag_threshold_std"])
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(res_path)
    beta_meta = json.load(open(cfg["io"]["beta_json_path"])) if Path(cfg["io"]["beta_json_path"]).exists() else {}
    p = len(beta_meta.get("beta", []))

    f1 = residuals_vs_fitted(df, figs_dir, thr_std)
    f2 = leverage_plot(df, p, figs_dir)
    print(f"Saved {f1} and {f2}")

if __name__ == "__main__":
    main()

# --- AUTO-ADDED STUBS: uniform visualization entrypoints ---
def plot_primary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # simple line of first numeric column or index
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c; break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index"); plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out

def plot_secondary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # histogram on first numeric column
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c; break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    try:
        plt.hist(df[col], bins=20)
    except Exception:
        plt.plot(range(len(df[col])), df[col])
    plt.title("Secondary Plot (stub)")
    plt.xlabel(str(col)); plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out

