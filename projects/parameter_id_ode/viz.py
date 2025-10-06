"""
Visualization: time series, phase portrait, residuals, and parameter uncertainty bars.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def load_all(cfg):
    try:
        df = pd.read_parquet(cfg["paths"]["results_parquet"])
    except Exception:
        df = pd.read_csv(cfg["paths"]["results_csv"])
    base = Path(cfg["paths"]["results_parquet"])
    outs = {}
    for m in ["nls", "collocation", "map"]:
        p = base.with_suffix(f".{m}.json")
        if p.exists():
            with open(p, "r") as f:
                outs[m] = json.load(f)
    return df, outs


def fig_time_series(df, outdir):
    plt.figure()
    plt.plot(df["t"], df["y"], label="observed y (x1 + noise)")
    plt.plot(df["t"], df["x1"], label="true x1")
    plt.xlabel("time [s]")
    plt.ylabel("angle x1 [rad]")
    plt.legend()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir) / "time_series.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def fig_phase(df, outdir):
    plt.figure()
    plt.plot(df["x1"], df["x2"], label="trajectory")
    plt.xlabel("angle x1 [rad]")
    plt.ylabel("angular velocity x2 [rad/s]")
    plt.legend()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir) / "phase_portrait.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def fig_param_bars(outs, outdir):
    if not outs:
        return None
    labels, theta1, theta2 = [], [], []
    for k, v in outs.items():
        labels.append(k.upper())
        theta1.append(v["theta_hat"][0])
        theta2.append(v["theta_hat"][1])
    x = np.arange(len(labels))
    # theta1
    plt.figure()
    plt.bar(x, theta1, tick_label=labels)
    plt.ylabel("theta1 [s^-2]")
    plt.title("Estimator comparison: theta1")
    path1 = Path(outdir) / "theta1_bars.png"
    plt.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close()
    # theta2
    plt.figure()
    plt.bar(x, theta2, tick_label=labels)
    plt.ylabel("theta2 [s^-1]")
    plt.title("Estimator comparison: theta2")
    path2 = Path(outdir) / "theta2_bars.png"
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    return path1, path2


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    df, outs = load_all(cfg)
    outdir = cfg["paths"]["figs_dir"]
    p1 = fig_time_series(df, outdir)
    p2 = fig_phase(df, outdir)
    p3 = fig_param_bars(outs, outdir)
    print("Saved figures:", p1, p2, p3)


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
                col = c
                break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index")
    plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
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
                col = c
                break
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
    plt.xlabel(str(col))
    plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out
