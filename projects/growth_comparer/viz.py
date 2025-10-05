"""
Visualization: time series and 1D phase portrait.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from model import rhs

BASE = Path(__file__).resolve().parent
FIGS = BASE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def load_results() -> pd.DataFrame:
    fp = BASE / "results.parquet"
    return pd.read_parquet(fp)

def plot_time_series(df: pd.DataFrame) -> None:
    t = df["t"].to_numpy()
    plt.figure()
    for col in [c for c in df.columns if c.startswith("y_")]:
        plt.plot(t, df[col].to_numpy(), label=col.replace("y_",""))
    plt.xlabel("time")
    plt.ylabel("y(t)")
    plt.legend()
    plt.title("Growth models: time evolution")
    plt.tight_layout()
    plt.savefig(FIGS / "time_series.png", dpi=180)
    plt.close()

def plot_phase_portrait(df: pd.DataFrame, models: Dict[str, dict]) -> None:
    t = df["t"].to_numpy()
    plt.figure()
    for name, params in models.items():
        y = df[f"y_{name}"].to_numpy()
        dy = np.array([rhs(float(tt), float(yy), params, name) for tt, yy in zip(t, y)])
        plt.plot(y, dy, label=name)
    plt.xlabel("y")
    plt.ylabel("dy/dt")
    plt.legend()
    plt.title("Phase portrait (y vs dy/dt)")
    plt.tight_layout()
    plt.savefig(FIGS / "phase_portrait.png", dpi=180)
    plt.close()

def main():
    import yaml
    cfg = yaml.safe_load((BASE / "config.yaml").read_text())
    df = load_results()
    plot_time_series(df)
    plot_phase_portrait(df, cfg["models"])

if __name__ == "__main__":
    main()
