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

