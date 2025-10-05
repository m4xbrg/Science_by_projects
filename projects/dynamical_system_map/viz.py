import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def load_data(results_path="results.parquet", meta_path="run_meta.json"):
    df = pd.read_parquet(results_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return df, meta

def plot_time_series(df: pd.DataFrame, outdir: str = "figs") -> str:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for col in df.columns:
        if col == "k": continue
        plt.plot(df["k"], df[col], label=col)  # components over k
    # also plot norm
    X = df[[c for c in df.columns if c != "k"]].to_numpy()
    norms = np.linalg.norm(X, axis=1)
    plt.plot(df["k"], norms, linestyle="--", label="||x_k||")
    plt.xlabel("k (steps)")
    plt.ylabel("state / norm (arb. units)")
    plt.legend()
    fname = str(out / "time_series.png")
    plt.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close()
    return fname

def plot_phase_portrait(df: pd.DataFrame, outdir: str = "figs", grid: Optional[int] = 21) -> Optional[str]:
    cols = [c for c in df.columns if c != "k"]
    if len(cols) < 2:
        return None
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # trajectory in (x0, x1)
    plt.figure()
    plt.plot(df[cols[0]], df[cols[1]], label="trajectory")
    plt.scatter([df[cols[0]].iloc[0]], [df[cols[1]].iloc[0]], marker="o", label="start")
    plt.scatter([0], [0], marker="x", label="fixed point")
    plt.xlabel(cols[0]); plt.ylabel(cols[1]); plt.legend()
    fname = str(out / "phase_portrait.png")
    plt.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close()
    return fname

def main():
    df, meta = load_data()
    ts = plot_time_series(df)
    pp = plot_phase_portrait(df)
    print("Wrote:", ts, pp, "Meta:", json.dumps(meta, indent=2))

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

