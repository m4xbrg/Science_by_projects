import pandas as pd
import matplotlib.pyplot as plt
import os

def _read_summary(path_parquet: str):
    try:
        return pd.read_parquet(path_parquet)
    except Exception:
        alt = path_parquet.replace('.parquet', '.csv')
        return pd.read_csv(alt)

def plot_time_series(mc_summary_path: str, model_type: str = "SIR", outpath: str = "figs/time_series.png"):
    df = _read_summary(mc_summary_path)

    plt.figure()
    if model_type.upper() == "SEIR":
        for var in ["E", "I"]:
            plt.fill_between(df["t"], df[f"{var}_q05"], df[f"{var}_q95"], alpha=0.2, label=f"{var} 90% CI")
            plt.plot(df["t"], df[f"{var}_q50"], label=f"{var} median")
    else:
        var = "I"
        plt.fill_between(df["t"], df[f"{var}_q05"], df[f"{var}_q95"], alpha=0.2, label=f"{var} 90% CI")
        plt.plot(df["t"], df[f"{var}_q50"], label=f"{var} median")

    plt.xlabel("Time (days)")
    plt.ylabel("Persons")
    plt.title("Infectious population with uncertainty ribbons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_phase_portrait(mc_summary_path: str, model_type: str = "SIR", outpath: str = "figs/phase_portrait.png"):
    df = _read_summary(mc_summary_path)
    plt.figure()
    plt.plot(df["S_q50"], df["I_q50"])
    plt.xlabel("S (median)")
    plt.ylabel("I (median)")
    plt.title("Phase portrait: S vs I (median trajectory)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

if __name__ == "__main__":
    plot_time_series("results_mc.parquet", "SIR", "figs/time_series.png")
    plot_phase_portrait("results_mc.parquet", "SIR", "figs/phase_portrait.png")

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

