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
