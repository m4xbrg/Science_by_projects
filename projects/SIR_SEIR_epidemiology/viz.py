def plot_primary(results_path: str, outdir: str) -> str:
    """
    Primary: time series of compartments S, (E), I, R.
    """
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    cols = [c for c in ["S", "E", "I", "R"] if c in df.columns]
    plt.figure()
    for c in cols:
        plt.plot(df["t"], df[c], label=c)
    plt.xlabel("time")
    plt.ylabel("population")
    plt.title("Epidemic compartments over time")
    plt.legend()
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_secondary(results_path: str, outdir: str) -> str:
    """
    Secondary: phase-like view (I vs S) if available; else stacked area of compartments.
    """
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    if "S" in df.columns and "I" in df.columns:
        plt.figure()
        plt.plot(df["S"], df["I"])
        plt.xlabel("S")
        plt.ylabel("I")
        plt.title("Phase portrait: I vs S")
        out = str(Path(outdir) / "secondary.png")
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        plt.close()
        return out
    # fallback: stacked area
    cols = [c for c in ["S", "E", "I", "R"] if c in df.columns]
    plt.figure()
    plt.stackplot(df["t"], *[df[c] for c in cols], labels=cols)
    plt.xlabel("time")
    plt.ylabel("population")
    plt.title("Stacked compartments")
    plt.legend(loc="upper right", ncols=2, fontsize=8)
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out
