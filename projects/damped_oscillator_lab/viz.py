def plot_primary(results_path: str, outdir: str) -> str:
    """Primary: x(t) time series."""
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    plt.plot(df["t"], df["x"])
    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title("Damped Oscillator: Time Series")
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_secondary(results_path: str, outdir: str) -> str:
    """Secondary: phase portrait v vs x."""
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    plt.plot(df["x"], df["v"])
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Phase Portrait")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out
