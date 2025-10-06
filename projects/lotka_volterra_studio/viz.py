def plot_primary(results_path: str, outdir: str) -> str:
    """Time series of prey (x) and predator (y)."""
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    plt.plot(df["t"], df["x"], label="prey x")
    plt.plot(df["t"], df["y"], label="predator y")
    plt.xlabel("time")
    plt.ylabel("population")
    plt.title("Lotka–Volterra: time series")
    plt.legend()
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_secondary(results_path: str, outdir: str) -> str:
    """Phase portrait (x vs y) with simple median guides."""
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)

    # Curve in phase space
    plt.figure()
    plt.plot(df["x"], df["y"])
    plt.xlabel("prey x")
    plt.ylabel("predator y")
    plt.title("Phase portrait")

    # Guides at medians (do not rely on config)
    x_med = float(np.median(df["x"]))
    y_med = float(np.median(df["y"]))
    plt.axhline(y_med, linestyle="--", linewidth=1)
    plt.axvline(x_med, linestyle="--", linewidth=1)

    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out
