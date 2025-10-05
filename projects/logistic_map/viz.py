def plot_primary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd, matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    plt.plot(df["t"], df["x"])
    plt.xlabel("t"); plt.ylabel("x_t"); plt.title("Logistic Map: Time Series")
    out = str(Path(outdir)/"primary.png"); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close(); return out

def plot_secondary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd, numpy as np, matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    xs = df["x"].to_numpy()
    # simple cobweb against x_{t+1} = r x (1-x), estimate r by regression on parabola apex scale is messy; display x vs shifted x
    plt.figure()
    plt.scatter(xs[:-1], xs[1:], s=6)
    plt.xlabel("x_t"); plt.ylabel("x_{t+1}"); plt.title("Cobweb Scatter (proxy)")
    out = str(Path(outdir)/"secondary.png"); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close(); return out
