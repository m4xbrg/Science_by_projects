import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_time_series(df: pd.DataFrame, fig_dir: str, n: int):
    """
    Plot time series for each IC (expm).
    """
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for ic_id, sub in df.groupby("ic_id"):
        for i in range(n):
            ax.plot(
                sub["t"].to_numpy(),
                sub[f"x{i}"].to_numpy(),
                label=f"IC {ic_id} x{i}(t)",
            )
    ax.set_xlabel("time t")
    ax.set_ylabel("state components")
    ax.legend(loc="best")
    fig.savefig(Path(fig_dir) / "time_series.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phase_portrait(A: np.ndarray, df: pd.DataFrame, fig_dir: str):
    """
    2D phase portrait (only for n==2).
    """
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    if A.shape != (2, 2):
        return
    # Determine plotting bounds from trajectories
    x_min = min(df["x0_0"].min(), df["x0_0"].min()) - 2.0
    y_min = min(df["x0_1"].min(), df["x0_1"].min()) - 2.0
    x_max = max(df["x0_0"].max(), df["x0_0"].max()) + 2.0
    y_max = max(df["x0_1"].max(), df["x0_1"].max()) + 2.0
    xs = np.linspace(x_min, x_max, 25)
    ys = np.linspace(y_min, y_max, 25)
    X, Y = np.meshgrid(xs, ys)
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y

    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, angles="xy")
    for ic_id, sub in df.groupby("ic_id"):
        ax.plot(sub["x0"].to_numpy(), sub["x1"].to_numpy(), label=f"IC {ic_id}")
        # mark start
        ax.plot(sub["x0"].iloc[0], sub["x1"].iloc[0], marker="o")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_xlim([xs.min(), xs.max()])
    ax.set_ylim([ys.min(), ys.max()])
    ax.legend(loc="best")
    fig.savefig(Path(fig_dir) / "phase_portrait.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    A = np.array(cfg["A"], dtype=float)
    df_exp = pd.read_parquet(cfg["output"]["results_path"])
    df_exp = df_exp[df_exp["method"] == "expm"]
    n = A.shape[0]
    plot_time_series(df_exp, cfg["output"]["fig_dir"], n)
    if n == 2:
        # rename columns for plotting convenience
        df_plot = df_exp.rename(columns={"x0": "x0", "x1": "x1"})
        plot_phase_portrait(A, df_plot, cfg["output"]["fig_dir"])


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
