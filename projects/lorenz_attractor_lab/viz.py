from __future__ import annotations
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from model import poincare_crossings


def read_df(path: str):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    # Try parquet then csv as fallback
    if os.path.exists("results.parquet"):
        return pd.read_parquet("results.parquet")
    return pd.read_csv("results.csv")


def plot_phase(df, outpath):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz Attractor (Phase Portrait)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_poincare(df, outpath, plane="x=0", direction="increasing"):
    ts = df["t"].to_numpy()
    xs = df[["x", "y", "z"]].to_numpy()
    pts = poincare_crossings(ts, xs, plane=plane, direction=direction)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    if len(pts) > 0:
        if plane.startswith("x="):
            ax.scatter(pts[:, 1], pts[:, 2], s=5)
            ax.set_xlabel("y")
            ax.set_ylabel("z")
        elif plane.startswith("y="):
            ax.scatter(pts[:, 0], pts[:, 2], s=5)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    ax.set_title(f"Poincaré Section on {plane}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_sensitivity(df, outpath, epsilon=1e-6, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    t = df["t"].to_numpy()
    dt = t[1] - t[0]
    X = df[["x", "y", "z"]].to_numpy()
    x0 = X[0].copy()
    x0[0] += epsilon

    def rhs(x):
        return np.array(
            [
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ]
        )

    def rk4_step(x):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    Y = np.zeros_like(X)
    y = x0.copy()
    for i in range(len(t)):
        Y[i] = y
        y = rk4_step(y)
    sep = np.linalg.norm(Y - X, axis=1)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, np.log(sep))
    ax.set_xlabel("t")
    ax.set_ylabel("log ||δx(t)||")
    ax.set_title("Sensitivity to Initial Conditions")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="results.parquet")
    ap.add_argument("--phase_out", type=str, default="figs/phase.png")
    ap.add_argument("--poincare_out", type=str, default="figs/poincare.png")
    ap.add_argument("--sens_out", type=str, default="figs/sensitivity.png")
    ap.add_argument("--plane", type=str, default="x=0")
    ap.add_argument("--direction", type=str, default="increasing")
    ap.add_argument("--epsilon", type=float, default=1e-6)
    args = ap.parse_args()
    df = read_df(args.infile)
    plot_phase(df, args.phase_out)
    plot_poincare(df, args.poincare_out, plane=args.plane, direction=args.direction)
    plot_sensitivity(df, args.sens_out, epsilon=args.epsilon)
    print("Wrote figures.")


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
