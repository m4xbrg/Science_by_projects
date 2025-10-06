from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import gram_schmidt


def _load_results(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def plot_loss_vs_cond(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.scatter(df["cond2_A"], df["ortho_frob"])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("cond₂(A)")
    plt.ylabel("‖QᵀQ − I‖_F")
    plt.title("Loss of orthogonality vs conditioning")
    plt.grid(True, which="both", ls=":")
    plt.savefig(os.path.join(outdir, "loss_vs_cond.png"), dpi=160)
    plt.close()


def plot_method_compare(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    grp = df.groupby(["method", "reorth"])["ortho_frob"].median().reset_index()
    labels = [f"{m.upper()}(r={r})" for m, r in zip(grp["method"], grp["reorth"])]
    vals = grp["ortho_frob"].values
    plt.figure()
    plt.bar(np.arange(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(np.arange(len(vals)), labels, rotation=0)
    plt.ylabel("median ‖QᵀQ − I‖_F (log)")
    plt.title("Method comparison (median across trials)")
    plt.grid(True, axis="y", ls=":")
    plt.savefig(os.path.join(outdir, "method_compare.png"), dpi=160)
    plt.close()


def animate_gs(dim: int, outpath: str):
    rng = np.random.default_rng(0)
    m, n = dim, min(3, dim)
    A = rng.standard_normal((m, n))
    steps = []
    Q = np.zeros_like(A)
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            rij = np.dot(Q[:, i], v)
            v = v - rij * Q[:, i]
            steps.append((v.copy(), Q[:, : j + 0].copy()))
        rjj = np.linalg.norm(v)
        Q[:, j] = v / rjj
        steps.append((Q[:, j].copy(), Q[:, : j + 1].copy()))

    if dim == 2:
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.axhline(0, lw=0.5)
        ax.axvline(0, lw=0.5)

        def update(k):
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal")
            ax.axhline(0, lw=0.5)
            ax.axvline(0, lw=0.5)
            v, Qcur = steps[k]
            ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1)
            for i in range(Qcur.shape[1]):
                q = Qcur[:, i]
                ax.quiver(0, 0, q[0], q[1], angles="xy", scale_units="xy", scale=1)
            ax.set_title(f"MGS step {k+1}/{len(steps)}")
            return []

        ani = FuncAnimation(fig, update, frames=len(steps), blit=False, repeat=False)
        ani.save(outpath, fps=2)
        plt.close(fig)

    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        def update(k):
            ax.cla()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
            v, Qcur = steps[k]
            ax.quiver(0, 0, 0, v[0], v[1], v[2])
            for i in range(Qcur.shape[1]):
                q = Qcur[:, i]
                ax.quiver(0, 0, 0, q[0], q[1], q[2])
            ax.set_title(f"MGS step {k+1}/{len(steps)}")
            return []

        ani = FuncAnimation(fig, update, frames=len(steps), blit=False, repeat=False)
        ani.save(outpath, fps=2)
        plt.close(fig)
    else:
        raise ValueError("dim must be 2 or 3.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="results/results.parquet")
    ap.add_argument("--outdir", type=str, default="figs")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--dim", type=int, default=2)
    args = ap.parse_args()

    df = _load_results(args.results)
    plot_loss_vs_cond(df, args.outdir)
    plot_method_compare(df, args.outdir)

    if args.animate:
        os.makedirs(args.outdir, exist_ok=True)
        outpath = os.path.join(args.outdir, "gs_animation.mp4")
        animate_gs(args.dim, outpath)


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
