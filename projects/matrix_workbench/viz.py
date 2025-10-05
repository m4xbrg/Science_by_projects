import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model import MatrixWorkbench

def plot_2d(A: np.ndarray, grid_res: int = 11, outdir: str = "figs"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    mw = MatrixWorkbench(A)

    P, edges = mw.unit_square_grid(grid_res)
    PT = mw.transform_points(P)
    E = edges
    ET = mw.transform_edges(E)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for e in E:
        ax.plot(e[0], e[1], linewidth=0.8, alpha=0.6)
    for e in ET:
        ax.plot(e[0], e[1], linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Unit Square under A")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(out / "square_transform.png", dpi=160)

    U, s, Vt = np.linalg.svd(A)
    theta = np.linspace(0, 2*np.pi, 300)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(circle[0], circle[1], linestyle="--", linewidth=0.8)
    ax2.plot(ellipse[0], ellipse[1], linewidth=1.2)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Unit Circle mapped by A")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig2.savefig(out / "circle_to_ellipse.png", dpi=160)

def plot_3d(A: np.ndarray, outdir: str = "figs"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    mw = MatrixWorkbench(A)
    edges = mw.unit_cube_edges()
    edgesT = mw.transform_edges(edges)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for e in edges:
        ax.plot(e[0], e[1], e[2], linewidth=0.8, alpha=0.6)
    for e in edgesT:
        ax.plot(e[0], e[1], e[2], linewidth=1.2)
    ax.set_title("Unit Cube under A")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.savefig(out / "cube_transform.png", dpi=160)

if __name__ == "__main__":
    A2 = np.array([[1.0, 0.5],[0.2, 1.2]])
    plot_2d(A2, 13)
    A3 = np.array([[1.2, 0.2, 0.0],[0.0, 0.9, -0.3],[0.1, 0.4, 1.1]])
    plot_3d(A3)

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

