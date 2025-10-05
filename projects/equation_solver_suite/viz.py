from __future__ import annotations
import math
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
from model import gaussian_elimination, solve_quadratic

Number = Union[float, complex]

def residual(A: List[List[Number]], x: List[Number], b: List[Number]) -> List[Number]:
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]

def plot_linear_residual_trace(A, b, eps=1e-12, save_path="figs/linear_residual_trace.png"):
    res = gaussian_elimination(A, b, eps=eps, pivoting=True, record_snapshots=True)
    if res.status != "ok":
        print("Linear solve failed:", res.status); return
    residuals = []
    x_partial = [None] * len(res.solution)
    for step in res.steps:
        if step.get("phase") == "back_sub":
            i = step["i"]
            x_partial[i] = step["x_i"]
            x_est = [xi if xi is not None else 0 for xi in x_partial]
            r = residual(A, x_est, b)
            r2 = math.sqrt(sum((ri.real if isinstance(ri, complex) else ri)**2 for ri in r))
            residuals.append(r2)
    plt.figure()
    plt.plot(range(1, len(residuals)+1), residuals, marker='o')
    plt.xlabel("Back-substitution step")
    plt.ylabel("Residual norm ||Ax - b||")
    plt.title("Gaussian Elimination: Residual vs Step")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight"); plt.close()

def plot_quadratic_sensitivity(a=1.0, c=1.0, b_min=-4.0, b_max=4.0, num=201, save_path="figs/quadratic_sensitivity.png"):
    import numpy as np
    bs = np.linspace(b_min, b_max, num)
    xr1, xi1, xr2, xi2 = [], [], [], []
    for b in bs:
        sol = solve_quadratic(a, b, c)
        (x1, x2) = sol.solution
        xr1.append((x1.real if isinstance(x1, complex) else x1))
        xi1.append((x1.imag if isinstance(x1, complex) else 0.0))
        xr2.append((x2.real if isinstance(x2, complex) else x2))
        xi2.append((x2.imag if isinstance(x2, complex) else 0.0))
    plt.figure()
    plt.plot(bs, xr1, label="Re(x1)")
    plt.plot(bs, xr2, label="Re(x2)")
    plt.xlabel("b"); plt.ylabel("Real part of roots")
    plt.title("Quadratic Roots (Real Parts) vs b")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(bs, xi1, label="Im(x1)")
    plt.plot(bs, xi2, label="Im(x2)")
    plt.axhline(0, linestyle="--")
    plt.xlabel("b"); plt.ylabel("Imag part of roots")
    plt.title("Quadratic Roots (Imag Parts) vs b")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path.replace(".png", "_imag.png"), bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    A = [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]]
    b = [1.0, -2.0, 0.0]
    plot_linear_residual_trace(A, b)
    plot_quadratic_sensitivity()
    print("Saved figures to figs/")

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

