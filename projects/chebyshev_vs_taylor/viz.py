"""
Visualization for Chebyshev vs Taylor project.
Generates:
- figs/approximants.png : f(x), Chebyshev, Taylor
- figs/errors.png       : |error| curves with Emax in legend
- figs/convergence.png  : (optional) Emax vs degree n (if sweep_parquet present)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_approximants(df: pd.DataFrame, out_path: Path, title: str = ""):
    x = df["x"].to_numpy()
    y = df["f"].to_numpy()
    p_cheb = df["p_cheb"].to_numpy()
    p_taylor = df["p_taylor"].to_numpy()

    plt.figure()
    plt.plot(x, y, label="f(x)")
    plt.plot(x, p_cheb, label="Chebyshev")
    plt.plot(x, p_taylor, label="Taylor")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.title(title if title else "Function and Approximants")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_errors(df: pd.DataFrame, meta: dict, out_path: Path, title: str = ""):
    x = df["x"].to_numpy()
    e_cheb = np.abs(df["err_cheb"].to_numpy())
    e_taylor = np.abs(df["err_taylor"].to_numpy())
    plt.figure()
    plt.plot(
        x,
        e_cheb,
        label=f"|error| Chebyshev (Emax={meta.get('Emax_cheb', np.max(e_cheb)):.3e})",
    )
    plt.plot(
        x,
        e_taylor,
        label=f"|error| Taylor (Emax={meta.get('Emax_taylor', np.max(e_taylor)):.3e})",
    )
    plt.xlabel("x")
    plt.ylabel("absolute error")
    plt.title(title if title else "Absolute Error on [a,b]")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_convergence(df_sweep: pd.DataFrame, out_path: Path):
    n = df_sweep["n"].to_numpy()
    e_cheb = df_sweep["Emax_cheb"].to_numpy()
    e_taylor = df_sweep["Emax_taylor"].to_numpy()
    plt.figure()
    plt.semilogy(n, e_cheb, marker="o", label="Chebyshev Emax")
    plt.semilogy(n, e_taylor, marker="s", label="Taylor Emax")
    plt.xlabel("degree n")
    plt.ylabel("max abs error on [a,b]")
    plt.title("Convergence of Max Error vs Degree")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    base = Path(".")
    out_dir = base / "results"
    figs = base / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    res_path = out_dir / "results.parquet"
    meta_path = out_dir / "results_meta.json"
    sweep_path = out_dir / "sweep.parquet"

    if not res_path.exists():
        raise SystemExit(f"Missing {res_path}. Run simulate.py first.")

    df = pd.read_parquet(res_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    plot_approximants(df, figs / "approximants.png")
    plot_errors(df, meta, figs / "errors.png")

    if sweep_path.exists():
        df_sweep = pd.read_parquet(sweep_path)
        plot_convergence(df_sweep, figs / "convergence.png")

    print(f"Figures written to {figs}/")


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
