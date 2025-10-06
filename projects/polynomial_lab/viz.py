from __future__ import annotations
import yaml, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from model import PolynomialLab


def plot_real_curve_with_roots(
    lab: PolynomialLab,
    x_min=-5,
    x_max=5,
    num=1000,
    root_tol=1e-8,
    out_path="figs/real_curve.png",
):
    xs = np.linspace(x_min, x_max, num)
    ys = np.real(lab.polyval(xs))
    plt.figure()
    plt.plot(xs, ys, label="P(x)")
    plt.axhline(0, linestyle="--", linewidth=1)
    rr = lab.roots_companion().roots
    real_roots = [r.real for r in rr if abs(r.imag) < root_tol]
    if real_roots:
        plt.scatter(real_roots, [0] * len(real_roots), marker="x", label="real roots")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.legend()
    plt.title("Polynomial Curve with Real Roots")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def plot_complex_roots(roots: np.ndarray, out_path="figs/complex_roots.png"):
    plt.figure()
    plt.scatter(np.real(roots), np.imag(roots))
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Complex Root Map")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    coeffs = cfg["polynomial"]["coeffs"]
    io = cfg["io"]
    viz_cfg = cfg["viz"]
    lab = PolynomialLab(coeffs)
    res = lab.roots_companion()

    plot_real_curve_with_roots(
        lab,
        x_min=viz_cfg["real_curve"]["x_min"],
        x_max=viz_cfg["real_curve"]["x_max"],
        num=viz_cfg["real_curve"]["num"],
        root_tol=viz_cfg["real_curve"]["root_tol"],
        out_path=io["real_curve_png"],
    )
    plot_complex_roots(res.roots, out_path=io["complex_map_png"])

    r_div = viz_cfg.get("synthetic_division_r", None)
    if r_div is not None:
        _, _, b = lab.synthetic_division(r_div)
        plt.figure()
        plt.plot(np.arange(len(b)), np.real(b), marker="o")
        plt.xlabel("k (Horner index)")
        plt.ylabel("Re(b_k)")
        plt.title(f"Horner Table (synthetic division) at r={r_div}")
        Path("figs").mkdir(parents=True, exist_ok=True)
        plt.savefig("figs/horner_table.png", bbox_inches="tight", dpi=160)
        plt.close()


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
