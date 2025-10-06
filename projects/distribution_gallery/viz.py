"""
viz.py — Generate visualizations:
1) Density/PMF overlay with histogram of samples.
2) Empirical CDF vs theoretical CDF.
3) Summary bar for mean error.
"""

from __future__ import annotations
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import make_model


def ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def plot_pdf_or_pmf(ax, model, spec, samples, cfg_plot):
    if spec["type"] == "continuous":
        xs = np.linspace(cfg_plot["x_min"], cfg_plot["x_max"], cfg_plot["num_points"])
        ax.plot(xs, model.pdf(xs), label="PDF")
        ax.hist(samples, bins=60, density=True, alpha=0.3, label="Sample histogram")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"{spec['name'].title()} — PDF & histogram")
    else:
        k = np.arange(cfg_plot["k_min"], cfg_plot["k_max"] + 1)
        ax.stem(k, model.pmf(k), basefmt=" ", label="PMF", use_line_collection=True)
        ax.hist(
            samples,
            bins=np.arange(cfg_plot["k_min"], cfg_plot["k_max"] + 2) - 0.5,
            density=True,
            alpha=0.3,
            label="Sample normalized counts",
        )
        ax.set_xlabel("k")
        ax.set_ylabel("probability")
        ax.set_title(f"{spec['name'].title()} — PMF & normalized counts")
    ax.legend()


def plot_cdf(ax, model, spec, samples, cfg_plot):
    if spec["type"] == "continuous":
        xs = np.linspace(cfg_plot["x_min"], cfg_plot["x_max"], cfg_plot["num_points"])
    else:
        xs = np.arange(cfg_plot["k_min"], cfg_plot["k_max"] + 1)
    ax.plot(xs, model.cdf(xs), label="Theoretical CDF")
    ecx, ecy = ecdf(samples)
    ax.step(ecx, ecy, where="post", label="Empirical CDF")
    ax.set_xlabel("x" if spec["type"] == "continuous" else "k")
    ax.set_ylabel("F(x)")
    ax.set_title(f"{spec['name'].title()} — CDF vs ECDF")
    ax.legend()


def main(cfg_path: str | Path = "config.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    fig_dir = Path(cfg["output"]["fig_dir"])
    fig_dir.mkdir(exist_ok=True)
    for spec in cfg["distributions"]:
        name = spec["name"]
        model = make_model(spec["family"], spec["type"], spec["params"])
        samples = np.load(Path("samples") / f"{name}.npy")
        fig1, ax1 = plt.subplots()
        plot_pdf_or_pmf(ax1, model, spec, samples, spec["plot"])
        fig1.savefig(fig_dir / f"{name}_pdf_pmf.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        fig2, ax2 = plt.subplots()
        plot_cdf(ax2, model, spec, samples, spec["plot"])
        fig2.savefig(fig_dir / f"{name}_cdf.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # Read results (parquet or csv) for mean error
    results_parquet = Path(cfg["output"]["results_parquet"])
    df = None
    if results_parquet.exists():
        try:
            df = pd.read_parquet(results_parquet)
        except Exception:
            pass
    if df is None:
        csv_path = results_parquet.with_suffix(".csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col="name")
    if df is not None:
        fig3, ax3 = plt.subplots()
        ax3.bar(df.index.astype(str), np.abs(df["error_mean"].values))
        ax3.set_xlabel("distribution")
        ax3.set_ylabel("|sample mean − theoretical mean|")
        ax3.set_title("Mean estimation error")
        fig3.savefig(fig_dir / "mean_error.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)


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
