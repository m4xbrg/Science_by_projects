from pathlib import Path
import yaml

from model import feasible_region_2d, plot_feasible_region, plot_feasibility_heatmap

ROOT = Path(__file__).parent
CONF = yaml.safe_load((ROOT / "config.yaml").read_text())

FIGS = ROOT / "figs"
RESULTS = ROOT / "results"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

def main():
    ineqs = CONF["inequalities"]
    view = CONF["view"]
    grid_n = CONF.get("heatmap_grid_n", 200)
    bbox = CONF["bounding_box"]
    tol = CONF.get("tolerance", 1e-9)

    res = feasible_region_2d(ineqs, add_bbox=bbox, tol=tol)

    # Plots
    plot_feasible_region(res, ineqs, tuple(view), title="Feasible Region", fname=str(FIGS / "feasible_region.png"))
    plot_feasibility_heatmap(ineqs, tuple(view), grid_n=grid_n, fname=str(FIGS / "feasibility_heatmap.png"))
    print("Saved figures to:", FIGS)

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

