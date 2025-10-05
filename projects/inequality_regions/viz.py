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
