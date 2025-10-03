# viz.py — standard visualization entrypoint for the template
from viz_helpers import plot_analysis, plot_value_hist
from function_explorer import FunctionExplorer
import yaml

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    expr = cfg["expression"]
    window = tuple(cfg["window"])
    samples = int(cfg.get("samples", 2000))

    fx = FunctionExplorer(expr)
    res = fx.analyze(window=window, samples=samples)

    plot_analysis(res, title=f"y = {expr}")
    plot_value_hist(res)

if __name__ == "__main__":
    main()
