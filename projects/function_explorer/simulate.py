# simulate.py — reads config.yaml, runs analysis, writes results.parquet
import yaml
import pandas as pd
from function_explorer import FunctionExplorer

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    expr = cfg["expression"]
    window = tuple(cfg["window"])
    samples = int(cfg.get("samples", 2000))
    tol = float(cfg.get("tol", 1e-9))

    fx = FunctionExplorer(expr)
    result = fx.analyze(window=window, samples=samples, tol=tol)

    # Save samples for downstream analysis
    df = pd.DataFrame({ "X": result.samples["X"], "Y": result.samples["Y"] })
    df.to_parquet("results.parquet", index=False)
    print("Saved analysis results to results.parquet")

if __name__ == "__main__":
    main()
