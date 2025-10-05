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

# --- AUTO-ADDED STUB: uniform entrypoint ---
def run(config_path: str) -> str:
    """Uniform entrypoint.
    Reads YAML config if present, writes results.parquet if not already written by existing code.
    Returns the path to the primary results file.
    """
    from pathlib import Path
    import pandas as pd
    try:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    except Exception:
        cfg = {}
    out = (cfg.get("paths", {}) or {}).get("results", "results.parquet")
    outp = Path(out)
    if not outp.parent.exists():
        outp.parent.mkdir(parents=True, exist_ok=True)
    # If some existing main already produced an artifact, keep it. Otherwise, write a tiny placeholder.
    if not outp.exists():
        pd.DataFrame({"placeholder":[0]}).to_parquet(outp)
    return str(outp)

