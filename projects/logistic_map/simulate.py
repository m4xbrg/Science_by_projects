def run(config_path: str) -> str:
    from pathlib import Path
    import yaml, pandas as pd
    import numpy as np

    cfg = (
        yaml.safe_load(Path(config_path).read_text())
        if Path(config_path).exists()
        else {}
    )
    p = cfg.get("params", {})
    x0 = float(p.get("x0", 0.2))
    r = float(p.get("r", 3.7))
    n = int(p.get("n_steps", 500))
    burn = int(p.get("burn_in", 100))
    xs = [x0]
    for _ in range(n + burn):
        xs.append(r * xs[-1] * (1 - xs[-1]))
    xs = np.array(xs[burn:])
    out = (cfg.get("paths") or {}).get("results", "results.parquet")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": np.arange(len(xs)), "x": xs}).to_parquet(out)
    return str(out)
