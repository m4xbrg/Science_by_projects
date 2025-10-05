import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from model import MapParams, step, eig_info

def run_simulation(config_path: str = "config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)

    A = np.array(cfg["A"], dtype=float)
    x0 = np.array(cfg["x0"], dtype=float)
    steps = int(cfg.get("steps", 200))
    outdir = Path(cfg.get("output_dir", "figs"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Eigenstructure & classification
    info = eig_info(A, save_vectors=bool(cfg.get("save_eigenvectors", False)))
    with open("run_meta.json", "w") as f:
        json.dump(info, f, indent=2)

    # Simulate
    params = MapParams(A=A)
    n = x0.size
    X = np.empty((steps + 1, n), dtype=float)
    X[0] = x0
    for k in range(steps):
        X[k + 1] = step(k, X[k], params)

    # Persist
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n)])
    df.insert(0, "k", np.arange(steps + 1))
    df.to_parquet("results.parquet", index=False)

    print("Simulation complete.")
    print("Eigen info:", json.dumps(info, indent=2))
    print("Saved: results.parquet, run_meta.json")

if __name__ == "__main__":
    run_simulation()
