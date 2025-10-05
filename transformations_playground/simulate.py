"""
simulate.py — Load config, run transformation pipeline, write results to Parquet.

Output schema:
- columns: ["step", "vertex_id", "x", "y"]
- one row per vertex per recorded step
"""
import os
import yaml
import numpy as np
import pandas as pd
from model import pipeline

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main(cfg_path: str = "config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("output_dir", "artifacts")
    record_steps = bool(cfg.get("record_steps", True))
    _ensure_dir(out_dir)

    verts = np.array(cfg["polygon"]["vertices"], dtype=float)
    ops = cfg.get("ops", [])

    A, V_final, records = pipeline(verts, ops, record_steps=record_steps)

    # Flatten records to a dataframe
    rows = []
    for rec in records:
        step = rec["step"]
        V = rec["vertices"]
        for vid, (x, y) in enumerate(V):
            rows.append({"step": step, "vertex_id": vid, "x": float(x), "y": float(y)})
    df = pd.DataFrame(rows).sort_values(["step", "vertex_id"]).reset_index(drop=True)

    df_path = os.path.join(out_dir, "results.parquet")
    df.to_parquet(df_path, index=False)

    # Save the final composite matrix for reproducibility
    import numpy as np
    mat_path = os.path.join(out_dir, "A.npy")
    np.save(mat_path, A)

    print(f"Wrote: {df_path}")
    print(f"Wrote: {mat_path}")

if __name__ == "__main__":
    main()
