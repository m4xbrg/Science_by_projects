import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from model import feasible_region_2d

ROOT = Path(__file__).parent
CONF = yaml.safe_load((ROOT / "config.yaml").read_text())

RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

def main():
    ineqs = CONF["inequalities"]
    bbox = CONF["bounding_box"]
    tol = CONF.get("tolerance", 1e-9)

    res = feasible_region_2d(ineqs, add_bbox=bbox, tol=tol)

    # Serialize vertices and raw candidates
    vtx = pd.DataFrame(res["vertices"], columns=["x", "y"])
    raw = pd.DataFrame(res["raw_points"], columns=["x", "y"])

    vtx.to_parquet(RESULTS / "vertices.parquet", index=False)
    raw.to_parquet(RESULTS / "raw_points.parquet", index=False)

    meta = {
        "bounded": bool(res["bounded"]),
        "num_vertices": int(len(res["vertices"])),
        "tolerance": tol,
    }
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2))

    print("Wrote:")
    print(RESULTS / "vertices.parquet")
    print(RESULTS / "raw_points.parquet")
    print(RESULTS / "meta.json")

if __name__ == "__main__":
    main()
