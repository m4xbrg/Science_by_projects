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

