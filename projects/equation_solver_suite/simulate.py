from __future__ import annotations
import json, yaml
import numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, Any
from model import gaussian_elimination, solve_quadratic


def run(config_path: str = "config.yaml") -> Path:
    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    out_path = Path(paths.get("results", "data/results.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    Lin = cfg["linear_example"]
    A = np.array(Lin["A"], dtype=float).tolist()
    b = np.array(Lin["b"], dtype=float).tolist()
    res_lin = gaussian_elimination(
        A,
        b,
        eps=float(Lin["eps"]),
        pivoting=bool(Lin["pivoting"]),
        record_snapshots=True,
    )

    Quad = cfg["quadratic_example"]
    res_quad = solve_quadratic(
        float(Quad["a"]), float(Quad["b"]), float(Quad["c"]), eps=float(Quad["eps"])
    )

    df = pd.DataFrame(
        [
            {
                "example": "linear",
                "status": res_lin.status,
                "solution": json.dumps(res_lin.solution),
                "steps": json.dumps(res_lin.steps),
            },
            {
                "example": "quadratic",
                "status": res_quad.status,
                "solution": json.dumps(res_quad.solution),
                "steps": json.dumps(res_quad.steps),
            },
        ]
    )
    df.to_parquet(out_path, index=False)

    cfg_out = Path("data/config_used.yaml")
    with open(cfg_out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out_path


if __name__ == "__main__":
    out = run()
    print(f"Wrote {out}")
