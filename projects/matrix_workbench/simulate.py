import json
import numpy as np
from pathlib import Path
from dataclasses import asdict
import pandas as pd

from model import MatrixWorkbench

def run(matrix: np.ndarray, grid_res: int = 11, compute_3d: bool = True, outdir: str = "results"):
    """
    Execute diagnostics and prepare transformed geometry samples.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    mw = MatrixWorkbench(matrix)
    rep = mw.report(compute_cond=True)

    # store report
    with open(out / "report.json", "w") as f:
        json.dump(asdict(rep), f, indent=2)

    # 2D samples
    if mw.n == 2:
        P, edges = mw.unit_square_grid(grid_res)
        PT = mw.transform_points(P)
        df = pd.DataFrame({"x": P[0], "y": P[1], "xT": PT[0], "yT": PT[1]})
        df.to_parquet(out / "points2d.parquet", index=False)

    # 3D wireframe
    if compute_3d and mw.n == 3:
        edges3 = mw.unit_cube_edges()
        edges3T = mw.transform_edges(edges3)
        rows = []
        for i, e in enumerate(edges3):
            rows.append({"seg": i, "space": "orig", "x": e[0,0], "y": e[1,0], "z": e[2,0]})
            rows.append({"seg": i, "space": "orig", "x": e[0,1], "y": e[1,1], "z": e[2,1]})
        for i, e in enumerate(edges3T):
            rows.append({"seg": i, "space": "trans", "x": e[0,0], "y": e[1,0], "z": e[2,0]})
            rows.append({"seg": i, "space": "trans", "x": e[0,1], "y": e[1,1], "z": e[2,1]})
        pd.DataFrame(rows).to_parquet(out / "edges3d.parquet", index=False)

if __name__ == "__main__":
    # Example default A; override via config.yaml in CLI wrapper if desired
    A = np.array([[1.0, 0.5],[0.2, 1.2]])
    run(A, grid_res=13, compute_3d=False)
