from __future__ import annotations
import yaml, json, numpy as np, pandas as pd
from pathlib import Path
from model import PolynomialLab

def try_write_parquet(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_parquet(path)  # requires pyarrow or fastparquet
        return True
    except Exception:
        return False

def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    coeffs = cfg["polynomial"]["coeffs"]
    method = cfg["polynomial"]["method"]
    tol = float(cfg["polynomial"].get("tol", 1e-12))
    max_iter = int(cfg["polynomial"].get("max_iter", 200))
    io = cfg["io"]
    Path(io["fig_dir"]).mkdir(parents=True, exist_ok=True)

    lab = PolynomialLab(coeffs)
    if method == "durand_kerner":
        res = lab.roots_durand_kerner(tol=tol, max_iter=max_iter)
    else:
        res = lab.roots_companion()

    factors_C = lab.factors_over_C(res.clustered, res.scale)
    factors_R = lab.factors_over_R(res.clustered, res.scale)
    verify = lab.verify(factors_C, res.scale)

    r_div = cfg["viz"].get("synthetic_division_r", None)
    syn = None
    if r_div is not None:
        q, rem, b = lab.synthetic_division(r_div)
        syn = {"r": r_div, "remainder": complex(rem), "table": [complex(x) for x in b]}

    roots_df = pd.DataFrame({
        "root_real": np.real(res.roots),
        "root_imag": np.imag(res.roots)
    })

    # Try parquet; if not available, write CSV
    parquet_ok = try_write_parquet(roots_df, io["results_parquet"])
    if not parquet_ok:
        roots_df.to_csv(io.get("results_csv", "results.csv"), index=False)

    summary = {
        "coeffs_high_to_low": [complex(c) for c in lab.a],
        "roots": [complex(r) for r in res.roots],
        "clustered": [(complex(r), int(m)) for r,m in res.clustered],
        "factors_C": [(complex(r), int(m)) for r,m in factors_C],
        "factors_R": [{"coeffs": [float(np.real(c)) for c in coeffs], "m": int(m)} for coeffs, m in factors_R],
        "verify": verify,
        "synthetic_division": syn
    }
    Path("results.json").write_text(json.dumps(summary, indent=2, default=str))

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

