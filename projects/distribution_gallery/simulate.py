"""
simulate.py — Run sampling for configured distributions, compute sample moments,
and save tidy results to CSV (Parquet if available).
"""

from __future__ import annotations
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from model import make_model


def sample_moments(x: np.ndarray) -> dict:
    """Compute sample moments: mean, variance (unbiased), skew, excess kurtosis."""
    n = len(x)
    mean = x.mean()
    var = x.var(ddof=1)
    centered = x - mean
    m2 = (centered**2).mean()
    m3 = (centered**3).mean()
    m4 = (centered**4).mean()
    skew = m3 / (m2**1.5 + 1e-12)
    kurt_excess = m4 / (m2**2 + 1e-12) - 3.0
    return {
        "n": n,
        "mean": float(mean),
        "var": float(var),
        "skew": float(skew),
        "kurtosis_excess": float(kurt_excess),
    }


def run(config_path: str | Path) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    rng = np.random.default_rng(cfg.get("seed", 0))
    rows = []
    all_samples = {}
    for spec in cfg["distributions"]:
        name = spec["name"]
        model = make_model(spec["family"], spec["type"], spec["params"])
        n = int(spec.get("n_samples", 10000))
        x = model.rvs(n, rng)
        all_samples[name] = x
        sm = sample_moments(x)
        tm = model.theoretical_moments()
        row = {"name": name, **sm}
        for k, v in tm.items():
            row[f"theory_{k}"] = v
            row[f"error_{k}"] = row.get(k, np.nan) - v if k in row else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("name")
    # Try Parquet first; fallback to CSV
    out_parquet = Path(cfg["output"].get("results_parquet", "results.parquet"))
    try:
        df.to_parquet(out_parquet)  # requires pyarrow or fastparquet
        out_path = out_parquet
    except Exception:
        out_csv = out_parquet.with_suffix(".csv")
        df.to_csv(out_csv)
        out_path = out_csv
    # Cache samples for plotting as .npy per dist
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    for name, x in all_samples.items():
        np.save(samples_dir / f"{name}.npy", x)
    return out_path


if __name__ == "__main__":
    out = run("config.yaml")
    print(f"Wrote {out.resolve()}")
