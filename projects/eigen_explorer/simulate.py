import argparse, json, yaml, pathlib
import numpy as np
import pandas as pd
from model import power_method, eig_invariant_axes


def run(config_path: str):
    cfg = yaml.safe_load(open(config_path, "r"))
    A = np.array(cfg["matrix"]["A"], dtype=float)
    x0 = np.array(
        cfg.get(
            "x0", np.random.default_rng(cfg.get("seed", 0)).normal(size=A.shape[0])
        ),
        dtype=float,
    )
    tol = cfg["power_method"]["tol"]
    max_iter = cfg["power_method"]["max_iter"]
    norm = cfg["power_method"].get("normalize", "l2")

    lam_pm, x_pm, hist = power_method(
        A, x0, tol=tol, max_iter=max_iter, norm=norm, return_history=True
    )
    vals, vecs = eig_invariant_axes(A)

    # Reference dominant by magnitude
    idx_dom = int(np.argmax(np.abs(vals)))
    lam_dom = float(vals[idx_dom])
    v_dom = vecs[:, idx_dom].real  # real part for plotting convenience

    # Build diagnostics DataFrame
    rows = []
    for h in hist:
        rows.append(
            {
                "k": h["k"],
                "lambda_pm": h["lambda"],
                "residual_pm": h["residual"],
            }
        )
    df_hist = pd.DataFrame(rows)

    # Final comparison
    comp = {
        "lambda_pm_final": float(lam_pm),
        "lambda_np_dom": lam_dom,
        "angle_between_deg": (
            float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.abs(
                                np.dot(h["x"], v_dom)
                                / (np.linalg.norm(h["x"]) * np.linalg.norm(v_dom))
                            ),
                            0.0,
                            1.0,
                        )
                    )
                )
            )
            if hist
            else None
        ),
    }
    df_eigs = pd.DataFrame(
        {
            "lambda": vals,
            "abs_lambda": np.abs(vals),
        }
    )

    outdir = pathlib.Path(".")
    # Save outputs
    df_hist.to_csv(outdir / "results.csv", index=False)
    with open(outdir / "summary.json", "w") as f:
        json.dump({"comparison": comp, "A": cfg["matrix"]["A"]}, f, indent=2)
    df_eigs.to_csv(outdir / "eigs.csv", index=False)
    print("Wrote results.csv, eigs.csv, summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    run(args.config)
