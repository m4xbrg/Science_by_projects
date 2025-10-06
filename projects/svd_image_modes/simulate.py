from pathlib import Path
import pandas as pd, numpy as np, yaml
from model import (
    load_image,
    to_float,
    dynamic_range,
    per_channel_svd,
    reconstruct_from_factors,
    mse,
    psnr,
    ssim_mean,
    energy_retained,
)


def run(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    recon_dir = Path(cfg["io"]["recon_dir"])
    figures_dir = Path(cfg["io"]["figures_dir"])
    results_path = Path(cfg["io"]["results_path"])
    recon_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    X = to_float(load_image(cfg["image_path"], cfg["color_mode"]))
    L = dynamic_range(X)
    svd_res = per_channel_svd(
        X,
        cfg["svd"]["method"],
        cfg["svd"]["k_max"],
        cfg["svd"].get("oversample", 10),
        cfg["svd"].get("n_power_iter", 1),
        cfg["seed"],
    )
    rows = []
    for k in cfg["ks"]:
        Xk = reconstruct_from_factors(svd_res, k, X.shape)
        import numpy as np

        Xk = np.clip(Xk, 0, L)
        m = mse(X, Xk)
        p = psnr(X, Xk, L)
        s = ssim_mean(X, Xk, L)
        e = energy_retained(svd_res, k)
        rows.append({"k": k, "mse": m, "psnr": p, "ssim": s, "energy_retained": e})
        if cfg["io"]["save_reconstructions"]:
            from PIL import Image

            Xsave = np.clip(Xk, 0, 255).astype(np.uint8)
            Image.fromarray(Xsave if Xsave.ndim == 2 else Xsave).save(
                recon_dir / f"recon_k={k:03d}.png"
            )
    pd.DataFrame(rows).sort_values("k").to_parquet(results_path, index=False)


if __name__ == "__main__":
    run()
