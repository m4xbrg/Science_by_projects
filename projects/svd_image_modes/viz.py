from pathlib import Path
import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

def plot_quality_curves(results_path: str, figures_dir: str):
    df = pd.read_parquet(results_path)
    plt.figure(); plt.plot(df["k"], df["psnr"], label="PSNR (dB)"); plt.xlabel("Rank k"); plt.ylabel("PSNR (dB)"); plt.title("PSNR vs Rank"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(Path(figures_dir)/"psnr_vs_k.png", dpi=180)
    plt.figure(); plt.plot(df["k"], df["ssim"], label="SSIM"); plt.xlabel("Rank k"); plt.ylabel("SSIM"); plt.title("SSIM vs Rank"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(Path(figures_dir)/"ssim_vs_k.png", dpi=180)
    plt.figure(); plt.plot(df["k"], df["energy_retained"], label="Energy Retained"); plt.xlabel("Rank k"); plt.ylabel("Fraction of Frobenius Energy"); plt.title("Spectral Energy Retention vs Rank"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(Path(figures_dir)/"energy_vs_k.png", dpi=180)

def main(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    plot_quality_curves(cfg["io"]["results_path"], cfg["io"]["figures_dir"])
    print(f"Figures saved to {cfg['io']['figures_dir']}")

if __name__ == "__main__":
    main()

# --- AUTO-ADDED STUBS: uniform visualization entrypoints ---
def plot_primary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # simple line of first numeric column or index
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c; break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    plt.plot(range(len(df[col])), df[col])
    plt.title("Primary Plot (stub)")
    plt.xlabel("index"); plt.ylabel(str(col))
    out = str(Path(outdir) / "primary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out

def plot_secondary(results_path: str, outdir: str) -> str:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(results_path)
    plt.figure()
    # histogram on first numeric column
    col = None
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                col = c; break
        except Exception:
            pass
    if col is None:
        df = df.reset_index()
        col = df.columns[0]
    try:
        plt.hist(df[col], bins=20)
    except Exception:
        plt.plot(range(len(df[col])), df[col])
    plt.title("Secondary Plot (stub)")
    plt.xlabel(str(col)); plt.ylabel("count")
    out = str(Path(outdir) / "secondary.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out

