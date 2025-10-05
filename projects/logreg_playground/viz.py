import json, os
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_pr(metrics_json_path: str, out_dir: str, dpi: int = 140):
    with open(metrics_json_path, "r") as f:
        M = json.load(f)

    plt.figure(dpi=dpi)
    for name in ["test_raw", "test_cal"]:
        roc = M[name]["roc_curve"]
        plt.plot(roc["fpr"], roc["tpr"], label=f"{name} AUC={M[name]['roc_auc']:.3f}")
    plt.plot([0,1], [0,1], "--", lw=1, label="chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "roc.png"), bbox_inches="tight")
    plt.close()

    plt.figure(dpi=dpi)
    for name in ["test_raw", "test_cal"]:
        pr = M[name]["pr_curve"]
        plt.plot(pr["recall"], pr["precision"], label=f"{name} AUPRC={M[name]['aupr']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves"); plt.legend()
    plt.savefig(os.path.join(out_dir, "pr.png"), bbox_inches="tight")
    plt.close()

def plot_calibration(metrics_json_path: str, out_dir: str, dpi: int = 140):
    with open(metrics_json_path, "r") as f:
        M = json.load(f)

    plt.figure(dpi=dpi)
    for name, ls in [("test_raw", ":"), ("test_cal", "-")]:
        rel = M[name]["reliability"]
        x = np.array(rel["avg_pred"], dtype=float)
        y = np.array(rel["emp_freq"], dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        plt.plot(x[mask], y[mask], ls, marker="o", label=f"{name} (Brier={M[name]['brier']:.3f})")
    plt.plot([0,1],[0,1], "--", lw=1, label="perfect")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Empirical positive frequency")
    plt.title("Reliability Diagram"); plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "calibration.png"), bbox_inches="tight")
    plt.close()

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

