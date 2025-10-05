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
