from dataclasses import dataclass
import json, os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

from model import fit_logreg, make_calibrator

@dataclass
class Config:
    seed: int
    dataset: Dict
    splits: Dict
    model: Dict
    calibration: Dict
    evaluation: Dict
    io: Dict
    viz: Dict

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def make_synthetic(dataset_cfg: Dict, seed: int):
    typ = dataset_cfg["type"]
    n = dataset_cfg["n_samples"]
    noise = dataset_cfg.get("noise", 0.2)
    if typ == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif typ == "circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=seed)
    elif typ == "blobs":
        X, y = make_blobs(
            n_samples=n, centers=2,
            cluster_std=noise if noise > 0 else 1.0,
            n_features=dataset_cfg.get("n_features", 2),
            center_box=(-dataset_cfg.get("class_sep", 1.0), dataset_cfg.get("class_sep", 1.0)),
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown dataset type: {typ}")
    return X, y

def evaluate_all(y_true, p, thresholds, n_bins) -> Dict:
    auc = roc_auc_score(y_true, p)
    aupr = average_precision_score(y_true, p)
    ll = log_loss(y_true, np.c_[1-p, p])
    brier = brier_score_loss(y_true, p)

    fpr, tpr, roc_thr = roc_curve(y_true, p)
    prec, rec, pr_thr = precision_recall_curve(y_true, p)

    conf = {}
    for t in thresholds:
        y_hat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        conf[str(t)] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    avg_pred, emp_freq, bin_count = [], [], []
    for k in range(n_bins):
        mask = idx == k
        if np.any(mask):
            avg_pred.append(float(p[mask].mean()))
            emp_freq.append(float(y_true[mask].mean()))
            bin_count.append(int(mask.sum()))
        else:
            avg_pred.append(float("nan"))
            emp_freq.append(float("nan"))
            bin_count.append(0)

    return {
        "roc_auc": float(auc),
        "aupr": float(aupr),
        "log_loss": float(ll),
        "brier": float(brier),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": roc_thr.tolist()},
        "pr_curve": {"recall": rec.tolist(), "precision": prec.tolist(), "thr": pr_thr.tolist()},
        "threshold_confusions": conf,
        "reliability": {"avg_pred": avg_pred, "emp_freq": emp_freq, "bin_count": bin_count, "bins": bins.tolist()},
    }

def run(config_path: str = "config.yaml") -> Dict:
    cfg = load_config(config_path)
    seed = cfg.seed

    X, y = make_synthetic(cfg.dataset, seed)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=cfg.splits["train"], random_state=seed, stratify=y)
    rel = cfg.splits["val"] / (cfg.splits["val"] + cfg.splits["test"])
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, train_size=rel, random_state=seed, stratify=y_tmp)

    fm = fit_logreg(
        X_tr, y_tr,
        lambda_l2=cfg.model["lambda_l2"],
        fit_intercept=cfg.model["fit_intercept"],
        max_iter=cfg.model["max_iter"],
        solver=cfg.model["solver"],
        seed=seed
    )

    from model import make_calibrator
    p_tr = fm.predict_proba(X_tr)[:, 1]
    p_va = fm.predict_proba(X_va)[:, 1]
    p_te = fm.predict_proba(X_te)[:, 1]

    method = cfg.calibration["method"]
    cal = make_calibrator(method)
    if cal is not None:
        cal.fit(p_va, y_va)
        p_te_cal = cal.transform(p_te)
    else:
        p_te_cal = p_te

    metrics = {
        "train": evaluate_all(y_tr, p_tr, cfg.evaluation["thresholds"], cfg.evaluation["n_bins"]),
        "val":   evaluate_all(y_va, p_va, cfg.evaluation["thresholds"], cfg.evaluation["n_bins"]),
        "test_raw": evaluate_all(y_te, p_te, cfg.evaluation["thresholds"], cfg.evaluation["n_bins"]),
        "test_cal": evaluate_all(y_te, p_te_cal, cfg.evaluation["thresholds"], cfg.evaluation["n_bins"]),
    }

    out_dir = cfg.io["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        "split": ["train", "val", "test_raw", "test_cal"],
        "roc_auc": [metrics[s]["roc_auc"] for s in metrics],
        "aupr": [metrics[s]["aupr"] for s in metrics],
        "log_loss": [metrics[s]["log_loss"] for s in metrics],
        "brier": [metrics[s]["brier"] for s in metrics],
    })
    df.to_parquet(os.path.join(out_dir, cfg.io["parquet_name"]))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({"y_test": y_te, "p_test_raw": p_te, "p_test_cal": p_te_cal})
    pred_df.to_parquet(os.path.join(out_dir, "test_predictions.parquet"))
    return {"metrics": metrics, "paths": {"summary": os.path.join(out_dir, cfg.io["parquet_name"])}}

if __name__ == "__main__":
    run()
