"""
Estimators for ODE parameter identification:
- Nonlinear Least Squares (trajectory-based)
- Collocation / derivative-matching
- MAP with Gaussian prior via Laplace approximation
Outputs a JSON summary and saves estimates to parquet.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline

from model import simulate_trajectory, rhs


def load_data(cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        df = pd.read_parquet(cfg["paths"]["results_parquet"])
    except Exception:
        df = pd.read_csv(cfg["paths"]["results_csv"])
    t = df["t"].to_numpy()
    X = df[["x1", "x2"]].to_numpy()
    y = df["y"].to_numpy()
    return t, X, y


def nls_estimate(
    t: np.ndarray, y: np.ndarray, x0: np.ndarray, theta0: np.ndarray
) -> Dict:
    """
    Trajectory NLS: minimize residual between observed y and simulated x1(t; theta).
    """

    def resid(theta):
        params = {"theta1": float(theta[0]), "theta2": float(theta[1])}
        X = simulate_trajectory(t, x0, params)
        return X[:, 0] - y  # model - data

    res = least_squares(
        resid,
        theta0,
        method="trf",
        jac="2-point",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-8,
        max_nfev=200,
    )
    theta_hat = res.x
    r = res.fun
    dof = len(y) - len(theta_hat)
    sigma2 = (r @ r) / max(dof, 1)
    # Approximate covariance via Gauss-Newton
    J = res.jac
    JTJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    out = {
        "theta_hat": theta_hat.tolist(),
        "sigma2_hat": float(sigma2),
        "cov": cov.tolist(),
        "success": bool(res.success),
        "nfev": int(res.nfev),
        "message": res.message,
    }
    return out


def collocation_estimate(
    t: np.ndarray, y: np.ndarray, smoothing: float, theta0: np.ndarray
) -> Dict:
    """
    Derivative matching via cubic smoothing spline on observed angle.
    We approximate x1 ~ s(t), x2 ~ s'(t), and enforce ODE on x2'(t).
    Minimize sum_t [ s''(t) + theta1*sin(s(t)) + theta2*s'(t) ]^2
    """
    # Fit smoothing spline
    s = UnivariateSpline(t, y, s=smoothing * len(t))  # scale smoothing by N
    # Derivatives
    s1 = s.derivative(1)
    s2 = s.derivative(2)

    def residual(theta):
        theta1, theta2 = theta
        r = s2(t) + theta1 * np.sin(s(t)) + theta2 * s1(t)
        return r

    res = least_squares(
        residual,
        theta0,
        method="trf",
        jac="2-point",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
        max_nfev=200,
    )
    theta_hat = res.x
    r = res.fun
    dof = len(y) - len(theta_hat)
    sigma2 = (r @ r) / max(dof, 1)
    J = res.jac
    JTJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    out = {
        "theta_hat": theta_hat.tolist(),
        "sigma2_hat": float(sigma2),
        "cov": cov.tolist(),
        "success": bool(res.success),
        "nfev": int(res.nfev),
        "message": res.message,
    }
    return out


def map_estimate(
    t: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    theta0: np.ndarray,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
) -> Dict:
    """
    Laplace-approximated MAP: minimize ||r(theta)||^2 / (2*sigma^2) + 0.5*(theta-mu0)^T Sigma0^{-1} (theta-mu0).
    In practice, we fold the prior as a Tikhonov-like penalty added to NLS objective.
    We approximate sigma^2 from residuals at the solution (empirical Bayes).
    """
    # We'll implement as an augmented residual vector: [r; L*(theta-mu0)], where L^T L = Sigma0^{-1}
    # Compute Cholesky of Sigma0^{-1}
    iSigma0 = np.linalg.inv(Sigma0)
    # Cholesky of iSigma0 (ensure PD)
    L = np.linalg.cholesky(iSigma0 + 1e-12 * np.eye(2))

    def resid(theta):
        params = {"theta1": float(theta[0]), "theta2": float(theta[1])}
        X = simulate_trajectory(t, x0, params)
        r_data = X[:, 0] - y
        r_prior = L @ (theta - mu0)
        return np.concatenate([r_data, r_prior])

    res = least_squares(
        resid,
        theta0,
        method="trf",
        jac="2-point",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-8,
        max_nfev=300,
    )
    theta_hat = res.x
    r = res.fun
    dof = len(r) - len(theta_hat)
    sigma2 = (r @ r) / max(dof, 1)
    J = res.jac
    JTJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    out = {
        "theta_hat": theta_hat.tolist(),
        "sigma2_hat": float(sigma2),
        "cov": cov.tolist(),
        "success": bool(res.success),
        "nfev": int(res.nfev),
        "message": res.message,
    }
    return out


def main(config_path: str, method: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    t, X, y = load_data(cfg)
    x0 = np.array(cfg["x0"], dtype=float)
    theta0 = np.array(
        [
            cfg["estimation"]["initial_guess"]["theta1"],
            cfg["estimation"]["initial_guess"]["theta2"],
        ],
        dtype=float,
    )

    if method == "nls":
        out = nls_estimate(t, y, x0, theta0)
    elif method == "collocation":
        smoothing = float(cfg["estimation"]["spline_smoothing"])
        out = collocation_estimate(t, y, smoothing, theta0)
    elif method == "map":
        mu0 = np.array(cfg["estimation"]["prior"]["mu"], dtype=float)
        Sigma0 = np.array(cfg["estimation"]["prior"]["Sigma"], dtype=float)
        out = map_estimate(t, y, x0, theta0, mu0, Sigma0)
    else:
        raise ValueError("Unknown method: choose from {nls, collocation, map}")

    summary_path = Path(cfg["paths"]["results_parquet"]).with_suffix(f".{method}.json")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--method", type=str, required=True, choices=["nls", "collocation", "map"]
    )
    args = p.parse_args()
    main(args.config, args.method)
