import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import qr, solve_triangular, svd, cho_factor, cho_solve


@dataclass
class LSConfig:
    add_intercept: bool = True
    solver: str = "qr"  # "qr" or "svd"
    weights: Optional[np.ndarray] = None  # shape (n,) or None
    ridge_lambda: float = 0.0  # ridge penalty >= 0


@dataclass
class LSResult:
    beta: np.ndarray
    y_hat: np.ndarray
    residuals: np.ndarray
    sigma2_hat: float
    H_diag: np.ndarray
    std_resid: np.ndarray
    cooks_d: np.ndarray
    meta: Dict[str, Any]


def _apply_intercept(X: np.ndarray, add_intercept: bool) -> np.ndarray:
    if add_intercept:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])
    return X


def _apply_weights(
    X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if w is None:
        return X, y, None
    w = np.asarray(w)
    assert w.ndim == 1 and w.shape[0] == X.shape[0], "weights must be length n"
    W2 = np.sqrt(w)
    return X * W2[:, None], y * W2, w


def _ridge_augmented_qr(
    X: np.ndarray, y: np.ndarray, ridge_lambda: float
) -> Tuple[np.ndarray, np.ndarray]:
    if ridge_lambda <= 0.0:
        return X, y
    p = X.shape[1]
    lamI = np.sqrt(ridge_lambda) * np.eye(p, dtype=X.dtype)
    X_aug = np.vstack([X, lamI])
    y_aug = np.concatenate([y, np.zeros(p, dtype=y.dtype)])
    return X_aug, y_aug


def _solve_beta(
    X: np.ndarray, y: np.ndarray, solver: str, ridge_lambda: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns beta and optional thin-Q matrix (for leverage computation).
    If solver='qr' we return Q (n×p). For 'svd', return None (compute H_diag via alt route).
    """
    if solver == "qr":
        X_aug, y_aug = _ridge_augmented_qr(X, y, ridge_lambda)
        Q, R, _ = qr(X_aug, mode="economic", pivoting=False)
        beta = solve_triangular(R, Q.T @ y_aug)
        Qn = Q[: X.shape[0], :]
        return beta, Qn
    elif solver == "svd":
        U, S, VT = svd(X, full_matrices=False)
        if ridge_lambda > 0.0:
            S_filt = S / (S**2 + ridge_lambda)
        else:
            tol = np.finfo(S.dtype).eps * max(X.shape) * S.max()
            S_inv = np.where(S > tol, 1.0 / S, 0.0)
            S_filt = S_inv
        beta = VT.T @ (S_filt * (U.T @ y))
        return beta, None
    else:
        raise ValueError("solver must be 'qr' or 'svd'")


def _sigma2_hat(residuals: np.ndarray, n: int, p: int) -> float:
    dof = max(n - p, 1)
    return float((residuals @ residuals) / dof)


def _hat_diag_from_Q(Q: np.ndarray) -> np.ndarray:
    return np.sum(Q * Q, axis=1)


def _hat_diag_from_xtx(X: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    c, lower = cho_factor(XtX, overwrite_a=False, check_finite=True)
    H_diag = np.empty(X.shape[0], dtype=X.dtype)
    for i in range(X.shape[0]):
        xi = X[i, :]
        z = cho_solve((c, lower), xi)
        H_diag[i] = xi @ z
    return H_diag


def fit_least_squares(X: np.ndarray, y: np.ndarray, cfg: LSConfig) -> LSResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    assert X.shape[0] == y.shape[0], "X and y must have same number of rows"

    X = _apply_intercept(X, cfg.add_intercept)
    Xw, yw, w = _apply_weights(X, y, cfg.weights)

    beta, Qn = _solve_beta(Xw, yw, cfg.solver, cfg.ridge_lambda)
    y_hat = X @ beta
    residuals = y - y_hat

    n, p = X.shape
    s2 = _sigma2_hat(residuals, n, p)

    if Qn is not None:
        H_diag = _hat_diag_from_Q(Qn)
    else:
        H_diag = _hat_diag_from_xtx(Xw)

    denom = np.sqrt(np.maximum(1.0 - H_diag, np.finfo(float).eps))
    std_resid = residuals / (np.sqrt(s2) * denom)

    with np.errstate(divide="ignore", invalid="ignore"):
        cooks_d = (residuals**2 / (p * s2)) * (H_diag / (1.0 - H_diag) ** 2)
        cooks_d = np.nan_to_num(cooks_d, nan=np.inf, posinf=np.inf, neginf=np.inf)

    meta = dict(
        n=int(n),
        p=int(p),
        add_intercept=cfg.add_intercept,
        solver=cfg.solver,
        ridge_lambda=float(cfg.ridge_lambda),
    )
    return LSResult(
        beta=beta,
        y_hat=y_hat,
        residuals=residuals,
        sigma2_hat=s2,
        H_diag=H_diag,
        std_resid=std_resid,
        cooks_d=cooks_d,
        meta=meta,
    )


def save_beta_json(beta: np.ndarray, path: str):
    with open(path, "w") as f:
        json.dump({"beta": beta.tolist()}, f, indent=2)
