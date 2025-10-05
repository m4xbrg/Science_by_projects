"""
Core regression primitives: polynomial features, ridge closed-form, k-fold CV.

API surface:
- poly_features(X, degree, include_bias) -> Phi
- Standardizer: fit/transform to z-score features
- RidgeRegressor: fit/predict with closed-form
- kfold_indices(n, k, seed)
- cross_validate(X, y, degrees, lambdas, k, scale, include_bias, seed) -> DataFrame
"""

from dataclasses import dataclass
from typing import Tuple, Iterable, Optional, Dict, Any
import numpy as np
import pandas as pd

def poly_features(x: np.ndarray, degree: int, include_bias: bool = True) -> np.ndarray:
    """
    Construct 1D polynomial features up to `degree` for vector x.
    For multi-D X, pass shape (n, d) and we build tensor-product monomials only for d==1 (simple studio).
    """
    X = np.atleast_2d(x)
    if X.shape[1] != 1:
        raise ValueError("This studio implements 1D polynomial features for clarity (shape (n,1) expected).")
    n = X.shape[0]
    Phi = np.column_stack([X[:, 0] ** k for k in range(1, degree + 1)])
    if include_bias:
        Phi = np.column_stack([np.ones(n), Phi])
    return Phi

@dataclass
class Standardizer:
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, ddof=0, keepdims=True)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer not fit.")
        return (X - self.mean_) / self.std_

@dataclass
class RidgeRegressor:
    lam: float = 0.0
    w_: Optional[np.ndarray] = None

    def fit(self, Phi: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        n, p = Phi.shape
        # Closed-form: (Phi^T Phi + n*lam I)^-1 Phi^T y
        A = Phi.T @ Phi
        A.flat[::p+1] += n * self.lam  # add n*lam to diagonal efficiently
        self.w_ = np.linalg.solve(A, Phi.T @ y)
        return self

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Model not fit.")
        return Phi @ self.w_

def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    r = y_hat - y
    return float(np.mean(r * r))

def kfold_indices(n: int, k: int, seed: int = 0) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    degrees: Iterable[int],
    lambdas: Iterable[float],
    k: int = 5,
    scale: bool = True,
    include_bias: bool = True,
    seed: int = 0
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with per-fold metrics for each (degree, lambda).
    """
    n = X.shape[0]
    rows = []
    for deg in degrees:
      for lam in lambdas:
        for fold, (tr, va) in enumerate(kfold_indices(n, k, seed), 1):
            Phi_tr = poly_features(X[tr], degree=deg, include_bias=include_bias)
            Phi_va = poly_features(X[va], degree=deg, include_bias=include_bias)
            if scale:
                scaler = Standardizer().fit(Phi_tr)
                Phi_tr = scaler.transform(Phi_tr)
                Phi_va = scaler.transform(Phi_va)
            model = RidgeRegressor(lam=lam).fit(Phi_tr, y[tr])
            yhat_tr = model.predict(Phi_tr)
            yhat_va = model.predict(Phi_va)
            rows.append({
                "degree": deg, "lambda": lam, "fold": fold,
                "mse_train": mse(yhat_tr, y[tr]),
                "mse_val": mse(yhat_va, y[va])
            })
    return pd.DataFrame(rows)

def synthetic_data(n: int, x_range=(-1,1), noise_sigma=0.2, seed: int = 0, func: str = "sin(2*pi*x)") -> Tuple[np.ndarray, np.ndarray, callable]:
    """
    Generate 1D synthetic data with known ground truth function for bias–variance study.
    Returns X (n,1), y (n,), and f(x) callable.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_range[0], x_range[1], size=n)
    def ftrue(xv):
        x = np.asarray(xv)
        return np.sin(2*np.pi*x) if func.strip() == "sin(2*pi*x)" else x**3 - 0.5*x
    y = ftrue(x) + rng.normal(0.0, noise_sigma, size=n)
    return x.reshape(-1,1), y, ftrue

def bias_variance_curves(
    ftrue, x_grid: np.ndarray,
    degrees: Iterable[int], lam: float,
    n_trials: int, n_train: int, noise_sigma: float,
    include_bias: bool = True, scale: bool = True, seed: int = 0
) -> pd.DataFrame:
    """
    Monte Carlo estimate of bias^2 and variance across x_grid for each degree.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for deg in degrees:
        preds = []
        for t in range(n_trials):
            # resample train set
            Xtr = rng.uniform(-1, 1, size=n_train).reshape(-1,1)
            ytr = ftrue(Xtr[:,0]) + rng.normal(0.0, noise_sigma, size=n_train)
            Phi_tr = poly_features(Xtr, degree=deg, include_bias=include_bias)
            scaler = Standardizer().fit(Phi_tr) if scale else None
            if scale:
                Phi_tr = scaler.transform(Phi_tr)
            model = RidgeRegressor(lam=lam).fit(Phi_tr, ytr)

            Phi_te = poly_features(x_grid.reshape(-1,1), degree=deg, include_bias=include_bias)
            if scale:
                Phi_te = scaler.transform(Phi_te)
            preds.append(model.predict(Phi_te))
        P = np.stack(preds, axis=0)  # (T, G)
        mean_pred = P.mean(axis=0)
        var_pred = P.var(axis=0)
        bias2 = (mean_pred - ftrue(x_grid))**2
        for xval, b2, v in zip(x_grid, bias2, var_pred):
            rows.append({"degree": deg, "x": float(xval), "bias2": float(b2), "variance": float(v), "noise": noise_sigma**2})
    return pd.DataFrame(rows)
