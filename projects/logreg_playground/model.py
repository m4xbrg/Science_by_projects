from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

CalibMethod = Literal["none", "platt", "isotonic"]


@dataclass
class FittedModel:
    scaler: StandardScaler
    clf: LogisticRegression

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities p(y=1|x) before post-hoc calibration."""
        Xt = self.scaler.transform(X)
        return self.clf.predict_proba(Xt)


def fit_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambda_l2: float = 1.0,
    fit_intercept: bool = True,
    max_iter: int = 200,
    solver: str = "lbfgs",
    seed: int = 42,
) -> FittedModel:
    """Fit L2-regularized logistic regression. sklearn uses C = 1/λ."""
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)

    C = 1e12 if lambda_l2 == 0 else 1.0 / lambda_l2
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        solver=solver,
        random_state=seed,
    )
    clf.fit(Xt, y_train)
    return FittedModel(scaler=scaler, clf=clf)


class PlattCalibrator:
    def __init__(self):
        self.a_: float = 1.0
        self.c_: float = 0.0

    @staticmethod
    def _logit(p, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, p_val: np.ndarray, y_val: np.ndarray, max_iter: int = 200):
        from sklearn.linear_model import LogisticRegression

        X = self._logit(p_val).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=max_iter)
        lr.fit(X, y_val)
        self.a_ = float(lr.coef_[0, 0])
        self.c_ = float(lr.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = self.a_ * self._logit(p) + self.c_
        return self._sigmoid(z)


class IsotonicCalibrator:
    def __init__(self):
        self.iso_: Optional[IsotonicRegression] = None

    def fit(self, p_val: np.ndarray, y_val: np.ndarray):
        self.iso_ = IsotonicRegression(out_of_bounds="clip")
        self.iso_.fit(p_val, y_val)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.iso_.transform(p)


def make_calibrator(method: CalibMethod):
    if method == "platt":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    return None
