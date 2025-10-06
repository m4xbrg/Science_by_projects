"""
Core equations with a clean RHS API: rhs(t, y, params, model)
- linear:      dy/dt = a
- exponential: dy/dt = r * y
- logistic:    dy/dt = r * y * (1 - y / K)
"""

from typing import Mapping, Literal
from dataclasses import dataclass

ModelName = Literal["linear", "exponential", "logistic"]


@dataclass
class LinearParams:
    a: float


@dataclass
class ExponentialParams:
    r: float


@dataclass
class LogisticParams:
    r: float
    K: float


def rhs(t: float, y: float, params: Mapping[str, float], model: ModelName) -> float:
    if model == "linear":
        return float(params["a"])
    if model == "exponential":
        return float(params["r"]) * y
    if model == "logistic":
        return float(params["r"]) * y * (1.0 - y / float(params["K"]))
    raise ValueError(f"Unknown model: {model}")


def validate_params(model: ModelName, params: Mapping[str, float]) -> None:
    required = {"linear": ["a"], "exponential": ["r"], "logistic": ["r", "K"]}[model]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Missing parameters for {model}: {missing}")
