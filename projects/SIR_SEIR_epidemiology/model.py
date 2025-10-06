"""
model.py: SIR/SEIR with interventions and Rt/R0 utilities
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np

Array = np.ndarray


def stepwise_beta(t: float, schedule: List[Tuple[float, float]]) -> float:
    """
    Piecewise-constant beta(t). Picks the latest (time, beta) with time <= t.
    Assumes schedule sorted by time ascending.
    """
    b = schedule[0][1]
    for ti, bi in schedule:
        if t >= ti:
            b = bi
        else:
            break
    return float(b)


@dataclass
class EpidemicParams:
    beta: float
    gamma: float
    sigma: float = 0.0  # used only for SEIR


@dataclass
class EpidemicModel:
    """
    Clean API: rhs(t, x, params) -> dx/dt for SIR or SEIR.
    """

    model_type: str  # "SIR" or "SEIR"
    N: float
    beta_schedule: List[Tuple[float, float]]  # [(t, beta)]
    beta_fn: Callable[[float], float] | None = None

    def __post_init__(self):
        if self.beta_fn is None:
            sched = sorted(self.beta_schedule, key=lambda p: p[0])
            self.beta_fn = lambda t: stepwise_beta(t, sched)

        lt = self.model_type.upper()
        if lt not in {"SIR", "SEIR"}:
            raise ValueError("model_type must be 'SIR' or 'SEIR'")
        self.model_type = lt

    def rhs(self, t: float, x: Array, p: EpidemicParams) -> Array:
        """
        Map math -> code:
        - SIR:
            dS/dt = -beta(t) * S * I / N
            dI/dt =  beta(t) * S * I / N - gamma * I
            dR/dt =  gamma * I
        - SEIR:
            dS/dt = -beta(t) * S * I / N
            dE/dt =  beta(t) * S * I / N - sigma * E
            dI/dt =  sigma * E - gamma * I
            dR/dt =  gamma * I
        """
        beta_t = float(self.beta_fn(t))
        if self.model_type == "SIR":
            S, I, R = x
            dS = -beta_t * S * I / self.N
            dI = beta_t * S * I / self.N - p.gamma * I
            dR = p.gamma * I
            return np.array([dS, dI, dR], dtype=float)
        else:  # SEIR
            S, E, I, R = x
            dS = -beta_t * S * I / self.N
            dE = beta_t * S * I / self.N - p.sigma * E
            dI = p.sigma * E - p.gamma * I
            dR = p.gamma * I
            return np.array([dS, dE, dI, dR], dtype=float)

    # Reproduction numbers
    def R0(self, p: EpidemicParams) -> float:
        """
        Basic reproduction number for constant parameters:
        - SIR:  R0 = beta / gamma
        - SEIR: R0 = beta / gamma  (same under mass-action; latent period shifts timing)
        """
        return float(p.beta / p.gamma)

    def Rt(self, t: float, x: Array, p: EpidemicParams) -> float:
        """
        Instantaneous effective reproduction number:
        Rt(t) = (beta(t) / gamma) * S(t) / N
        """
        S = x[0]
        beta_t = float(self.beta_fn(t))
        return float((beta_t / p.gamma) * (S / self.N))
