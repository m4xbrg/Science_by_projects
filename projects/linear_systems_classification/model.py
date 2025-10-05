
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from numpy.typing import ArrayLike
from scipy.linalg import expm, eig

@dataclass
class LinearSystem:
    """
    Linear time-invariant (LTI) system: x'(t) = A x(t).
    Attributes
    ----------
    A : np.ndarray
        System matrix (n x n).
    """
    A: np.ndarray

    def rhs(self, t: float, x: ArrayLike, params: Dict[str, Any] | None = None) -> np.ndarray:
        """
        Right-hand side of the ODE: dx/dt = A x.
        Parameters
        ----------
        t : float
            Time (unused for LTI but kept for API consistency).
        x : ArrayLike
            State vector.
        params : dict | None
            Unused placeholder for API compatibility across projects.
        Returns
        -------
        np.ndarray
            Derivative A @ x.
        """
        x = np.asarray(x)
        return self.A @ x

    def expAt(self, t: float) -> np.ndarray:
        """
        Compute the matrix exponential e^{A t}.
        Parameters
        ----------
        t : float
            Time.
        Returns
        -------
        np.ndarray
            Matrix exponential expm(A * t).
        """
        return expm(self.A * t)

    def propagate_via_expm(self, t: float, x0: ArrayLike) -> np.ndarray:
        """
        Propagate state using x(t) = e^{A t} x0.
        """
        E = self.expAt(t)
        return E @ np.asarray(x0)

    def eigenstructure(self) -> dict:
        """
        Compute eigenvalues/eigenvectors and classify the system.
        Returns
        -------
        dict
            Contains 'eigvals', 'eigvecs', 'diagonalizable', 'types', 'stability'.
        """
        vals, vecs = eig(self.A)
        # Diagonalizable test: eigenvectors matrix must be full rank.
        diag_ok = np.linalg.matrix_rank(vecs) == self.A.shape[0]

        # Stability: based on real parts.
        re = np.real(vals)
        im = np.imag(vals)
        eps = 1e-10

        def _type_2d(vals) -> str:
            if len(vals) != 2:
                return "n/a (type classification implemented for n=2)"
            lam1, lam2 = vals
            r1, r2 = np.real(lam1), np.real(lam2)
            i1, i2 = np.imag(lam1), np.imag(lam2)
            if abs(i1) > eps or abs(i2) > eps:
                if r1 < -eps and r2 < -eps:
                    return "stable spiral (focus)"
                if r1 > eps and r2 > eps:
                    return "unstable spiral (focus)"
                if abs(r1) <= eps and abs(r2) <= eps:
                    return "center (pure rotation)"
                return "spiral saddle (mixed real parts)"
            # real eigenvalues
            if r1 * r2 < -eps:
                return "saddle"
            if r1 < -eps and r2 < -eps:
                return "stable node"
            if r1 > eps and r2 > eps:
                return "unstable node"
            if abs(r1) <= eps and abs(r2) <= eps:
                return "degenerate (line of fixed points / shear)"
            return "mixed / degenerate"

        if np.all(re < -eps):
            stability = "asymptotically stable"
        elif np.all(re <= eps) and np.any(np.abs(im) > eps):
            stability = "marginal (center/rotation)"
        elif np.any(re > eps) and np.any(re < -eps):
            stability = "saddle-like (unstable)"
        elif np.any(re > eps):
            stability = "unstable"
        else:
            stability = "neutral/degenerate"

        type2d = _type_2d(vals)

        return {
            "eigvals": vals,
            "eigvecs": vecs,
            "diagonalizable": bool(diag_ok),
            "types": {"n2_class": type2d},
            "stability": stability,
        }
