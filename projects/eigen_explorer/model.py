import numpy as np
from typing import Dict, Tuple, List


def rayleigh_quotient(x: np.ndarray, A: np.ndarray) -> float:
    """Return Rayleigh quotient rho(x) = (x^T A x) / (x^T x)."""
    num = float(x.T @ (A @ x))
    den = float(x.T @ x)
    return num / den


def normalize(x: np.ndarray, norm: str = "l2") -> np.ndarray:
    """Normalize a vector with the specified norm."""
    if norm == "l2":
        s = np.linalg.norm(x)
    elif norm == "l1":
        s = np.sum(np.abs(x))
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    if s == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return x / s


def power_method(
    A: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 1000,
    norm: str = "l2",
    return_history: bool = True,
) -> Tuple[float, np.ndarray, List[Dict]]:
    """Compute the dominant eigenpair of A via the power method.

    Parameters
    ----------
    A : np.ndarray
        Square matrix (n x n).
    x0 : np.ndarray
        Initial vector (n, ) with nonzero component along dominant eigenvector.
    tol : float
        Residual tolerance on ||A x - lambda x||.
    max_iter : int
        Maximum number of iterations.
    norm : str
        Normalization ('l2' or 'l1').
    return_history : bool
        If True, return iteration diagnostics.

    Returns
    -------
    (lambda_hat, x_hat, history)
    """
    A = np.asarray(A, dtype=float)
    x = normalize(np.asarray(x0, dtype=float), norm=norm)
    history = []
    for k in range(1, max_iter + 1):
        y = A @ x
        x = normalize(y, norm=norm)
        lam = rayleigh_quotient(x, A)
        residual = np.linalg.norm(A @ x - lam * x)
        if return_history:
            history.append({"k": k, "lambda": lam, "residual": residual, "x": x.copy()})
        if residual < tol:
            break
    return lam, x, history


def eig_invariant_axes(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) via numpy.linalg.eig.

    Columns of V are eigenvectors, matching NumPy's convention."""
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A)
    return vals, vecs
