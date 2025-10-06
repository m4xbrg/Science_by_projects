import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass


@dataclass
class MapParams:
    """Container for map parameters."""

    A: np.ndarray  # (n,n) matrix


def step(k: int, x: np.ndarray, params: MapParams) -> np.ndarray:
    """
    One-step map for x_{k+1} = A x_k.

    Parameters
    ----------
    k : int
        Discrete time index (unused but kept for API symmetry).
    x : np.ndarray, shape (n,)
        Current state x_k.
    params : MapParams
        Parameters containing matrix A.

    Returns
    -------
    np.ndarray, shape (n,)
        Next state x_{k+1}.
    """
    # Math → code: x_{k+1} = A x_k
    return params.A @ x


def eig_info(A: ArrayLike, save_vectors: bool = False) -> dict:
    """
    Compute eigenstructure and stability/type classification.

    Parameters
    ----------
    A : ArrayLike
        Square matrix.
    save_vectors : bool
        Whether to include eigenvectors in the returned dict.

    Returns
    -------
    dict
        Contains eigenvalues, spectral radius, stability, and (for 2D) type.
    """
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A)
    rho = np.max(np.abs(vals))

    if rho < 1 - 1e-12:
        stability = "asymptotically stable"
    elif rho > 1 + 1e-12:
        stability = "unstable"
    else:
        stability = "marginal (check Jordan blocks / normality)"

    type2d = None
    if A.shape == (2, 2):
        a, b = vals
        ar, br = np.abs(a), np.abs(b)
        if np.iscomplex(a) and np.iscomplex(b):
            r = np.abs(a)
            if r < 1 - 1e-12:
                type2d = "stable spiral (focus)"
            elif r > 1 + 1e-12:
                type2d = "unstable spiral (focus)"
            else:
                type2d = "center (if normal); otherwise marginal"
        else:
            # real-eigenvalue cases (including sign flips)
            if ar < 1 and br < 1:
                type2d = "stable node (possibly improper/with flips)"
            elif ar > 1 and br > 1:
                type2d = "unstable node (possibly improper/with flips)"
            elif (ar < 1 and br > 1) or (br < 1 and ar > 1):
                type2d = "saddle"
            else:
                type2d = "marginal/degenerate (|lambda|=1 for at least one)"

    out = {
        "eigenvalues": vals.tolist(),
        "spectral_radius": float(rho),
        "stability": stability,
        "type2d": type2d,
    }
    if save_vectors:
        out["eigenvectors"] = vecs.tolist()
    return out
