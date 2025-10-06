from __future__ import annotations
import numpy as np
from numpy.linalg import norm, cond
from typing import Dict, Tuple


def gram_schmidt(
    A: np.ndarray, method: str = "mgs", reorth: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Gram–Schmidt orthogonalization.
    Parameters
    ----------
    A : (m, n) array
        Input full-column-rank matrix.
    method : {"cgs","mgs"}
        Classical or modified Gram–Schmidt.
    reorth : {0,1,2}
        Number of MGS-style reorthogonalization passes per column.
    Returns
    -------
    Q : (m, n) array
        Orthonormal basis (columns).
    R : (n, n) array
        Upper-triangular factor.
    Notes
    -----
    Maps math to code:
      - CGS: project against fixed {q_i} using a_j; v := a_j - Σ r_ij q_i
      - MGS: sequentially deflate v by each q_i; r_ij = q_i^T v; v := v - r_ij q_i
      - Reorth: repeat MGS projections to reduce roundoff accumulation.
    """
    A = np.asarray(A)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)

    for j in range(n):
        if method == "cgs":
            for i in range(j):
                R[i, j] = np.dot(Q[:, i].conj(), A[:, j])
            v = A[:, j] - Q[:, :j] @ R[:j, j]
        elif method == "mgs":
            v = A[:, j].copy()
            for i in range(j):
                R[i, j] = np.dot(Q[:, i].conj(), v)
                v -= R[i, j] * Q[:, i]
        else:
            raise ValueError("method must be 'cgs' or 'mgs'")

        for _ in range(reorth):
            for i in range(j):
                delta = np.dot(Q[:, i].conj(), v)
                R[i, j] += delta
                v -= delta * Q[:, i]

        R[j, j] = norm(v)
        if R[j, j] == 0:
            raise np.linalg.LinAlgError("Rank deficiency detected.")
        Q[:, j] = v / R[j, j]

    return Q, R


def compute_metrics(A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Dict[str, float]:
    """Compute diagnostic metrics for loss of orthogonality and reconstruction.
    Returns
    -------
    dict with keys:
      - ortho_frob: ||Q^T Q - I||_F
      - resid_frob_rel: ||A - Q R||_F / ||A||_F
      - max_offdiag_abs: max_{i≠j} |(Q^T Q)_{ij}|
      - max_pairwise_angle_deg: max angle deviation from 90°
      - cond2_A: 2-norm condition number of A
    """
    from numpy.linalg import norm

    G = Q.T @ Q
    n = G.shape[0]
    I_mat = np.eye(n, dtype=G.dtype)
    ortho_frob = norm(G - I_mat, "fro")
    resid_frob_rel = norm(A - Q @ R, "fro") / max(norm(A, "fro"), np.finfo(float).eps)
    G_off = G - np.diag(np.diag(G))
    max_offdiag_abs = np.max(np.abs(G_off)) if n > 1 else 0.0

    if n > 1:
        cos_vals = np.clip(np.abs(G - np.eye(n)), 0, 1)
        idx = np.triu_indices(n, 1)
        cos_max = np.max(cos_vals[idx]) if idx[0].size else 0.0
        max_pairwise_angle_deg = (
            float(np.degrees(np.arccos(1 - cos_max))) if cos_max <= 1 else 0.0
        )
    else:
        max_pairwise_angle_deg = 0.0

    try:
        cond2 = float(cond(A))
    except np.linalg.LinAlgError:
        cond2 = np.inf

    return dict(
        ortho_frob=float(ortho_frob),
        resid_frob_rel=float(resid_frob_rel),
        max_offdiag_abs=float(max_offdiag_abs),
        max_pairwise_angle_deg=float(max_pairwise_angle_deg),
        cond2_A=cond2,
    )


def make_matrix(cfg: dict):
    """Construct a test matrix according to config.
    Supported kinds:
      - gaussian: iid N(0,1)
      - correlated: columns with AR(1)-like correlation rho
      - hilbert: classical ill-conditioned Hilbert matrix (m x n submatrix)
      - vandermonde: Vandermonde with points in [0,1]
    Returns
    -------
    A : (m, n) array
    info : dict with 'kind'
    """
    rng = np.random.default_rng(cfg.get("seed", None))
    m = cfg["m"]
    n = cfg["n"]
    kind = cfg.get("matrix", {}).get("kind", "gaussian")
    noise = cfg.get("matrix", {}).get("noise_level", 0.0)
    scale = cfg.get("matrix", {}).get("scale", 1.0)
    dtype = np.float64 if cfg.get("dtype", "float64") == "float64" else np.float32

    if kind == "gaussian":
        A = rng.standard_normal((m, n), dtype=dtype) * scale
    elif kind == "correlated":
        rho = float(cfg.get("matrix", {}).get("correlation", 0.95))
        idx = np.arange(m)
        Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(m))
        A = L @ rng.standard_normal((m, n), dtype=dtype) * scale
    elif kind == "hilbert":
        i = np.arange(1, m + 1)[:, None]
        j = np.arange(1, m + 1)[None, :]
        H = 1.0 / (i + j - 1.0)
        A = H[:, :n].astype(dtype) * scale
    elif kind == "vandermonde":
        x = rng.random(m, dtype=dtype)
        V = np.vander(x, N=n, increasing=True)
        A = V * scale
    else:
        raise ValueError(f"Unknown matrix kind: {kind}")

    if noise > 0:
        A = A + noise * rng.standard_normal(A.shape, dtype=A.dtype)

    return A, {"kind": kind}
