from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

EPS = np.finfo(float).eps

@dataclass
class LinAlgReport:
    rank: int
    tol: float
    det: float
    cond2: Optional[float]  # 2-norm condition number (if computed)
    notes: str

class MatrixWorkbench:
    """
    A reusable workbench for inspecting square matrices:
    - rank (via SVD)
    - determinant (via LU with partial pivoting)
    - LU and QR factorizations (Householder QR)
    - geometric action on [0,1]^n (n=2 or 3)
    """

    def __init__(self, A: np.ndarray):
        """
        Parameters
        ----------
        A : (n,n) ndarray
            Real square matrix.
        """
        A = np.array(A, dtype=float, copy=True)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        self.A = A
        self.n = A.shape[0]

    # ---------- Diagnostics ----------
    def svd_rank(self, tol: Optional[float] = None) -> Tuple[int, float, np.ndarray]:
        """
        Compute numerical rank via SVD.

        Returns
        -------
        rank : int
        tol  : float
        s    : ndarray
        """
        s = np.linalg.svd(self.A, compute_uv=False)
        if tol is None:
            tol = max(self.A.shape) * s.max() * EPS
        rank = int(np.sum(s > tol))
        return rank, tol, s

    def lu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LU with partial pivoting implemented via Doolittle + pivoting.

        Returns
        -------
        P, L, U : ndarrays
        """
        A = self.A.copy()
        n = self.n
        P = np.eye(n)
        L = np.zeros_like(A)
        U = A.copy()

        for k in range(n):
            # pivot
            p = np.argmax(np.abs(U[k:, k])) + k
            if U[p, k] == 0.0:
                continue
            if p != k:
                U[[k, p], :] = U[[p, k], :]
                P[[k, p], :] = P[[p, k], :]
                L[[k, p], :k] = L[[p, k], :k]
            L[k, k] = 1.0
            # elimination
            for i in range(k+1, n):
                if U[k, k] == 0.0:
                    continue
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] = U[i, k:] - L[i, k]*U[k, k:]
                U[i, k] = 0.0
        return P, L, U

    def det_via_lu(self) -> float:
        """
        Determinant via LU with partial pivoting: det(A) = sign(P) * prod(diag(U))
        """
        P, _, U = self.lu()
        signP = np.linalg.det(P)  # ±1
        return float(signP * np.prod(np.diag(U)))

    def qr_householder(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Householder QR factorization.

        Returns
        -------
        Q, R : ndarrays
        """
        A = self.A.copy()
        n = self.n
        Q = np.eye(n)
        R = A.copy()

        for k in range(n-1):
            x = R[k:, k]
            normx = np.linalg.norm(x)
            if normx == 0:
                continue
            e1 = np.zeros_like(x)
            e1[0] = 1.0
            v = x + np.sign(x[0] if x[0] != 0 else 1.0) * normx * e1
            v = v / np.linalg.norm(v)
            # Apply reflector to R (left)
            R[k:, k:] = R[k:, k:] - 2.0 * np.outer(v, v @ R[k:, k:])
            # Accumulate Q (right-multiply by reflector^T)
            Q[:, k:] = Q[:, k:] - 2.0 * (Q[:, k:] @ v)[:, None] * v[None, :]
        return Q, R

    def condition_number_2(self) -> float:
        """2-norm condition number via singular values."""
        s = np.linalg.svd(self.A, compute_uv=False)
        if np.min(s) == 0:
            return np.inf
        return float(np.max(s) / np.min(s))

    def report(self, compute_cond: bool = True) -> LinAlgReport:
        """
        Compute a one-shot diagnostic report.
        """
        rank, tol, _ = self.svd_rank()
        detA = self.det_via_lu()
        cond2 = self.condition_number_2() if compute_cond else None
        notes = "Orientation flip" if detA < 0 else "Orientation preserved"
        return LinAlgReport(rank=rank, tol=tol, det=detA, cond2=cond2, notes=notes)

    # ---------- Geometry ----------
    def transform_points(self, X: np.ndarray) -> np.ndarray:
        """Apply A to a set of column-stacked points X (shape: n x m)."""
        if X.shape[0] != self.n:
            raise ValueError("Point dimensionality mismatch")
        return self.A @ X

    @staticmethod
    def unit_square_grid(resolution: int = 11):
        """
        Generate grid and wireframe edges for the unit square [0,1]^2.

        Returns
        -------
        P : (2, m) grid points (column-stacked)
        edges : list of (2,2) arrays of segment endpoints
        """
        xs = np.linspace(0, 1, resolution)
        ys = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        P = np.vstack([X.ravel(), Y.ravel()])

        # edges along square boundary
        corners = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=float).T  # (2,5)
        edges = [corners[:, i:i+2] for i in range(4)]

        # internal grid lines
        for x in xs:
            edges.append(np.array([[x, x],[0,1]]))
        for y in ys:
            edges.append(np.array([[0,1],[y, y]]))
        return P, edges

    @staticmethod
    def unit_cube_edges() -> list:
        """
        Wireframe edges for the unit cube [0,1]^3; returns list of (3,2) segments.
        """
        verts = np.array([[x,y,z] for x in [0,1] for y in [0,1] for z in [0,1]], dtype=float)
        edges = []
        idx = lambda x,y,z: x*4 + y*2 + z
        for x in [0,1]:
            for y in [0,1]:
                edges.append(np.c_[verts[idx(x,y,0)], verts[idx(x,y,1)]].T)
            for z in [0,1]:
                edges.append(np.c_[verts[idx(x,0,z)], verts[idx(x,1,z)]].T)
        for y in [0,1]:
            for z in [0,1]:
                edges.append(np.c_[verts[idx(0,y,z)], verts[idx(1,y,z)]].T)
        return edges

    def transform_edges(self, edges: list) -> list:
        """Apply A to a list of edge segments of shape (n,2)."""
        return [self.A @ e for e in edges]
