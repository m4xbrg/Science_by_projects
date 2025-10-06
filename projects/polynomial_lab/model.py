from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RootResult:
    roots: np.ndarray
    scale: complex
    clustered: List[Tuple[complex, int]]


class PolynomialLab:
    def __init__(self, coeffs):
        a = np.array(coeffs, dtype=np.complex128)
        nz = np.nonzero(np.abs(a) > 0)[0]
        if len(nz) == 0:
            raise ValueError("Zero polynomial not supported.")
        self.a = a[nz[0] :]
        self.n = self.a.size - 1

    @staticmethod
    def _normalize(a: np.ndarray):
        scale = a[0]
        return a / scale, scale

    def polyval(self, x: np.ndarray) -> np.ndarray:
        a = self.a
        y = np.zeros_like(x, dtype=np.complex128)
        for c in a:
            y = y * x + c
        return y

    def roots_companion(self) -> RootResult:
        a_monic, scale = self._normalize(self.a)
        n = a_monic.size - 1
        if n == 0:
            return RootResult(np.array([]), scale, [])
        C = np.zeros((n, n), dtype=np.complex128)
        C[:-1, 1:] = np.eye(n - 1, dtype=np.complex128)
        C[-1, :] = -a_monic[1:].conj()
        r = np.linalg.eigvals(C)
        clustered = self._cluster_roots(r)
        return RootResult(r, scale, clustered)

    def roots_durand_kerner(
        self, tol: float = 1e-12, max_iter: int = 200
    ) -> RootResult:
        a_monic, scale = self._normalize(self.a)
        n = a_monic.size - 1
        if n == 0:
            return RootResult(np.array([]), scale, [])
        angles = 2j * np.pi * np.arange(n) / n
        r = 0.4 * np.exp(angles)
        r = r.astype(np.complex128)

        def P(z):
            y = np.zeros_like(z, dtype=np.complex128)
            for c in a_monic:
                y = y * z + c
            return y

        for _ in range(max_iter):
            r_new = r.copy()
            for j in range(n):
                denom = np.prod(r[j] - np.delete(r, j))
                r_new[j] = r[j] - P(r[j]) / denom
            if np.max(np.abs(r_new - r)) < tol:
                r = r_new
                break
            r = r_new
        clustered = self._cluster_roots(r)
        return RootResult(r, scale, clustered)

    def _cluster_roots(self, r: np.ndarray, tol: float = 1e-7):
        r_sorted = sorted(r, key=lambda z: (np.real(z), np.imag(z)))
        clusters = []
        for z in r_sorted:
            if not clusters:
                clusters.append([z])
            else:
                if abs(z - clusters[-1][-1]) < tol:
                    clusters[-1].append(z)
                else:
                    clusters.append([z])
        return [(sum(g) / len(g), len(g)) for g in clusters]

    def factors_over_C(self, clustered, scale):
        return clustered

    def factors_over_R(self, clustered, scale, tol: float = 1e-10):
        out = []
        used = [False] * len(clustered)
        for i, (ri, mi) in enumerate(clustered):
            if used[i]:
                continue
            if abs(np.imag(ri)) < tol:
                out.append((np.array([1.0, -np.real(ri)]), mi))
                used[i] = True
            else:
                found = False
                for k, (rk, mk) in enumerate(clustered):
                    if used[k] or k == i:
                        continue
                    if abs(rk - np.conj(ri)) < 1e-6 and mk == mi:
                        Re, Im = np.real(ri), np.imag(ri)
                        coeffs = np.array([1.0, -2 * Re, Re * Re + Im * Im])
                        out.append((coeffs, mi))
                        used[i] = used[k] = True
                        found = True
                        break
                if not found:
                    Re, Im = np.real(ri), np.imag(ri)
                    coeffs = np.array([1.0, -2 * Re, Re * Re + Im * Im])
                    out.append((coeffs, mi))
                    used[i] = True
        return out

    def synthetic_division(self, r: complex):
        a = self.a
        n = a.size - 1
        b = np.zeros_like(a, dtype=np.complex128)
        b[0] = a[0]
        for k in range(1, n + 1):
            b[k] = a[k] + r * b[k - 1]
        q = b[:-1]
        rem = b[-1]
        return q, rem, b

    @staticmethod
    def expand_factors_C(scale, factors_C):
        coeffs = np.array([1.0], dtype=np.complex128)
        for r, m in factors_C:
            base = np.array([1.0, -r], dtype=np.complex128)
            for _ in range(m):
                coeffs = np.convolve(coeffs, base)
        return scale * coeffs

    def verify(self, factors_C, scale):
        a_rec = PolynomialLab.expand_factors_C(scale, factors_C)
        m = max(len(a_rec), len(self.a))
        a1 = np.pad(self.a, (m - len(self.a), 0))
        a2 = np.pad(a_rec, (m - len(a_rec), 0))
        coeff_rel_err = np.linalg.norm(a1 - a2) / max(1e-16, np.linalg.norm(a1))
        rr = [r for r, _ in self._cluster_roots(np.roots(a2))]
        res = np.max(np.abs(self.polyval(np.array(rr))))
        return {"coeff_rel_err": float(coeff_rel_err), "residual_on_roots": float(res)}
