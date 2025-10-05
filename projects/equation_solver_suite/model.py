from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any
import math, cmath

Number = Union[float, complex]

@dataclass
class SolveResult:
    solution: Union[List[Number], Tuple[Number, Number], None]
    status: str
    steps: List[Dict[str, Any]]

def snapshot_augmented(A: List[List[Number]], b: List[Number]) -> List[List[Number]]:
    return [row[:] + [b[i]] for i, row in enumerate(A)]

def gaussian_elimination(A_in: List[List[Number]], b_in: List[Number], eps: float = 1e-12, pivoting: bool = True, record_snapshots: bool = True) -> SolveResult:
    n = len(A_in)
    A = [row[:] for row in A_in]
    b = b_in[:]
    steps: List[Dict[str, Any]] = []
    if record_snapshots:
        steps.append({"phase": "init", "Aug": snapshot_augmented(A, b)})
    for k in range(n):
        if pivoting:
            pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))
            if pivot_row != k:
                A[k], A[pivot_row] = A[pivot_row], A[k]
                b[k], b[pivot_row] = b[pivot_row], b[k]
                if record_snapshots:
                    steps.append({"phase": "pivot", "k": k, "pivot_row": pivot_row, "pivot_value": A[k][k], "Aug": snapshot_augmented(A, b)})
        if abs(A[k][k]) < eps:
            steps.append({"phase": "singular", "k": k, "pivot": A[k][k]})
            return SolveResult(None, "singular", steps)
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            if record_snapshots:
                steps.append({"phase": "multiplier", "i": i, "k": k, "m": m})
            for j in range(k, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
            if record_snapshots:
                steps.append({"phase": "eliminate", "i": i, "k": k, "m": m, "Aug": snapshot_augmented(A, b)})
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        if abs(A[i][i]) < eps:
            steps.append({"phase": "singular_back", "i": i, "pivot": A[i][i]})
            return SolveResult(None, "singular", steps)
        x[i] = (b[i] - s) / A[i][i]
        if record_snapshots:
            steps.append({"phase": "back_sub", "i": i, "x_i": x[i]})
    if record_snapshots:
        steps.append({"phase": "done", "solution": x})
    return SolveResult(x, "ok", steps)

def solve_quadratic(a: Number, b: Number, c: Number, eps: float = 1e-15) -> SolveResult:
    steps: List[Dict[str, Any]] = []
    if abs(a) < eps:
        steps.append({"phase": "degenerate", "a": a})
        return SolveResult(None, "degenerate", steps)
    Delta = b*b - 4*a*c
    steps.append({"phase": "discriminant", "Delta": Delta})
    if isinstance(Delta, complex) or Delta < 0:
        sqrtD = cmath.sqrt(Delta)
        x1 = (-b + sqrtD) / (2 * a)
        x2 = (-b - sqrtD) / (2 * a)
        status = "complex"
        steps.append({"phase": "complex_roots", "sqrtD": sqrtD, "x1": x1, "x2": x2})
    else:
        sqrtD = math.sqrt(Delta)
        if b != 0:
            q = -0.5 * (b + math.copysign(sqrtD, b))
            x1 = q / a
            x2 = c / q
            branch = "stable_q"
        else:
            x1 =  sqrtD / (2 * a)
            x2 = -sqrtD / (2 * a)
            branch = "b_zero"
        status = "real" if Delta > 0 else "repeated"
        steps.append({"phase": "real_roots", "branch": branch, "sqrtD": sqrtD, "x1": x1, "x2": x2})
    steps.append({"phase": "done", "status": status})
    return SolveResult((x1, x2), status, steps)
