"""
model.py — Core affine transformation primitives (2D) and polygon utilities.

API highlights:
- Transform2D: static builders for translation/rotation/reflection matrices.
- compose(ops): compose a list of operation dicts into a single 3x3 matrix.
- apply(A, V): apply a 3x3 homogeneous transform to Nx2 vertices.
- pipeline(V, ops, record_steps): run ops sequentially, optionally recording steps.

Conventions:
- Radians for angles; right-handed coordinates.
- Vertices shape: (N, 2). We'll not close polygons; plotting can close visually.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from typing import List, Dict, Tuple, Optional

Array = np.ndarray

I3 = np.eye(3)


@dataclass(frozen=True)
class OpSpec:
    """Canonical op spec for stronger typing (dicts also supported)."""

    type: str
    theta: Optional[float] = None
    center: Optional[Tuple[float, float]] = None
    tx: Optional[float] = None
    ty: Optional[float] = None
    normal: Optional[Tuple[float, float]] = None
    angle: Optional[float] = None
    point: Optional[Tuple[float, float]] = None


class Transform2D:
    """Factory for 3x3 homogeneous transforms."""

    @staticmethod
    def translate(tx: float, ty: float) -> Array:
        """Translation matrix T(tx, ty)."""
        T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=float)
        return T

    @staticmethod
    def rotate(theta: float, center: Optional[Tuple[float, float]] = None) -> Array:
        """Rotation matrix R(theta) about origin, or about 'center' if provided."""
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        if center is None:
            return R
        cx, cy = center
        return Transform2D.translate(cx, cy) @ R @ Transform2D.translate(-cx, -cy)

    @staticmethod
    def _householder_from_normal(nx: float, ny: float) -> Array:
        """2x2 Householder for reflection across line with unit normal (nx, ny)."""
        # Normalize to guard against non-unit inputs
        n = np.array([nx, ny], dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            raise ValueError("Normal vector must be non-zero.")
        n /= norm
        nx, ny = n
        H2 = np.array(
            [[1 - 2 * nx * nx, -2 * nx * ny], [-2 * nx * ny, 1 - 2 * ny * ny]],
            dtype=float,
        )
        # Lift to 3x3 homogeneous
        H = np.eye(3, dtype=float)
        H[:2, :2] = H2
        return H

    @staticmethod
    def reflect(
        normal: Optional[Tuple[float, float]] = None,
        angle: Optional[float] = None,
        point: Optional[Tuple[float, float]] = None,
    ) -> Array:
        """
        Reflection matrix across a line:
        - If 'normal' provided: reflect across line with that (unit or non-unit) normal at origin.
        - Else if 'angle' provided: reflect across line whose unit normal is (cos(angle), sin(angle)).
        - If 'point' provided: line is translated to pass through 'point'.
        """
        if normal is None and angle is None:
            raise ValueError("Provide either 'normal' or 'angle' for reflection.")

        if normal is None:
            nx, ny = math.cos(angle), math.sin(angle)
        else:
            nx, ny = normal

        H = Transform2D._householder_from_normal(nx, ny)

        if point is not None:
            px, py = point
            return Transform2D.translate(px, py) @ H @ Transform2D.translate(-px, -py)
        return H


def build_matrix(op: Dict) -> Array:
    """
    Build a 3x3 matrix from an op dict (or OpSpec).
    Supported op['type'] in {'translate', 'rotate', 'reflect'}.
    """
    t = op.get("type")
    if t == "translate":
        return Transform2D.translate(float(op["tx"]), float(op["ty"]))
    elif t == "rotate":
        theta = float(op["theta"])
        center = op.get("center")
        if center is not None:
            center = (float(center[0]), float(center[1]))
        return Transform2D.rotate(theta, center=center)
    elif t == "reflect":
        normal = op.get("normal")
        angle = op.get("angle")
        point = op.get("point")
        if normal is not None:
            normal = (float(normal[0]), float(normal[1]))
        if point is not None:
            point = (float(point[0]), float(point[1]))
        return Transform2D.reflect(normal=normal, angle=angle, point=point)
    else:
        raise ValueError(f"Unsupported op type: {t}")


def compose(ops: List[Dict]) -> Array:
    """
    Compose a list of transforms (right-multiplication order): A = A_k ... A_1.
    """
    A = I3.copy()
    for op in ops:
        A = build_matrix(op) @ A
    return A


def apply(A: Array, V: Array) -> Array:
    """
    Apply 3x3 homogeneous transform A to Nx2 vertex array V -> Nx2.
    """
    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("V must have shape (N, 2)")
    N = V.shape[0]
    Vh = np.hstack([V, np.ones((N, 1), dtype=float)])
    Vh2 = (A @ Vh.T).T
    return Vh2[:, :2]


def pipeline(V: Array, ops: List[Dict], record_steps: bool = True):
    """
    Execute ops sequentially, returning (A_total, V_final, records).
    'records' is a list of dicts with 'step' and 'vertices' if record_steps=True.
    """
    records = []
    A_total = I3.copy()
    if record_steps:
        records.append({"step": 0, "vertices": V.copy()})
    for i, op in enumerate(ops, start=1):
        Ai = build_matrix(op)
        A_total = Ai @ A_total
        V = apply(Ai, V)
        if record_steps:
            records.append({"step": i, "vertices": V.copy()})
    return A_total, V, records
