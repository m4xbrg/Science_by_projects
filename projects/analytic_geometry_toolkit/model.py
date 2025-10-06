from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict, Any
import math
import numpy as np

Point = Tuple[float, float]
IntersectResult = Tuple[List[Point], str]
DEFAULT_EPS = 1e-9


def point_distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points in R^2."""
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


def midpoint(p1: Point, p2: Point) -> Point:
    """Midpoint of the segment connecting p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def almost_equal(a: float, b: float, eps: float = DEFAULT_EPS) -> bool:
    """Floating-point comparison within tolerance eps."""
    return abs(a - b) <= eps


@dataclass(frozen=True)
class Line:
    """Line in implicit form: a*x + b*y + c = 0 (optionally normalized)."""

    a: float
    b: float
    c: float
    normalized: bool = True

    @staticmethod
    def from_points(
        p1: Point, p2: Point, normalize: bool = True, eps: float = DEFAULT_EPS
    ) -> "Line":
        if point_distance(p1, p2) <= eps:
            raise ValueError("Degenerate line: points are coincident.")
        x1, y1 = p1
        x2, y2 = p2
        a = y1 - y2
        b = x2 - x1
        c = -(a * x1 + b * y1)
        if normalize:
            s = math.hypot(a, b)
            a, b, c = a / s, b / s, c / s
        return Line(a, b, c, normalized=normalize)

    @staticmethod
    def from_coeffs(
        a: float, b: float, c: float, normalize: bool = True, eps: float = DEFAULT_EPS
    ) -> "Line":
        if math.hypot(a, b) <= eps:
            raise ValueError("Invalid line: a and b cannot both be ~0.")
        if normalize:
            s = math.hypot(a, b)
            a, b, c = a / s, b / s, c / s
        return Line(a, b, c, normalized=normalize)

    def distance_to_point(self, p: Point) -> float:
        x, y = p
        denom = 1.0 if self.normalized else math.hypot(self.a, self.b)
        return abs(self.a * x + self.b * y + self.c) / denom

    def foot_of_perpendicular(self, p: Point) -> Point:
        x0, y0 = p
        denom = self.a**2 + self.b**2
        t = (self.a * x0 + self.b * y0 + self.c) / denom
        return (x0 - self.a * t, y0 - self.b * t)

    def direction_unit(self) -> Point:
        denom = math.hypot(self.a, self.b)
        return (self.b / denom, -self.a / denom)

    def intersect_line(
        self, other: "Line", eps: float = DEFAULT_EPS
    ) -> IntersectResult:
        a1, b1, c1 = self.a, self.b, self.c
        a2, b2, c2 = other.a, other.b, other.c
        det = a1 * b2 - a2 * b1
        if abs(det) <= eps:
            denom = a2**2 + b2**2
            x_test = -a2 * c2 / denom
            y_test = -b2 * c2 / denom
            if self.distance_to_point((x_test, y_test)) <= eps:
                return ([], "coincident")
            else:
                return ([], "parallel")
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        return ([(x, y)], "proper")


@dataclass(frozen=True)
class Circle:
    """Circle defined by center (x0, y0) and radius r>0."""

    x0: float
    y0: float
    r: float

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError("Circle radius must be positive.")

    @property
    def center(self) -> Point:
        return (self.x0, self.y0)

    def intersect_line(self, line: Line, eps: float = DEFAULT_EPS) -> IntersectResult:
        d = line.distance_to_point(self.center)
        if d > self.r + eps:
            return ([], "disjoint")
        p_perp = line.foot_of_perpendicular(self.center)
        if abs(d - self.r) <= eps:
            return ([p_perp], "tangent")
        u = line.direction_unit()
        delta = math.sqrt(max(self.r**2 - d**2, 0.0))
        p1 = (p_perp[0] - delta * u[0], p_perp[1] - delta * u[1])
        p2 = (p_perp[0] + delta * u[0], p_perp[1] + delta * u[1])
        return ([p1, p2], "secant")

    def intersect_circle(
        self, other: "Circle", eps: float = DEFAULT_EPS
    ) -> IntersectResult:
        O1 = np.array(self.center)
        O2 = np.array(other.center)
        d = float(np.linalg.norm(O2 - O1))

        if abs(d) <= eps:
            if abs(self.r - other.r) <= eps:
                return ([], "coincident")
            else:
                return ([], "concentric-disjoint")

        r1, r2 = self.r, other.r
        if d > r1 + r2 + eps or d < abs(r1 - r2) - eps:
            return ([], "disjoint")

        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h2 = r1**2 - a**2
        if h2 < 0 and abs(h2) <= eps:
            h2 = 0.0
        h = math.sqrt(max(h2, 0.0))

        e_hat = (O2 - O1) / d
        P0 = O1 + a * e_hat
        n_hat = np.array([-e_hat[1], e_hat[0]])

        if abs(h) <= eps:
            return ([(float(P0[0]), float(P0[1]))], "tangent")

        P_plus = P0 + h * n_hat
        P_minus = P0 - h * n_hat
        pts = [
            (float(P_minus[0]), float(P_minus[1])),
            (float(P_plus[0]), float(P_plus[1])),
        ]
        return (pts, "secant")
