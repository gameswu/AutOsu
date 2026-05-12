"""
Slider path computation for osu! std mode.

Implements Bezier, Linear, and Perfect Circle curve types with
adaptive subdivision and distance-based equalization.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]

BEZIER_TOLERANCE = 0.25
CIRCULAR_ARC_TOLERANCE = 0.1


# ── Utilities ────────────────────────────────────────────────────────────

def _dist(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _dist_sq(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


# ── Bezier ───────────────────────────────────────────────────────────────

def _bezier_subdivide(points: List[Point]) -> Tuple[List[Point], List[Point]]:
    """De Casteljau subdivision at t = 0.5."""
    n = len(points)
    left = [points[0]] + [None] * (n - 1)  # type: ignore
    right = [None] * (n - 1) + [points[-1]]  # type: ignore

    work = list(points)
    for j in range(1, n):
        for i in range(n - j):
            work[i] = ((work[i][0] + work[i + 1][0]) * 0.5,
                        (work[i][1] + work[i + 1][1]) * 0.5)
        left[j] = work[0]
        right[n - 1 - j] = work[n - 1 - j]
    return left, right  # type: ignore


def _bezier_is_flat(points: List[Point]) -> bool:
    """Check whether all interior control points lie close to the baseline."""
    if len(points) <= 2:
        return True
    p0 = points[0]
    pn = points[-1]
    for i in range(1, len(points) - 1):
        # Perpendicular distance to line p0→pn
        px, py = points[i]
        dx, dy = pn[0] - p0[0], pn[1] - p0[1]
        line_len_sq = dx * dx + dy * dy
        if line_len_sq < 1e-12:
            d_sq = _dist_sq(points[i], p0)
        else:
            t = max(0.0, min(1.0, ((px - p0[0]) * dx + (py - p0[1]) * dy) / line_len_sq))
            proj = (p0[0] + t * dx, p0[1] + t * dy)
            d_sq = _dist_sq(points[i], proj)
        if d_sq > BEZIER_TOLERANCE * BEZIER_TOLERANCE:
            return False
    return True


def _flatten_bezier_segment(points: List[Point]) -> List[Point]:
    """Flatten a single bezier segment into a polyline via adaptive subdivision."""
    if len(points) < 2:
        return list(points)

    output: List[Point] = []
    stack = [points]

    while stack:
        current = stack.pop()
        if _bezier_is_flat(current):
            # Emit all but the last point (last will come from next segment or be appended)
            if not output:
                output.append(current[0])
            for i in range(1, len(current) - 1):
                output.append(current[i])
        else:
            left, right = _bezier_subdivide(current)
            stack.append(right)
            stack.append(left)

    output.append(points[-1])
    return output


def _compute_bezier(control_points: List[Point]) -> List[Point]:
    """Handle multi-segment bezier (split at repeated control points)."""
    segments: List[List[Point]] = []
    current: List[Point] = [control_points[0]]

    for i in range(1, len(control_points)):
        if (abs(control_points[i][0] - control_points[i - 1][0]) < 1e-6 and
                abs(control_points[i][1] - control_points[i - 1][1]) < 1e-6):
            if len(current) >= 2:
                segments.append(current)
            current = [control_points[i]]
        else:
            current.append(control_points[i])
    if len(current) >= 2:
        segments.append(current)

    path: List[Point] = []
    for seg in segments:
        flat = _flatten_bezier_segment(seg)
        if path and flat:
            # skip duplicate junction point
            if _dist(path[-1], flat[0]) < 0.01:
                flat = flat[1:]
        path.extend(flat)
    return path


# ── Perfect circle ───────────────────────────────────────────────────────

def _compute_perfect_circle(p0: Point, p1: Point, p2: Point) -> List[Point]:
    """Arc through three points."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2

    # Collinearity check
    cross = (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)
    if abs(cross) < 1e-6:
        return _compute_bezier([p0, p1, p2])

    # Circumscribed circle
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    a_sq = ax * ax + ay * ay
    b_sq = bx * bx + by * by
    c_sq = cx * cx + cy * cy

    centre_x = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d
    centre_y = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d

    dx, dy = ax - centre_x, ay - centre_y
    radius = math.sqrt(dx * dx + dy * dy)

    theta_start = math.atan2(ay - centre_y, ax - centre_x)
    theta_mid = math.atan2(by - centre_y, bx - centre_x)
    theta_end = math.atan2(cy - centre_y, cx - centre_x)

    # Determine direction using cross product
    while theta_mid < theta_start:
        theta_mid += 2 * math.pi
    while theta_end < theta_start:
        theta_end += 2 * math.pi

    if theta_mid > theta_end:
        # p1 is not between p0 and p2 going counter-clockwise → go clockwise
        direction = -1.0
        theta_range_raw = 2 * math.pi - (theta_end - theta_start)
        if theta_range_raw > 2 * math.pi:
            theta_range_raw -= 2 * math.pi
        theta_range = theta_range_raw
    else:
        direction = 1.0
        theta_range = theta_end - theta_start

    if radius < 1e-6:
        return [p0, p2]

    # Number of points
    if 2 * radius <= CIRCULAR_ARC_TOLERANCE:
        n_points = 2
    else:
        step = math.acos(max(-1.0, min(1.0, 1.0 - CIRCULAR_ARC_TOLERANCE / radius)))
        n_points = max(2, int(math.ceil(theta_range / step)) + 1)

    output: List[Point] = []
    for i in range(n_points):
        frac = i / (n_points - 1)
        theta = theta_start + direction * frac * theta_range
        output.append((centre_x + radius * math.cos(theta),
                        centre_y + radius * math.sin(theta)))
    return output


# ── Linear ───────────────────────────────────────────────────────────────

def _compute_linear(control_points: List[Point]) -> List[Point]:
    return list(control_points)


# ── Path equalization ────────────────────────────────────────────────────

def _cumulative_distances(path: List[Point]) -> List[float]:
    dists = [0.0]
    for i in range(1, len(path)):
        dists.append(dists[-1] + _dist(path[i], path[i - 1]))
    return dists


def position_at_distance(
    path: List[Point],
    cum_dist: List[float],
    target: float,
) -> Point:
    """Interpolate position at *target* distance along the polyline."""
    if target <= 0:
        return path[0]
    total = cum_dist[-1]
    if target >= total:
        return path[-1]

    # Binary search
    lo, hi = 0, len(cum_dist) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cum_dist[mid] < target:
            lo = mid
        else:
            hi = mid

    seg_len = cum_dist[hi] - cum_dist[lo]
    t = (target - cum_dist[lo]) / seg_len if seg_len > 1e-9 else 0.0
    return _lerp(path[lo], path[hi], t)


def equalize_path(raw: List[Point], pixel_length: float, step: float = 1.0) -> List[Point]:
    """Resample *raw* polyline to evenly-spaced points up to *pixel_length*."""
    if len(raw) < 2:
        return list(raw)

    cum = _cumulative_distances(raw)
    total = min(pixel_length, cum[-1])

    out: List[Point] = [raw[0]]
    d = step
    while d < total:
        out.append(position_at_distance(raw, cum, d))
        d += step
    out.append(position_at_distance(raw, cum, total))
    return out


# ── Public API ───────────────────────────────────────────────────────────

def compute_slider_path(
    curve_type_code: int,
    control_points: List[Point],
    pixel_length: float,
) -> Tuple[List[Point], List[float]]:
    """
    Compute the final equalized slider path.

    Parameters
    ----------
    curve_type_code : CurveType int value (0=Bezier, 1=Catmull, 2=Linear, 3=Perfect)
    control_points : list of (x, y) tuples, first point = slider head
    pixel_length : slider pixel_length from .osu

    Returns
    -------
    (equalized_path, cumulative_distances)
    """
    from src.data.osu_parser import CurveType

    if curve_type_code == CurveType.BEZIER or curve_type_code == CurveType.CATMULL:
        raw = _compute_bezier(control_points)
    elif curve_type_code == CurveType.PERFECT:
        if len(control_points) == 3:
            raw = _compute_perfect_circle(control_points[0], control_points[1], control_points[2])
        else:
            raw = _compute_bezier(control_points)
    elif curve_type_code == CurveType.LINEAR:
        raw = _compute_linear(control_points)
    else:
        raw = _compute_linear(control_points)

    equalized = equalize_path(raw, pixel_length, step=2.0)
    cum = _cumulative_distances(equalized)
    return equalized, cum
