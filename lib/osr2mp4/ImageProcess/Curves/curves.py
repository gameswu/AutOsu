"""
Pure-Python bridge replacing the osr2mp4 C extension (ccurves).

Uses AutOsu's slider_path.py for curve computation.
"""

from src.data.slider_path import (
    _compute_bezier,
    _compute_perfect_circle,
    _compute_linear,
)


# Map osr2mp4 slider type strings to curve computation functions
_TYPE_MAP = {
    "B": "bezier",
    "L": "linear",
    "P": "perfect",
    "C": "catmull",  # treated as bezier (same as osu! lazer)
}


def create_curve(slider_type, control_points, expect_length):
    """
    Replaces ccurves.create_curve.

    Parameters
    ----------
    slider_type : str  ("B", "L", "P", "C")
    control_points : list of [x, y]
    expect_length : float  (pixel_length)

    Returns
    -------
    (pos, cum_length) : (list of [x,y], list of float)
        pos has N points, cum_length has N-1 entries (one per segment).
        cum_length[i] = cumulative distance from start through segment i+1.
        This matches the C extension's adjust_curve() contract exactly.
    """
    import math

    # Convert control points to tuples
    pts = [(float(p[0]), float(p[1])) for p in control_points]

    kind = _TYPE_MAP.get(slider_type, "bezier")

    if kind == "perfect" and len(pts) == 3:
        raw = _compute_perfect_circle(pts[0], pts[1], pts[2])
    elif kind == "linear":
        raw = _compute_linear(pts)
    else:
        # bezier and catmull both go through bezier
        raw = _compute_bezier(pts)

    if len(raw) < 2:
        # Degenerate: single point
        return [[raw[0][0], raw[0][1]]] if raw else [[0, 0]], []

    # Convert to list-of-lists (matching C extension output format)
    path = [[float(p[0]), float(p[1])] for p in raw]

    # ── adjust_curve logic (mirrors curves.cpp:94-128 exactly) ──
    MIN_SEGMENT_LENGTH = 0.0001

    def _dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    total = sum(_dist(path[i - 1], path[i]) for i in range(1, len(path)))
    excess = total - expect_length

    # Trim from the end (same as C++ adjust_curve)
    while len(path) >= 2 and excess > 0:
        v1 = path[-2]
        v2 = path[-1]
        last_line_length = _dist(v1, v2)

        if last_line_length > excess + MIN_SEGMENT_LENGTH:
            if v1[0] != v2[0] or v1[1] != v2[1]:
                dx, dy = v2[0] - v1[0], v2[1] - v1[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    nx, ny = dx / length, dy / length
                    new_len = last_line_length - excess
                    path[-1] = [v1[0] + nx * new_len, v1[1] + ny * new_len]
            break
        path.pop()
        excess -= last_line_length

    # Build cum_length: N-1 entries for N path points
    # cum_length[i] = cumulative distance from path[0] through path[i+1]
    cum_length = []
    t = 0.0
    for i in range(1, len(path)):
        t += _dist(path[i - 1], path[i])
        cum_length.append(t)

    return path, cum_length


def _binary_search(v, value):
    """Binary search matching ccurves.pyx:25-38 exactly."""
    left = 0
    right = len(v)
    while left < right:
        mid = left + (right - left) // 2
        if value > v[mid]:
            left = mid + 1
        elif value < v[mid]:
            right = mid
        else:
            break
    return right


def get_pos_at(path, cum_length, length):
    """
    Replaces ccurves.get_pos_at / position_at.

    Mirrors ccurves.pyx:40-74 exactly.
    path: list of [x,y] with N points
    cum_length: list of float with N-1 entries
    length: distance along curve
    """
    if len(path) == 0 or len(cum_length) == 0:
        return [0, 0]
    if length == 0:
        return list(path[0])
    if length > cum_length[-1]:
        return list(path[-1])

    i = _binary_search(cum_length, length)
    if i > len(cum_length) - 1:
        i = len(cum_length) - 1

    length_next = cum_length[i]
    length_previous = 0 if i == 0 else cum_length[i - 1]

    i += 1

    res = [path[i - 1][0], path[i - 1][1]]
    p1 = path[i - 1]
    p2 = path[i]

    if length_previous != length_next:
        t = (length - length_previous) / (length_next - length_previous)
        res[0] = res[0] + (p2[0] - p1[0]) * t
        res[1] = res[1] + (p2[1] - p1[1]) * t

    return res


class Curve:
    def __init__(self, slider_type, control_points, expect_length):
        self.slider_type = slider_type
        self.control_points = control_points
        self.pos, self.cum_length = create_curve(slider_type, control_points, expect_length)

    def at(self, distance):
        return get_pos_at(self.pos, self.cum_length, distance)


def getclass(slidertype, points, pixellength):
    return Curve(slidertype, points, pixellength)
