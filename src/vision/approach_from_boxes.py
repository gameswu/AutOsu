"""
Approach-ratio estimation from YOLO boxes.

Primary timing path for the redesigned player. Instead of measuring the
approach ring's radius from raw pixels (see ``approach_geometry.py``, kept as a
fallback), we read it straight off the detector: the network emits a dedicated
``approach_circle`` class (id 5) whose bounding box is concentric with the
hitcircle / slider-head it belongs to. Because YOLO separates overlapping
instances, this is far more robust than pixel CV in stacked / streamed
patterns — exactly the regime the geometric estimator struggled with.

Geometry (identical to ``approach_geometry``)::

    s     = r_ring / R_disc          # ring scale, 4 (just appeared) -> 1 (hit)
    ratio = clamp((4 - s) / 3, 0, 1) # 0 = just appeared, 1 = hit time

The detector emits class id 4 (``approach_circle``); its box is concentric with
the hitcircle / slider-head it belongs to.

**Edge clipping.** A just-appeared ring is ~4x the disc and frequently extends
past the capture frame, so the detector's bounding box is *clamped* to the frame
edge. A clamped box has both a shifted centre (breaks concentric pairing) and a
shrunken size (inflates the ratio) — which made fresh objects spuriously read
``ratio = 1.0``. :func:`box_geometry` reconstructs the true centre and radius
from whichever axis / edges are *not* clipped, fixing both.

An actionable object with no matched ring is **not** assumed to be at hit time;
defaulting to ``1.0`` there caused phantom taps when a (clipped) ring merely
failed to pair. Instead the temporal filter carries the object's own history
forward (a genuinely collapsed ring stays high; a fresh unmatched object stays
low).

An optional temporal filter exploits the fact that the ratio rises linearly at
``1 / time_preempt`` for *every* object in a map: a robust global slope is
pooled across tracks and each object only needs its own phase. This denoises
the least-reliable low-ratio readings.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from src.vision.detector import Detection, FrameDetections, ObjClass

S_START = 4.0  # ring scale at spawn (matches osu!lazer DrawableHitCircle Scale=4)
_EDGE_EPS = 1.5  # px; a box side this close to the frame edge is treated as clipped


def box_geometry(det: Detection, frame_w: float, frame_h: float) -> Tuple[float, float, float]:
    """Return (cx, cy, radius) for a box, corrected for frame-edge clipping.

    The detector clamps boxes to the frame, so a ring extending off-screen loses
    part of its extent. We recover the true centre/radius from the unclipped
    axis (a circle's radius equals its half-extent along any axis that is not
    truncated), and the true centre from any real (unclipped) edge.
    """
    x1 = det.cx - det.w / 2.0
    x2 = det.cx + det.w / 2.0
    y1 = det.cy - det.h / 2.0
    y2 = det.cy + det.h / 2.0

    left_clip = x1 <= _EDGE_EPS
    right_clip = x2 >= frame_w - _EDGE_EPS
    top_clip = y1 <= _EDGE_EPS
    bot_clip = y2 >= frame_h - _EDGE_EPS

    half_w = (x2 - x1) / 2.0
    half_h = (y2 - y1) / 2.0

    # Radius: prefer an axis with both ends intact; else the larger half-extent.
    radii = []
    if not left_clip and not right_clip:
        radii.append(half_w)
    if not top_clip and not bot_clip:
        radii.append(half_h)
    radius = max(radii) if radii else max(half_w, half_h)

    # Centre: use the full axis if intact, else extend inward from a real edge.
    if not left_clip and not right_clip:
        cx = (x1 + x2) / 2.0
    elif not left_clip:
        cx = x1 + radius
    elif not right_clip:
        cx = x2 - radius
    else:
        cx = (x1 + x2) / 2.0

    if not top_clip and not bot_clip:
        cy = (y1 + y2) / 2.0
    elif not top_clip:
        cy = y1 + radius
    elif not bot_clip:
        cy = y2 - radius
    else:
        cy = (y1 + y2) / 2.0

    return cx, cy, radius


def ratio_from_scale(s: float) -> float:
    """Map ring scale (r_ring / R_disc) to approach ratio in [0, 1]."""
    return min(1.0, max(0.0, (S_START - s) / (S_START - 1.0)))


class BoxApproachEstimator:
    """
    Set ``approach_ratio`` on actionable detections from approach-ring boxes.

    Usage mirrors ``GeometricApproachEstimator`` so the pipeline can swap them::

        est = BoxApproachEstimator()
        est.reset()
        est.estimate(frame_detections, t_ms=ts)   # sets det.approach_ratio
    """

    def __init__(
        self,
        match_dist: float = 14.0,    # px (model-input space) for ring<->disc pairing
        temporal: bool = True,
        max_age_ms: float = 600.0,
        min_reliable: int = 2,
        reset_drop: float = 0.4,
        frame_w: float = 640.0,      # model-input frame size for edge-clip recovery
        frame_h: float = 384.0,
    ):
        self.match_dist = match_dist
        self.temporal = temporal
        self.max_age_ms = max_age_ms
        self.min_reliable = min_reliable
        self.reset_drop = reset_drop
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._tracks: List[dict] = []
        self._global_slope: Optional[float] = None  # ratio per ms

    def reset(self) -> None:
        self._tracks = []
        self._global_slope = None

    # ── public API ────────────────────────────────────────────────────────

    def estimate(
        self,
        detections: FrameDetections,
        t_ms: Optional[float] = None,
    ) -> None:
        """
        Pair each actionable object with its approach ring and set the ratio
        in place. ``detections`` must be the full :class:`FrameDetections` so
        the ``approach_circle`` boxes are available.
        """
        actionable = detections.actionable_objects
        rings = detections.approach_circles
        if not actionable:
            return

        # Edge-clip-corrected geometry for every ring (centre + true radius).
        ring_geo = [box_geometry(r, self.frame_w, self.frame_h) for r in rings]

        # Per-frame raw measurement by nearest concentric ring.
        raw: List[tuple] = []  # (det, ratio_or_None, found_ring)
        used = [False] * len(rings)
        for d in actionable:
            dcx, dcy, r_disc = box_geometry(d, self.frame_w, self.frame_h)
            best_j, best_dist = -1, self.match_dist
            for j, (rcx, rcy, _) in enumerate(ring_geo):
                if used[j]:
                    continue
                dist = ((rcx - dcx) ** 2 + (rcy - dcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j >= 0 and r_disc > 1e-3:
                used[best_j] = True
                s = ring_geo[best_j][2] / r_disc
                raw.append((d, ratio_from_scale(s), True))
            else:
                # No ring paired. Do NOT assume hit time (constant 1.0 caused
                # phantom taps): leave the measurement empty and let the
                # temporal filter carry this object's own history forward.
                raw.append((d, None, False))

        if t_ms is None or not self.temporal:
            for d, ratio, _ in raw:
                d.approach_ratio = ratio if ratio is not None else 0.0
            return

        self._temporal_update(raw, t_ms)

    # ── temporal filtering ────────────────────────────────────────────────

    def _temporal_update(self, raw: List[tuple], t_ms: float) -> None:
        self._tracks = [tr for tr in self._tracks
                        if (t_ms - tr["last_t"]) <= self.max_age_ms]

        assigned = []
        for d, ratio, found in raw:
            tr = self._associate(float(d.cx), float(d.cy))
            if found:
                # Monotonic guard: a large drop in a reliable reading means the
                # screen position has been recycled by a new (stacked) object.
                if tr["pts"] and (tr["last_out"] - ratio) > self.reset_drop:
                    tr["pts"] = []
                tr["pts"].append((t_ms, ratio, True))
            # Unmatched objects contribute no fabricated point; prediction below
            # falls back to the track's own carried history (last_out).
            tr["cx"], tr["cy"], tr["last_t"] = float(d.cx), float(d.cy), t_ms
            cutoff = t_ms - self.max_age_ms
            tr["pts"] = [p for p in tr["pts"] if p[0] >= cutoff]
            fallback = ratio if found else tr["last_out"]
            assigned.append((d, tr, fallback))

        self._recompute_slope()

        for d, tr, ratio in assigned:
            out = self._predict(tr, t_ms, ratio)
            tr["last_out"] = out
            d.approach_ratio = out

    def _associate(self, cx: float, cy: float) -> dict:
        best, best_d = None, self.match_dist
        for tr in self._tracks:
            dist = ((tr["cx"] - cx) ** 2 + (tr["cy"] - cy) ** 2) ** 0.5
            if dist < best_d:
                best_d, best = dist, tr
        if best is None:
            best = {"cx": cx, "cy": cy, "last_t": 0.0, "last_out": 0.0, "pts": []}
            self._tracks.append(best)
        return best

    def _recompute_slope(self) -> None:
        """Pool a robust global slope (ratio per ms) across reliable tracks."""
        slopes = []
        for tr in self._tracks:
            pts = [(t, r) for (t, r, ok) in tr["pts"] if ok]
            if len(pts) < 2:
                continue
            t0, r0 = pts[0]
            t1, r1 = pts[-1]
            if t1 > t0 and r1 > r0:
                slopes.append((r1 - r0) / (t1 - t0))
        if slopes:
            slopes.sort()
            self._global_slope = slopes[len(slopes) // 2]  # median

    def _predict(self, tr: dict, t_ms: float, raw_ratio: float) -> float:
        reliable = [(t, r) for (t, r, ok) in tr["pts"] if ok]
        if self._global_slope is not None and len(reliable) >= self.min_reliable:
            # Fix the phase (intercept) from reliable points given the shared
            # slope, then evaluate the line at the current time.
            slope = self._global_slope
            intercept = sum(r - slope * t for t, r in reliable) / len(reliable)
            est = slope * t_ms + intercept
            return min(1.0, max(0.0, est))
        return raw_ratio
