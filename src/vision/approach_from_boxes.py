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

An actionable object with no matched ring is treated as ``ratio = 1`` (the ring
has collapsed onto the disc, i.e. the hit is imminent).

An optional temporal filter exploits the fact that the ratio rises linearly at
``1 / time_preempt`` for *every* object in a map: a robust global slope is
pooled across tracks and each object only needs its own phase. This denoises
the least-reliable low-ratio readings.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.vision.detector import Detection, FrameDetections, ObjClass

S_START = 4.0  # ring scale at spawn (matches osu!lazer DrawableHitCircle Scale=4)


def _radius(det: Detection) -> float:
    """Mean half-extent of a box, in box-coordinate pixels."""
    return 0.25 * (float(det.w) + float(det.h))


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
    ):
        self.match_dist = match_dist
        self.temporal = temporal
        self.max_age_ms = max_age_ms
        self.min_reliable = min_reliable
        self.reset_drop = reset_drop
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

        # Per-frame raw measurement by nearest concentric ring.
        raw: List[tuple] = []  # (det, ratio, found_ring)
        used = [False] * len(rings)
        for d in actionable:
            r_disc = _radius(d)
            best_j, best_dist = -1, self.match_dist
            for j, ring in enumerate(rings):
                if used[j]:
                    continue
                dist = ((ring.cx - d.cx) ** 2 + (ring.cy - d.cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j >= 0 and r_disc > 1e-3:
                used[best_j] = True
                s = _radius(rings[best_j]) / r_disc
                raw.append((d, ratio_from_scale(s), True))
            else:
                # No ring -> collapsed onto the disc -> imminent hit.
                raw.append((d, 1.0, False))

        if t_ms is None or not self.temporal:
            for d, ratio, _ in raw:
                d.approach_ratio = ratio
            return

        self._temporal_update(raw, t_ms)

    # ── temporal filtering ────────────────────────────────────────────────

    def _temporal_update(self, raw: List[tuple], t_ms: float) -> None:
        self._tracks = [tr for tr in self._tracks
                        if (t_ms - tr["last_t"]) <= self.max_age_ms]

        assigned = []
        for d, ratio, found in raw:
            tr = self._associate(float(d.cx), float(d.cy))
            # Monotonic guard: a large drop in a reliable reading means the
            # screen position has been recycled by a new (stacked) object.
            if tr["pts"] and found and (tr["last_out"] - ratio) > self.reset_drop:
                tr["pts"] = []
            tr["cx"], tr["cy"], tr["last_t"] = float(d.cx), float(d.cy), t_ms
            tr["pts"].append((t_ms, ratio, found))
            cutoff = t_ms - self.max_age_ms
            tr["pts"] = [p for p in tr["pts"] if p[0] >= cutoff]
            assigned.append((d, tr, ratio))

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
