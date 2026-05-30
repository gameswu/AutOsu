"""
Geometric approach-ratio estimator (traditional CV, no neural network).

The osu! approach circle is a bright ring, concentric with the hitcircle,
that shrinks from 4.0x the hitcircle radius (object just appeared) down to
1.0x (hit time). Because we render / play with **no background** and a single
fixed skin, the approach ring is the only large bright ring around each
detected object, so we can measure its radius directly and convert it to an
approach ratio with exact geometry — no learning, no train/skew/domain-gap.

Pipeline per detected circle / slider-head:
  1. Crop a square region (sized from the YOLO bbox) around the object centre.
  2. Polar-unwrap around the centre (cv2.warpPolar) so concentric rings become
     horizontal bands.
  3. Average over angle -> 1-D radial intensity profile (robust to the combo
     number and to a neighbour object intruding into part of the ring).
  4. Read the inner disc edge radius  R  and the outer approach-ring radius
     r_approach  from the profile.
  5. ratio = (S_START - r_approach / R) / (S_START - 1)   with S_START = 4.0.

ratio = 0 -> just appeared (ring at 4.0x),  ratio = 1 -> hit time (ring on disc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Approach circle starts at S_START * hitcircle_radius and shrinks to 1.0 *.
# Matches real osu! stable AND our patched osr2mp4 renderer (Circles.py).
S_START = 4.0

# How far out (in hitcircle radii) to crop so the full ring always fits.
CROP_RADII = 4.6

# Number of angular samples for the polar unwrap.
N_ANGLES = 180


@dataclass
class ApproachMeasurement:
    """Result of a single geometric approach-ratio measurement."""
    ratio: float            # final approach_ratio in [0, 1]
    r_disc: float           # measured hitcircle disc radius (crop px)
    r_approach: float       # measured approach-ring radius (crop px)
    confidence: float       # 0..1 heuristic confidence
    found_ring: bool        # whether a distinct outer ring was detected


def _radial_profile(
    frame: np.ndarray,
    cx: float,
    cy: float,
    max_radius: int,
) -> np.ndarray:
    """
    Return the 1-D radial intensity profile around (cx, cy).

    Uses the per-pixel max over BGR channels (the HSV "value") so that any
    saturated combo colour shows up as bright, then polar-unwraps and averages
    over angle.

    Args:
        frame: HxWx3 BGR uint8
        cx, cy: centre in frame pixels
        max_radius: outer radius to sample (pixels)

    Returns:
        profile: (max_radius,) float32, index = radius in pixels
    """
    # Value channel = max over BGR (bright for any saturated colour on black bg)
    value = frame.max(axis=2).astype(np.float32)

    # Polar unwrap: output[angle, radius]
    polar = cv2.warpPolar(
        value,
        (max_radius, N_ANGLES),       # dsize = (width=radius, height=angle)
        (float(cx), float(cy)),
        float(max_radius),
        cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR,
    )
    # Average over angle -> radial profile
    profile = polar.mean(axis=0)
    return profile


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple odd-window moving average."""
    if k <= 1 or x.size < k:
        return x
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")


def _find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end_inclusive) index runs where mask is True."""
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def measure_ratio(
    frame: np.ndarray,
    cx: float,
    cy: float,
    r_bbox: float,
    thr_frac: float = 0.30,
) -> ApproachMeasurement:
    """
    Measure the approach ratio for a single object via radial geometry.

    Args:
        frame: HxWx3 BGR uint8 (no background)
        cx, cy: object centre in frame pixels (from YOLO)
        r_bbox: hitcircle radius hint in frame pixels (from YOLO bbox/2)
        thr_frac: bright/dark threshold as fraction of (max-min) profile range

    Returns:
        ApproachMeasurement
    """
    r_bbox = max(2.0, float(r_bbox))
    max_radius = int(np.ceil(CROP_RADII * r_bbox))
    max_radius = max(8, max_radius)

    profile = _radial_profile(frame, cx, cy, max_radius)
    profile = _smooth(profile, k=3)

    pmin = float(profile.min())
    pmax = float(profile.max())
    rng = pmax - pmin

    # No meaningful brightness -> object likely already hit / gone.
    if rng < 8.0 or pmax < 16.0:
        return ApproachMeasurement(
            ratio=1.0, r_disc=r_bbox, r_approach=r_bbox,
            confidence=0.0, found_ring=False,
        )

    thr = pmin + thr_frac * rng
    above = profile > thr
    runs = _find_runs(above)

    # Drop tiny runs (noise), keep runs spanning >= 1 px
    runs = [(a, b) for (a, b) in runs if (b - a) >= 0]

    if not runs:
        return ApproachMeasurement(
            ratio=1.0, r_disc=r_bbox, r_approach=r_bbox,
            confidence=0.1, found_ring=False,
        )

    # Inner disc = first bright band starting near the centre.
    disc_start, disc_end = runs[0]
    r_disc = float(disc_end)
    # Fall back to bbox radius if the inner band is implausibly small/large.
    if r_disc < 0.4 * r_bbox or r_disc > 2.0 * r_bbox:
        r_disc = r_bbox

    # Outer ring = the outermost bright band beyond the disc, if separated by
    # a dark gap. If only one band exists, the ring has merged with the disc
    # (ratio ~ 1).
    if len(runs) >= 2:
        ring_start, ring_end = runs[-1]
        # Peak (argmax) within the outer band for sub-pixel-ish centre
        seg = profile[ring_start:ring_end + 1]
        r_approach = float(ring_start + int(np.argmax(seg)))
        found_ring = r_approach > r_disc + 1.0
    else:
        r_approach = r_disc
        found_ring = False

    if not found_ring:
        # Disc and ring merged -> at/after hit time.
        return ApproachMeasurement(
            ratio=1.0, r_disc=r_disc, r_approach=r_disc,
            confidence=0.5, found_ring=False,
        )

    s = r_approach / max(1.0, r_disc)
    ratio = (S_START - s) / (S_START - 1.0)
    ratio = float(min(1.0, max(0.0, ratio)))

    # Confidence: higher when the outer ring is clearly separated and bright.
    sep = (r_approach - r_disc) / max(1.0, r_bbox)
    confidence = float(min(1.0, 0.4 + 0.6 * min(1.0, sep)))

    return ApproachMeasurement(
        ratio=ratio, r_disc=r_disc, r_approach=r_approach,
        confidence=confidence, found_ring=True,
    )


class GeometricApproachEstimator:
    """
    Drop-in replacement for the old CNN ApproachEstimatorInference.

    Operates directly on the captured frame + YOLO detections, setting
    `detection.approach_ratio` in place. No model file, no GPU.

    Temporal mode (default, when a frame timestamp is supplied)
    -----------------------------------------------------------
    osu! hit objects are **positionally static** — only the approach ring
    shrinks — and it shrinks at a **constant rate**: the ratio rises linearly
    from 0 (just appeared) to 1 (hit time) over ``time_preempt`` ms. Two
    consequences are exploited here:

      * The same object is trivially tracked frame-to-frame by its centre.
      * The slope of ratio-vs-time ( = 1 / time_preempt ) is *identical for
        every object* in a beatmap, so we pool a robust **global slope**
        across all tracks. Each object then only needs its **phase** (line
        intercept), fixed from its own reliable (ring-found) measurements.

    This denoises the single-frame measurement, which is least reliable at low
    ratios (ring far out and faint) — exactly the regime the raw geometry
    over-estimates.
    """

    def __init__(
        self,
        s_start: float = S_START,
        temporal: bool = True,
        match_dist: float = 8.0,        # px; objects are static, so small
        max_age_ms: float = 600.0,      # forget a track unseen this long
        conf_thresh: float = 0.4,       # min confidence for a "reliable" point
        min_reliable: int = 1,          # reliable pts needed to trust the fit
        reset_drop: float = 0.4,        # raw-ratio drop that signals a new object
    ):
        self.s_start = s_start
        self.temporal = temporal
        self.match_dist = match_dist
        self.max_age_ms = max_age_ms
        self.conf_thresh = conf_thresh
        self.min_reliable = min_reliable
        self.reset_drop = reset_drop
        self._tracks: List[dict] = []
        self._global_slope: Optional[float] = None  # ratio per ms

    def reset(self) -> None:
        """Clear all tracking state (call between independent sequences)."""
        self._tracks = []
        self._global_slope = None

    def estimate(
        self, frame: np.ndarray, detections: Sequence, t_ms: Optional[float] = None
    ) -> None:
        """
        Set `approach_ratio` on each detection in-place.

        Args:
            frame: HxWx3 BGR uint8 (model-input resolution, no background)
            detections: iterable of objects with .cx, .cy, .w, .h,
                        .approach_ratio attributes
            t_ms: frame timestamp (ms). If None or temporal disabled, falls
                  back to pure per-frame geometric measurement.
        """
        measures = [
            measure_ratio(frame, d.cx, d.cy, 0.25 * (float(d.w) + float(d.h)))
            for d in detections
        ]

        if t_ms is None or not self.temporal:
            for d, m in zip(detections, measures):
                d.approach_ratio = m.ratio
            return

        self._prune(t_ms)

        assigned = []
        for d, m in zip(detections, measures):
            tr = self._associate(float(d.cx), float(d.cy))
            reliable = m.found_ring and m.confidence >= self.conf_thresh
            # Approach ratio is monotonic non-decreasing for a given object.
            # A large drop in a *reliable* reading versus this track's last
            # output means the position has been recycled by a new object
            # (stacked / streamed notes). We require reliability so that a
            # transient glitch near hit time (ring merging into the disc,
            # found_ring=False) does not wipe an otherwise good line.
            if tr["pts"] and reliable and (tr["last_out"] - m.ratio) > self.reset_drop:
                tr["pts"] = []
            tr["cx"], tr["cy"], tr["last_t"] = float(d.cx), float(d.cy), t_ms
            tr["pts"].append((t_ms, m.ratio, reliable))
            cutoff = t_ms - self.max_age_ms
            tr["pts"] = [p for p in tr["pts"] if p[0] >= cutoff]
            assigned.append((d, tr, m))

        self._recompute_slope()

        for d, tr, m in assigned:
            ratio = self._predict(tr, t_ms, m.ratio)
            tr["last_out"] = ratio
            d.approach_ratio = ratio

    # ── tracking internals ────────────────────────────────────────────────

    def _prune(self, t_ms: float) -> None:
        self._tracks = [
            tr for tr in self._tracks if t_ms - tr["last_t"] <= self.max_age_ms
        ]

    def _associate(self, cx: float, cy: float) -> dict:
        """Return the nearest existing track within match_dist, else a new one."""
        best, best_d2 = None, self.match_dist * self.match_dist
        for tr in self._tracks:
            d2 = (tr["cx"] - cx) ** 2 + (tr["cy"] - cy) ** 2
            if d2 <= best_d2:
                best_d2, best = d2, tr
        if best is None:
            best = {"cx": cx, "cy": cy, "last_t": 0.0, "pts": [], "last_out": 0.0}
            self._tracks.append(best)
        return best

    def _recompute_slope(self) -> None:
        """
        Robust global slope (ratio per ms) = median of within-track pairwise
        slopes over reliable points. Shared by all objects (constant shrink
        rate). Only positive slopes are kept (ratio increases over time).
        """
        slopes: List[float] = []
        for tr in self._tracks:
            rel = [(t, r) for (t, r, ok) in tr["pts"] if ok]
            for i in range(len(rel)):
                for j in range(i + 1, len(rel)):
                    dt = rel[j][0] - rel[i][0]
                    if dt > 1e-3:
                        s = (rel[j][1] - rel[i][1]) / dt
                        if s > 0:
                            slopes.append(s)
        if len(slopes) >= 3:
            self._global_slope = float(np.median(slopes))

    def _predict(self, tr: dict, t_ms: float, fallback: float) -> float:
        """
        Predict ratio at t_ms using the global slope and this track's phase.
        Falls back to the raw per-frame measurement when there is not yet
        enough reliable history.
        """
        if self._global_slope is None:
            return fallback
        rel = [(t, r) for (t, r, ok) in tr["pts"] if ok]
        if len(rel) < self.min_reliable:
            return fallback
        m = self._global_slope
        # Per-object intercept: median of (ratio_i - m * t_i) over reliable pts.
        intercepts = [r - m * t for (t, r) in rel]
        c = float(np.median(intercepts))
        pred = m * t_ms + c
        return float(min(1.0, max(0.0, pred)))

    # Convenience for single-object measurement / diagnostics.
    def measure(
        self, frame: np.ndarray, cx: float, cy: float, r_bbox: float
    ) -> ApproachMeasurement:
        return measure_ratio(frame, cx, cy, r_bbox)
