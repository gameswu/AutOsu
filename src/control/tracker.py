"""
Timing tracker for the deterministic controller.

osu! gives every hit object in a beatmap the **same** approach duration
(``TimePreempt``): the approach ratio rises linearly from 0 (spawn) to 1 (hit
time) at rate ``1 / TimePreempt``. We do not know that constant at inference
(vision only), so we estimate it online from how fast the ratio of a tracked
object grows between frames, pooling a single robust value across all objects.

From that we can convert any object's current ratio into a **time-to-hit**::

    time_to_hit_ms = (1 - ratio) * preempt_ms

which the motion planner uses to arrive exactly on the beat.
"""

from __future__ import annotations

from typing import List, Optional

# Reasonable osu! preempt bounds (AR10 ~= 450ms, AR0 ~= 1800ms). Used to clamp
# noisy online estimates and as the initial guess before enough data arrives.
_PREEMPT_MIN = 300.0
_PREEMPT_MAX = 2000.0
_PREEMPT_DEFAULT = 800.0


class TimingTracker:
    """Online estimator of the shared approach preempt (ms)."""

    def __init__(self, match_dist: float = 24.0, ema: float = 0.1):
        self.match_dist = match_dist        # osu!px to match an object frame-to-frame
        self.ema = ema                      # smoothing for the pooled preempt
        self._preempt: float = _PREEMPT_DEFAULT
        self._tracks: List[dict] = []       # {x, y, ratio, t}

    def reset(self) -> None:
        self._preempt = _PREEMPT_DEFAULT
        self._tracks = []

    @property
    def preempt_ms(self) -> float:
        return self._preempt

    def time_to_hit_ms(self, ratio: float) -> float:
        """Estimated ms until ``ratio`` reaches 1.0 (the hit moment)."""
        return max(0.0, (1.0 - float(ratio)) * self._preempt)

    def update(self, objects: List[dict], t_ms: float) -> None:
        """
        Refine the preempt estimate.

        Args:
            objects: list of dicts with keys ``x``, ``y``, ``ratio`` (osu!px).
            t_ms: frame timestamp.
        """
        new_tracks: List[dict] = []
        for obj in objects:
            x, y, ratio = obj["x"], obj["y"], obj["ratio"]
            prev = self._match(x, y)
            if prev is not None:
                dt = t_ms - prev["t"]
                dr = ratio - prev["ratio"]
                # Only trust forward, in-range motion (avoid recycled positions
                # and the noisy ratio==1 plateau).
                if dt > 1.0 and 1e-3 < dr < 0.9 and 0.02 < ratio < 0.98:
                    slope = dr / dt                 # ratio per ms
                    preempt = 1.0 / slope
                    if _PREEMPT_MIN <= preempt <= _PREEMPT_MAX:
                        self._preempt = (1 - self.ema) * self._preempt + self.ema * preempt
            new_tracks.append({"x": x, "y": y, "ratio": ratio, "t": t_ms})
        self._tracks = new_tracks

    def _match(self, x: float, y: float) -> Optional[dict]:
        best, best_d = None, self.match_dist
        for tr in self._tracks:
            d = ((tr["x"] - x) ** 2 + (tr["y"] - y) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, tr
        return best
