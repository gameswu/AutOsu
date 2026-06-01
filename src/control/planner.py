"""Scene construction and target selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.vision.detector import FrameDetections, ObjClass


@dataclass
class SceneObject:
    kind: str               # "circle" | "slider" | "spinner"
    x: float                # osu! coords
    y: float
    approach_ratio: float = 0.0
    head_visible: bool = True
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    has_ball: bool = False
    # Half-extent of the detection box in osu! coords (used to estimate CS).
    box_radius_osu: float = 0.0


@dataclass
class Scene:
    objects: List[SceneObject]
    spinner: Optional[SceneObject] = None
    has_slider_body: bool = False  # any SLIDER_BODY detection present

    @property
    def actionables(self) -> List[SceneObject]:
        return [o for o in self.objects
                if o.kind in ("circle", "slider") and o.head_visible]


def _nearest(px: float, py: float, candidates: List[Tuple[float, float]],
             max_dist: float) -> Optional[int]:
    best, best_d = -1, max_dist
    for i, (cx, cy) in enumerate(candidates):
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d < best_d:
            best_d, best = d, i
    return best if best >= 0 else None


def build_scene(
    detections: FrameDetections,
    to_osu: Callable[[float, float], Tuple[float, float]],
    pair_dist_osu: float = 320.0,
) -> Scene:
    """Build a Scene in osu! coordinates from raw detections."""
    balls = [to_osu(d.cx, d.cy) for d in detections.slider_balls]

    # Scale factor: model-input px -> osu! px (approximate, uniform).
    # to_osu maps (0,0)->(ox0,oy0) and (model_w, model_h)->(ox1,oy1).
    # We only need the ratio for converting box half-extents.
    _ox0, _ = to_osu(0.0, 0.0)
    _ox1, _ = to_osu(1.0, 0.0)
    px_to_osu = abs(_ox1 - _ox0)  # osu!px per model-input px

    objects: List[SceneObject] = []

    for d in detections.hitcircles:
        ox, oy = to_osu(d.cx, d.cy)
        br = max(d.w, d.h) / 2.0 * px_to_osu
        objects.append(SceneObject("circle", ox, oy, d.approach_ratio,
                                   box_radius_osu=br))

    for d in detections.slider_heads:
        ox, oy = to_osu(d.cx, d.cy)
        br = max(d.w, d.h) / 2.0 * px_to_osu
        obj = SceneObject("slider", ox, oy, d.approach_ratio, box_radius_osu=br)
        bi = _nearest(ox, oy, balls, pair_dist_osu)
        if bi is not None:
            obj.ball_x, obj.ball_y = balls[bi]
            obj.has_ball = True
        objects.append(obj)

    spinner: Optional[SceneObject] = None
    if detections.spinners:
        d = detections.spinners[0]
        ox, oy = to_osu(d.cx, d.cy)
        spinner = SceneObject("spinner", ox, oy, d.approach_ratio)

    # Orphan balls (head already faded): follow-only, never a tap target.
    used_balls = {(_round(o.ball_x), _round(o.ball_y))
                  for o in objects if o.has_ball}
    for bx, by in balls:
        if (_round(bx), _round(by)) in used_balls:
            continue
        objects.append(SceneObject("slider", bx, by, 1.0,
                                   head_visible=False,
                                   ball_x=bx, ball_y=by, has_ball=True))

    return Scene(objects=objects, spinner=spinner,
                has_slider_body=len(detections.slider_bodies) > 0)


def _round(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(round(v))


def select_target(scene: Scene) -> Optional[SceneObject]:
    """Pick the most imminent actionable (highest approach_ratio)."""
    actionables = scene.actionables
    if not actionables:
        return None
    return max(actionables, key=lambda o: o.approach_ratio)


def select_targets(scene: Scene, n: int = 3) -> List[SceneObject]:
    """Return up to *n* actionables sorted by approach_ratio descending."""
    return sorted(scene.actionables,
                  key=lambda o: o.approach_ratio, reverse=True)[:n]


def estimate_hit_radius(targets: List[SceneObject],
                        fallback: float = 36.0) -> float:
    """Estimate the hit-circle radius (osu!px) from detection box sizes.

    Uses the median box_radius_osu of visible actionables.  Falls back to
    *fallback* (CS4 ≈ 36 osu!px) when no boxes are available.
    """
    radii = [t.box_radius_osu for t in targets if t.box_radius_osu > 1e-3]
    if not radii:
        return fallback
    radii.sort()
    return radii[len(radii) // 2]
