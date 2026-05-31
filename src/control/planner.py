"""
Scene construction and target selection for the deterministic controller.

Turns raw :class:`FrameDetections` (model-input pixel boxes) into a compact
list of :class:`SceneObject` in osu! coordinates, pairing slider heads with
their follow-balls, and exposes helpers to choose what the player
should be doing *right now*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.vision.detector import FrameDetections, ObjClass


@dataclass
class SceneObject:
    kind: str                       # "circle" | "slider" | "spinner"
    x: float                        # osu! coords
    y: float
    approach_ratio: float = 0.0
    # True if this object's head/disc is visible this frame (so it is a valid
    # thing to *approach and tap*). Ball-only sliders (head already faded) are
    # follow-targets only, not tap targets.
    head_visible: bool = True
    # Slider-only extras (osu! coords)
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    has_ball: bool = False


@dataclass
class Scene:
    objects: List[SceneObject]
    spinner: Optional[SceneObject] = None

    @property
    def actionables(self) -> List[SceneObject]:
        # Only head-visible circles/sliders can be approached & tapped.
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
    """
    Build a :class:`Scene` in osu! coordinates.

    Args:
        detections: detections with ``approach_ratio`` already set.
        to_osu: maps a model-input (cx, cy) to osu! (x, y).
        pair_dist_osu: max distance to associate a slider ball to a head.
    """
    balls = [to_osu(d.cx, d.cy) for d in detections.slider_balls]

    objects: List[SceneObject] = []

    for d in detections.hitcircles:
        ox, oy = to_osu(d.cx, d.cy)
        objects.append(SceneObject("circle", ox, oy, d.approach_ratio))

    for d in detections.slider_heads:
        ox, oy = to_osu(d.cx, d.cy)
        obj = SceneObject("slider", ox, oy, d.approach_ratio)
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

    # If a slider is mid-flight, its follow-ball may be detected without a head
    # (head already hit / faded). Surface those as standalone slider objects so
    # the controller can keep following — but mark head_visible=False so they
    # are never chosen as a new tap target nor fed to the timing tracker.
    used_balls = {(_round(o.ball_x), _round(o.ball_y))
                  for o in objects if o.has_ball}
    for (bx, by) in balls:
        if (_round(bx), _round(by)) in used_balls:
            continue
        objects.append(SceneObject("slider", bx, by, 1.0,
                                    head_visible=False,
                                    ball_x=bx, ball_y=by, has_ball=True))

    return Scene(objects=objects, spinner=spinner)


def _round(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(round(v))


def select_target(scene: Scene) -> Optional[SceneObject]:
    """
    Pick the single object to act on this frame.

    Priority: the most imminent actionable (highest approach ratio). Spinners
    are handled separately by the controller, so they are not returned here.
    """
    actionables = scene.actionables
    if not actionables:
        return None
    return max(actionables, key=lambda o: o.approach_ratio)
