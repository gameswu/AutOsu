"""
Scene construction and target selection for the deterministic controller.

Turns raw :class:`FrameDetections` (model-input pixel boxes) into a compact
list of :class:`SceneObject` in osu! coordinates, pairing slider heads with
their ends and follow-balls, and exposes helpers to choose what the player
should be doing *right now*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from src.vision.detector import FrameDetections, ObjClass


@dataclass
class SceneObject:
    kind: str                       # "circle" | "slider" | "spinner"
    x: float                        # osu! coords
    y: float
    approach_ratio: float = 0.0
    # Slider-only extras (osu! coords)
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    ball_x: Optional[float] = None
    ball_y: Optional[float] = None
    has_ball: bool = False


@dataclass
class Scene:
    objects: List[SceneObject]
    spinner: Optional[SceneObject] = None
    # Raw slider-part positions (osu! coords), used as follow fallbacks while
    # sliding when the ball detection drops out.
    slider_ends: List[Tuple[float, float]] = field(default_factory=list)
    slider_bodies: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def actionables(self) -> List[SceneObject]:
        return [o for o in self.objects if o.kind in ("circle", "slider")]


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
        pair_dist_osu: max distance to associate a slider end / ball to a head.
    """
    ends = [to_osu(d.cx, d.cy) for d in detections.slider_ends]
    balls = [to_osu(d.cx, d.cy) for d in detections.slider_balls]
    bodies = [to_osu(d.cx, d.cy) for d in detections.slider_bodies]

    objects: List[SceneObject] = []

    for d in detections.hitcircles:
        ox, oy = to_osu(d.cx, d.cy)
        objects.append(SceneObject("circle", ox, oy, d.approach_ratio))

    for d in detections.slider_heads:
        ox, oy = to_osu(d.cx, d.cy)
        obj = SceneObject("slider", ox, oy, d.approach_ratio)
        ei = _nearest(ox, oy, ends, pair_dist_osu)
        if ei is not None:
            obj.end_x, obj.end_y = ends[ei]
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
    # the controller can keep following.
    used_balls = {(_round(o.ball_x), _round(o.ball_y))
                  for o in objects if o.has_ball}
    for (bx, by) in balls:
        if (_round(bx), _round(by)) in used_balls:
            continue
        objects.append(SceneObject("slider", bx, by, 1.0,
                                    ball_x=bx, ball_y=by, has_ball=True))

    return Scene(objects=objects, spinner=spinner,
                 slider_ends=ends, slider_bodies=bodies)


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
