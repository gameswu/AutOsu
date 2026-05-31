#!/usr/bin/env python3
"""
Offline motion-policy dataset builder.

Reconstructs the deterministic *navigation goal* frame-by-frame from beatmap
ground truth (no rendering, geometry only), pairs it with the real human cursor
from the matched replay, and records the supervised target — the **human cursor
velocity** (osu!px per ms)::

    velocity(t) = (human_cursor(t + dt) - human_cursor(t)) / dt

together with the goal-relative features the policy consumes
(:func:`src.control.motion_net.build_features`). Velocity (not per-frame
displacement) is the label so the policy is resampling-rate independent.

This runs **entirely offline** on matched (.osu, .osr) pairs. The runtime player
never reads beatmaps or replays — this only distils *how* humans move toward the
navigation goal into training data for ``scripts/train_motion.py``.

Output is a ``.npz`` with ``features`` (N, FEATURE_DIM) and ``velocity`` (N, 2,
osu!px/ms). Train with the **same** ``--max-speed`` the runtime ``MotionPolicy``
uses (default 4.0 osu!px/ms).

Usage::

    python scripts/build_motion_dataset.py --data raw_data --output runs/motion/dataset.npz
    python scripts/build_motion_dataset.py --data raw_data --fps 60 --max-replays 50
"""

from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

Vec = Tuple[float, float]


# ── replay cursor sampler ──────────────────────────────────────────────────

class _CursorTrack:
    """Linear-interpolating cursor lookup over sorted replay frames."""

    def __init__(self, replay):
        self._t = [f.time_ms for f in replay.frames]
        self._x = [f.x for f in replay.frames]
        self._y = [f.y for f in replay.frames]

    @property
    def span(self) -> Tuple[int, int]:
        return (self._t[0], self._t[-1])

    def at(self, t: float) -> Vec:
        ts = self._t
        if t <= ts[0]:
            return (self._x[0], self._y[0])
        if t >= ts[-1]:
            return (self._x[-1], self._y[-1])
        i = bisect.bisect_right(ts, t)
        t0, t1 = ts[i - 1], ts[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (self._x[i - 1] + (self._x[i] - self._x[i - 1]) * a,
                self._y[i - 1] + (self._y[i] - self._y[i - 1]) * a)


# ── beatmap -> per-frame scene reconstruction ──────────────────────────────

class _ObjInfo:
    """Precomputed timing/geometry for one hit object."""
    __slots__ = ("kind", "x", "y", "appear", "hit", "end", "preempt",
                 "path", "cum", "pixel_length", "slides")

    def __init__(self, kind, x, y, appear, hit, end, preempt,
                 path=None, cum=None, pixel_length=0.0, slides=1):
        self.kind = kind
        self.x = x
        self.y = y
        self.appear = appear
        self.hit = hit
        self.end = end
        self.preempt = preempt
        self.path = path
        self.cum = cum
        self.pixel_length = pixel_length
        self.slides = slides

    def ratio(self, t: float) -> float:
        if self.preempt <= 0:
            return 1.0
        return max(0.0, min(1.0, (t - self.appear) / self.preempt))

    def ball(self, t: float) -> Optional[Vec]:
        if self.kind != "slider" or self.path is None or self.slides <= 0:
            return None
        slide_dur = (self.end - self.hit) / self.slides
        if slide_dur <= 0:
            return None
        elapsed = t - self.hit
        idx = min(int(elapsed // slide_dur), self.slides - 1)
        frac = (elapsed - idx * slide_dur) / slide_dur
        if idx % 2 == 1:
            frac = 1.0 - frac
        from src.data.slider_path import position_at_distance
        return position_at_distance(self.path, self.cum, frac * self.pixel_length)


def _build_objects(bm) -> List[_ObjInfo]:
    from src.data.osu_parser import HitCircle, Slider, Spinner
    from src.data.slider_path import compute_slider_path

    preempt = bm.difficulty.preempt_ms
    out: List[_ObjInfo] = []
    for o in bm.hit_objects:
        if isinstance(o, HitCircle):
            out.append(_ObjInfo("circle", float(o.x), float(o.y),
                                o.time - preempt, o.time, o.time, preempt))
        elif isinstance(o, Slider):
            try:
                cps = [(p.x, p.y) for p in o.control_points]
                path, cum = compute_slider_path(int(o.curve_type), cps, o.pixel_length)
            except Exception:
                path, cum = None, None
            out.append(_ObjInfo("slider", float(o.x), float(o.y),
                                o.time - preempt, o.time, o.end_time, preempt,
                                path=path, cum=cum, pixel_length=o.pixel_length,
                                slides=max(1, o.slides)))
        elif isinstance(o, Spinner):
            out.append(_ObjInfo("spinner", 256.0, 192.0,
                                o.time, o.time, o.end_time, 0.0))
    out.sort(key=lambda i: i.appear)
    return out


def _scene_at(objs: List[_ObjInfo], start_idx: int, t: float):
    """Build a (Scene, new_start_idx) for time ``t``."""
    from src.control.planner import Scene, SceneObject

    # Advance past objects that have fully ended.
    while start_idx < len(objs) and objs[start_idx].end < t:
        start_idx += 1

    scene_objs: List[SceneObject] = []
    spinner: Optional[SceneObject] = None
    i = start_idx
    while i < len(objs) and objs[i].appear <= t:
        o = objs[i]
        i += 1
        if o.end < t:
            continue
        if o.kind == "circle":
            if o.appear <= t <= o.hit:
                scene_objs.append(SceneObject("circle", o.x, o.y, o.ratio(t)))
        elif o.kind == "slider":
            if t < o.hit:
                scene_objs.append(SceneObject("slider", o.x, o.y, o.ratio(t)))
            elif o.hit <= t <= o.end:
                b = o.ball(t)
                if b is not None:
                    scene_objs.append(SceneObject(
                        "slider", b[0], b[1], 1.0, head_visible=False,
                        ball_x=b[0], ball_y=b[1], has_ball=True))
        elif o.kind == "spinner":
            if o.hit <= t <= o.end and spinner is None:
                spinner = SceneObject("spinner", o.x, o.y, 1.0)
    return Scene(objects=scene_objs, spinner=spinner), start_idx


# ── driver ──────────────────────────────────────────────────────────────────

def build(data_dir: str, fps: int, max_replays: Optional[int],
          max_speed: float):
    from src.data.replay_parser import find_replay_pairs, parse_replay
    from src.data.osu_parser import OsuParser
    from src.control.reference import ReferenceController
    from src.control.motion_net import build_features

    pairs = find_replay_pairs(data_dir)
    if not pairs:
        print(f"ERROR: no (beatmap, replay) pairs found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    dt = 1000.0 / float(fps)
    feats_all: List[List[float]] = []
    vel_all: List[Tuple[float, float]] = []
    n_replays = 0

    for osu_path, osr_paths in pairs:
        try:
            bm = OsuParser.parse(osu_path)
        except Exception as e:
            print(f"  skip beatmap {osu_path.name}: {e}")
            continue
        objs = _build_objects(bm)
        if not objs:
            continue
        t_start = min(o.appear for o in objs)
        t_end = max(o.end for o in objs)

        for osr_path in osr_paths:
            if max_replays is not None and n_replays >= max_replays:
                break
            try:
                replay = parse_replay(osr_path)
            except Exception as e:
                print(f"  skip replay {osr_path.name}: {e}")
                continue
            if not replay.frames:
                continue
            n_replays += 1
            track = _CursorTrack(replay)
            rs, re = track.span
            lo = max(t_start, rs)
            hi = min(t_end, re)

            rc = ReferenceController()
            cur = track.at(lo)
            rc.reset(cursor=cur)
            prev = cur
            start_idx = 0
            kept = 0

            t = lo
            while t <= hi:
                cur = track.at(t)
                nxt = track.at(t + dt)
                scene, start_idx = _scene_at(objs, start_idx, t)
                ref = rc.step(scene, cur, t)
                if ref.phase != "idle":
                    vx = max(-max_speed, min(max_speed, (nxt[0] - cur[0]) / dt))
                    vy = max(-max_speed, min(max_speed, (nxt[1] - cur[1]) / dt))
                    feats_all.append(build_features(ref, cur, prev, dt))
                    vel_all.append((vx, vy))
                    kept += 1
                prev = cur
                t += dt

            print(f"  [{n_replays}] {osu_path.parent.name}/{osr_path.name} "
                  f"-> {kept} samples")

        if max_replays is not None and n_replays >= max_replays:
            break

    if not feats_all:
        print("ERROR: no samples produced", file=sys.stderr)
        sys.exit(1)

    return feats_all, vel_all, n_replays


def main():
    ap = argparse.ArgumentParser(
        description="Build a motion-policy dataset from real osu! replays.")
    ap.add_argument("--data", "-d", default="raw_data",
                    help="raw_data dir (beatmaps/ + replays/)")
    ap.add_argument("--output", "-o", default="runs/motion/dataset.npz",
                    help="output .npz path")
    ap.add_argument("--fps", type=int, default=60,
                    help="resampling rate for the reference simulation")
    ap.add_argument("--max-replays", "-n", type=int, default=None,
                    help="cap the number of replays processed")
    ap.add_argument("--max-speed", type=float, default=4.0,
                    help="clip |velocity| to this (osu!px/ms); MUST match "
                         "training / MotionPolicy.max_speed_osu_pms")
    args = ap.parse_args()

    feats, vel, n_replays = build(args.data, args.fps, args.max_replays,
                                  args.max_speed)

    import numpy as np
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=np.asarray(feats, dtype=np.float32),
        velocity=np.asarray(vel, dtype=np.float32),
        max_speed=np.float32(args.max_speed),
        fps=np.int32(args.fps),
    )

    print("\n── dataset ─────────────────────────────")
    print(f"  replays   : {n_replays}")
    print(f"  samples   : {len(feats)}")
    print(f"  max_speed : {args.max_speed} osu!px/ms")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
