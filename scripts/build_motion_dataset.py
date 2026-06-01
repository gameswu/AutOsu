#!/usr/bin/env python3
"""
Offline motion-policy dataset builder.

Reconstructs the scene frame-by-frame from beatmap ground truth (no rendering,
geometry only), pairs it with the real human cursor from the matched replay,
and records the supervised target — the **full human cursor velocity** (not a
residual) together with the scene context the trajectory model consumes.

For each timestep:
  - cursor_state:  (cx, cy, vx, vy, phase_one_hot×4)
  - targets:       variable-length list of (dx, dy, tth, ratio, circle, slider,
                   spinner, active), padded to MAX_TARGETS
  - target_mask:   bool mask for valid targets
  - velocity:      (vx, vy) in osu!px/ms — raw human cursor velocity

Output is a ``.npz`` with arrays ready for training::

    cursor_features: (N, CURSOR_DIM)
    target_features: (N, MAX_TARGETS, TARGET_DIM)
    target_masks:    (N, MAX_TARGETS)
    velocities:      (N, 2)

Usage::

    python scripts/build_motion_dataset.py -d raw_data -o runs/motion/dataset.npz
    python scripts/build_motion_dataset.py -d raw_data --fps 60 --max-replays 50
"""

from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
          ref_kwargs: Optional[dict] = None):
    import numpy as np
    from src.data.replay_parser import find_replay_pairs, parse_replay
    from src.data.osu_parser import OsuParser
    from src.control.reference import ReferenceController, build_targets, PHASE_SPIN
    from src.control.motion_net import (
        build_cursor_features, build_target_features, spin_tangential,
        CURSOR_DIM, TARGET_DIM, MAX_TARGETS,
    )

    pairs = find_replay_pairs(data_dir)
    if not pairs:
        print(f"ERROR: no (beatmap, replay) pairs found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    rk = ref_kwargs or {}
    dt = 1000.0 / float(fps)

    cursor_all: list = []
    target_all: list = []
    mask_all: list = []
    vel_all: list = []
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
        preempt = bm.difficulty.preempt_ms

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

            rc = ReferenceController(**rk)
            cur = track.at(lo)
            rc.reset(cursor=cur)
            prev_cur = cur
            prev_vel: Vec = (0.0, 0.0)
            start_idx = 0
            kept = 0

            tth_fn = lambda r, _p=preempt: (1.0 - r) * _p  # noqa: E731

            t = lo
            while t <= hi:
                cur = track.at(t)
                nxt = track.at(t + dt)
                scene, start_idx = _scene_at(objs, start_idx, t)
                ref = rc.step(scene, cur, t)

                if ref.phase != "idle":
                    vhx = (nxt[0] - cur[0]) / dt
                    vhy = (nxt[1] - cur[1]) / dt

                    # Spin: runtime keeps only the tangential component, so the
                    # label must live in the same subspace.  arrival_safeguard
                    # (approach/slide) is a runtime-only floor — a no-op on real
                    # human motion — so those labels stay pure human velocity.
                    if ref.phase == PHASE_SPIN:
                        center = next((t for t in ref.targets
                                       if t.kind == "spinner"), None)
                        if center is not None:
                            vhx, vhy = spin_tangential(
                                (vhx, vhy), cur, (center.x, center.y))

                    cur_vel = prev_vel
                    cf = build_cursor_features(cur, cur_vel, ref.phase)
                    tf, tm = build_target_features(cur, ref.targets, tth_fn)

                    # Pad / truncate to MAX_TARGETS.
                    n_tgt = len(tf)
                    if n_tgt < MAX_TARGETS:
                        tf.extend([[0.0] * TARGET_DIM] * (MAX_TARGETS - n_tgt))
                        tm.extend([False] * (MAX_TARGETS - n_tgt))
                    elif n_tgt > MAX_TARGETS:
                        tf = tf[:MAX_TARGETS]
                        tm = tm[:MAX_TARGETS]

                    cursor_all.append(cf)
                    target_all.append(tf)
                    mask_all.append(tm)
                    vel_all.append([vhx, vhy])

                    prev_vel = (vhx, vhy)
                    kept += 1
                else:
                    prev_vel = (0.0, 0.0)

                prev_cur = cur
                t += dt

            print(f"  [{n_replays}] {osu_path.parent.name}/{osr_path.name} "
                  f"-> {kept} samples")

        if max_replays is not None and n_replays >= max_replays:
            break

    if not cursor_all:
        print("ERROR: no samples produced", file=sys.stderr)
        sys.exit(1)

    return cursor_all, target_all, mask_all, vel_all, n_replays


def main():
    ap = argparse.ArgumentParser(
        description="Build a trajectory-model dataset from real osu! replays.")
    ap.add_argument("--config", "-c", default=None,
                    help="Path to config YAML (shares controller params with runtime)")
    ap.add_argument("--data", "-d", default="raw_data",
                    help="raw_data dir (beatmaps/ + replays/)")
    ap.add_argument("--output", "-o", default="runs/motion/dataset.npz",
                    help="output .npz path")
    ap.add_argument("--fps", type=int, default=60,
                    help="resampling rate for the reference simulation")
    ap.add_argument("--max-replays", "-n", type=int, default=None,
                    help="cap the number of replays processed")
    args = ap.parse_args()

    import yaml
    cfg: dict = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    ref_kwargs = {
        "hit_window": cfg.get("hit_window", 0.90),
        "tap_hold_ms": cfg.get("tap_hold_ms", 40.0),
        "tap_refractory_ms": cfg.get("tap_refractory_ms", 70.0),
        "slide_grace_ms": cfg.get("slide_grace_ms", 90.0),
        "spin_grace_ms": cfg.get("spin_grace_ms", 120.0),
    }

    data_dir = args.data
    if args.data == "raw_data":
        data_dir = cfg.get("data", {}).get("raw_data_dir", "raw_data")

    print(f"[build_motion] fps={args.fps}  ref_kwargs={ref_kwargs}")

    cursor_all, target_all, mask_all, vel_all, n_replays = build(
        data_dir, args.fps, args.max_replays, ref_kwargs=ref_kwargs,
    )

    import numpy as np
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        cursor_features=np.asarray(cursor_all, dtype=np.float32),
        target_features=np.asarray(target_all, dtype=np.float32),
        target_masks=np.asarray(mask_all, dtype=np.bool_),
        velocities=np.asarray(vel_all, dtype=np.float32),
        fps=np.int32(args.fps),
    )

    print("\n── dataset ─────────────────────────────")
    print(f"  replays      : {n_replays}")
    print(f"  samples      : {len(cursor_all)}")
    print(f"  cursor_dim   : {len(cursor_all[0])}")
    print(f"  max_targets  : {len(target_all[0])}")
    print(f"  target_dim   : {len(target_all[0][0])}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
