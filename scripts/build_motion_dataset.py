#!/usr/bin/env python3
"""
Offline motion dataset builder.

Reconstructs the scene frame-by-frame from beatmap ground truth (no rendering,
geometry only), pairs it with the real human cursor from the matched replay,
and records phase-routed supervision for the multi-task motion model.

For each frame we store the model input (cursor_features + padded targets) plus,
depending on the reference phase:

  * APPROACH — the movement-primitive boundary and the human path to clone::
        bnd      = [x0, y0, v0x, v0y, gx, gy, vgx, vgy, tau0]
                   x0      current human cursor
                   v0      τ-space velocity dx/dτ of the primary target
                   g       primary object position (goal)
                   vg      flow velocity = |v0| toward the next object
                   tau0    primary approach_ratio (phase parameter)
        path_tau = PATH_SAMPLES ratios uniformly in [tau0, 1]
        path_xy  = human cursor at those ratios (the supervised path)
        velocity = (0, 0)   (unused)

  * SLIDE / SPIN — the raw human velocity (osu!px/ms); spin is projected onto
    the tangential subspace to match the runtime ``spin_tangential`` constraint::
        velocity = (vx, vy)
        bnd / path_* = 0   (unused)

A boolean ``is_approach`` routes the loss in train_motion.py.

Output ``.npz`` arrays::

    cursor_features: (N, CURSOR_DIM)
    target_features: (N, MAX_TARGETS, TARGET_DIM)
    target_masks:    (N, MAX_TARGETS)
    is_approach:     (N,)
    bnd:             (N, 9)
    path_tau:        (N, PATH_SAMPLES)
    path_xy:         (N, PATH_SAMPLES, 2)
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

PATH_SAMPLES = 8   # ratios sampled over [tau0, 1] for the approach path label


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


# ── approach-primitive supervision helpers ─────────────────────────────────

def _primary_obj(objs: List[_ObjInfo], t: float) -> Optional[_ObjInfo]:
    """The object currently being approached: nearest upcoming head hit."""
    best: Optional[_ObjInfo] = None
    for o in objs:
        if o.kind not in ("circle", "slider"):
            continue
        if o.appear <= t <= o.hit:
            if best is None or o.hit < best.hit:
                best = o
    return best


def _next_obj(objs: List[_ObjInfo], primary: _ObjInfo) -> Optional[_ObjInfo]:
    """The object hit immediately after *primary* (for flow-aim v_g)."""
    best: Optional[_ObjInfo] = None
    for o in objs:
        if o.kind not in ("circle", "slider"):
            continue
        if o.hit > primary.hit:
            if best is None or o.hit < best.hit:
                best = o
    return best


def _approach_sample(primary: _ObjInfo, nxt: Optional[_ObjInfo],
                     cur: Vec, prev_cur: Vec, dt: float, t: float, track):
    """Build (bnd, path_tau, path_xy) for one approach frame at time *t*.

    bnd = [x0,y0, v0x,v0y, gx,gy, vgx,vgy, tau0].  Velocities are τ-space
    (dx/dτ); since τ = (t - appear)/preempt is linear in t, dτ = dt/preempt, so
    v0 = (cur - prev_cur) * preempt / dt.
    """
    preempt = primary.preempt
    tau0 = primary.ratio(t)
    g = (primary.x, primary.y)

    if preempt > 0 and dt > 0:
        scale = preempt / dt
        v0 = ((cur[0] - prev_cur[0]) * scale, (cur[1] - prev_cur[1]) * scale)
    else:
        v0 = (0.0, 0.0)

    # Flow aim: |v0| toward the next object.
    vg = (0.0, 0.0)
    if nxt is not None:
        dx, dy = nxt.x - g[0], nxt.y - g[1]
        d = (dx * dx + dy * dy) ** 0.5
        if d > 1e-6:
            sp = (v0[0] * v0[0] + v0[1] * v0[1]) ** 0.5
            vg = (sp * dx / d, sp * dy / d)

    bnd = [cur[0], cur[1], v0[0], v0[1], g[0], g[1], vg[0], vg[1], tau0]

    # Human path over the remaining ratio interval [tau0, 1].
    path_tau: List[float] = []
    path_xy: List[List[float]] = []
    span = 1.0 - tau0
    for k in range(PATH_SAMPLES):
        frac = k / (PATH_SAMPLES - 1) if PATH_SAMPLES > 1 else 1.0
        tau_k = tau0 + span * frac
        t_k = primary.appear + tau_k * preempt
        px, py = track.at(t_k)
        path_tau.append(tau_k)
        path_xy.append([px, py])
    return bnd, path_tau, path_xy


# ── driver ──────────────────────────────────────────────────────────────────

def build(data_dir: str, fps: int, max_replays: Optional[int],
          ref_kwargs: Optional[dict] = None):
    import numpy as np
    from src.data.replay_parser import find_replay_pairs, parse_replay
    from src.data.osu_parser import OsuParser
    from src.control.reference import (
        ReferenceController, build_targets,
        PHASE_APPROACH, PHASE_SPIN,
    )
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
    isapproach_all: list = []
    bnd_all: list = []
    ptau_all: list = []
    pxy_all: list = []
    vel_all: list = []
    n_replays = 0
    _ZERO_BND = [0.0] * 9
    _ZERO_PTAU = [0.0] * PATH_SAMPLES
    _ZERO_PXY = [[0.0, 0.0] for _ in range(PATH_SAMPLES)]

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
                    cf = build_cursor_features(cur, prev_vel, ref.phase)
                    tf, tm = build_target_features(cur, ref.targets, tth_fn)

                    # Pad / truncate to MAX_TARGETS.
                    n_tgt = len(tf)
                    if n_tgt < MAX_TARGETS:
                        tf.extend([[0.0] * TARGET_DIM] * (MAX_TARGETS - n_tgt))
                        tm.extend([False] * (MAX_TARGETS - n_tgt))
                    elif n_tgt > MAX_TARGETS:
                        tf = tf[:MAX_TARGETS]
                        tm = tm[:MAX_TARGETS]

                    # Human velocity (px/ms) — the slide/spin label and the next
                    # frame's cursor-feature velocity.
                    vhx = (nxt[0] - cur[0]) / dt
                    vhy = (nxt[1] - cur[1]) / dt

                    primary = _primary_obj(objs, t)
                    is_approach = (ref.phase == PHASE_APPROACH and primary is not None)

                    if is_approach:
                        # Movement-primitive supervision: clone the human path
                        # over the remaining ratio interval.
                        bnd, ptau, pxy = _approach_sample(
                            primary, _next_obj(objs, primary),
                            cur, prev_cur, dt, t, track)
                        cursor_all.append(cf)
                        target_all.append(tf)
                        mask_all.append(tm)
                        isapproach_all.append(True)
                        bnd_all.append(bnd)
                        ptau_all.append(ptau)
                        pxy_all.append(pxy)
                        vel_all.append([0.0, 0.0])
                        kept += 1
                    else:
                        # Slide / spin: velocity label.  Spin lives in the
                        # tangential subspace to match runtime spin_tangential.
                        if ref.phase == PHASE_SPIN:
                            center = next((tt for tt in ref.targets
                                           if tt.kind == "spinner"), None)
                            if center is not None:
                                vhx, vhy = spin_tangential(
                                    (vhx, vhy), cur, (center.x, center.y))
                        cursor_all.append(cf)
                        target_all.append(tf)
                        mask_all.append(tm)
                        isapproach_all.append(False)
                        bnd_all.append(list(_ZERO_BND))
                        ptau_all.append(list(_ZERO_PTAU))
                        pxy_all.append([list(p) for p in _ZERO_PXY])
                        vel_all.append([vhx, vhy])
                        kept += 1

                    prev_vel = (vhx, vhy)
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

    return (cursor_all, target_all, mask_all, isapproach_all,
            bnd_all, ptau_all, pxy_all, vel_all, n_replays)


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

    (cursor_all, target_all, mask_all, isapproach_all,
     bnd_all, ptau_all, pxy_all, vel_all, n_replays) = build(
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
        is_approach=np.asarray(isapproach_all, dtype=np.bool_),
        bnd=np.asarray(bnd_all, dtype=np.float32),
        path_tau=np.asarray(ptau_all, dtype=np.float32),
        path_xy=np.asarray(pxy_all, dtype=np.float32),
        velocities=np.asarray(vel_all, dtype=np.float32),
        fps=np.int32(args.fps),
    )

    n_ap = int(np.sum(isapproach_all))
    print("\n── dataset ─────────────────────────────")
    print(f"  replays      : {n_replays}")
    print(f"  samples      : {len(cursor_all)}")
    print(f"  approach     : {n_ap}  ({100*n_ap/max(1,len(cursor_all)):.1f}%)")
    print(f"  slide/spin   : {len(cursor_all) - n_ap}")
    print(f"  cursor_dim   : {len(cursor_all[0])}")
    print(f"  max_targets  : {len(target_all[0])}")
    print(f"  target_dim   : {len(target_all[0][0])}")
    print(f"  path_samples : {PATH_SAMPLES}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
