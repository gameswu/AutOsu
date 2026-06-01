#!/usr/bin/env python3
"""
Estimate controller parameters from real osu! replays.

Analyses matched (beatmap, replay) pairs and extracts statistics that map
directly to the runtime controller parameters::

    python scripts/estimate_params.py -d raw_data
    python scripts/estimate_params.py -c configs/default.yaml -n 30

Measured parameters:
    max_speed_osu_pms      99th-percentile cursor speed
    max_accel_osu_pms2     99th-percentile cursor acceleration
    seek_tau_ms            exponential approach time-constant (fitted)
    tap_hold_ms            median key-down duration
    tap_refractory_ms      5th-percentile inter-tap gap
    spin_speed             median angular velocity during spinners (rad/ms)
    spin_radius_osu        median orbital radius during spinners
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── helpers ────────────────────────────────────────────────────────────────

def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _median(vals: List[float]) -> float:
    return _percentile(vals, 50.0)


# ── per-replay extractors ─────────────────────────────────────────────────

def _collect_speed_accel(replay, speeds: List[float], accels: List[float]):
    """Frame-to-frame cursor speed and acceleration (osu!px/ms)."""
    frames = replay.frames
    prev_v = 0.0
    for i in range(1, len(frames)):
        dt = frames[i].time_ms - frames[i - 1].time_ms
        if dt < 1:
            continue
        dx = frames[i].x - frames[i - 1].x
        dy = frames[i].y - frames[i - 1].y
        v = (dx * dx + dy * dy) ** 0.5 / dt
        speeds.append(v)
        if i >= 2:
            a = abs(v - prev_v) / dt
            accels.append(a)
        prev_v = v


def _collect_tap_stats(replay, holds: List[float], gaps: List[float]):
    """Key-down durations and inter-tap gaps."""
    frames = replay.frames
    if not frames:
        return
    # Track key-down/up edges for both Z and X.
    prev_z = prev_x = False
    z_down_t = x_down_t = -1e9
    last_up_t = -1e9
    for f in frames:
        t = f.time_ms
        # Z edges
        if f.key_z and not prev_z:
            if last_up_t > 0:
                gaps.append(t - last_up_t)
            z_down_t = t
        if not f.key_z and prev_z and z_down_t > 0:
            holds.append(t - z_down_t)
            last_up_t = t
        # X edges
        if f.key_x and not prev_x:
            if last_up_t > 0:
                gaps.append(t - last_up_t)
            x_down_t = t
        if not f.key_x and prev_x and x_down_t > 0:
            holds.append(t - x_down_t)
            last_up_t = t
        prev_z, prev_x = f.key_z, f.key_x


def _collect_spinner_stats(replay, spinners, spin_speeds: List[float],
                           spin_radii: List[float]):
    """Angular velocity and orbital radius during spinner sections."""
    CX, CY = 256.0, 192.0
    for sp in spinners:
        win = replay.frames_in_range(sp.time + 50, sp.end_time)
        if len(win) < 8:
            continue
        prev_angle = None
        for f in win:
            dx, dy = f.x - CX, f.y - CY
            r = (dx * dx + dy * dy) ** 0.5
            if r < 10:
                prev_angle = None
                continue
            spin_radii.append(r)
            angle = math.atan2(dy, dx)
            if prev_angle is not None:
                da = angle - prev_angle
                # Unwrap
                if da > math.pi:
                    da -= 2 * math.pi
                elif da < -math.pi:
                    da += 2 * math.pi
                dt = f.dt_ms if f.dt_ms > 0 else 16.0
                spin_speeds.append(abs(da) / dt)
            prev_angle = angle


def _collect_seek_tau(replay, circles, preempt_ms: float,
                      taus: List[float]):
    """Fit exponential approach time-constant from circle approaches.

    For each circle, look at the last ~preempt window of cursor motion
    toward it and fit d(t) ~ d0 * exp(-t/tau).
    """
    for c in circles:
        tx, ty, hit_t = float(c.x), float(c.y), c.time
        # Window: [hit_t - preempt, hit_t]
        win = replay.frames_in_range(int(hit_t - preempt_ms * 0.8),
                                     int(hit_t + 20))
        if len(win) < 6:
            continue
        # Compute distances to target.
        dists = []
        times = []
        for f in win:
            d = ((f.x - tx) ** 2 + (f.y - ty) ** 2) ** 0.5
            dists.append(d)
            times.append(f.time_ms)
        # Need decreasing distance (actual approach). Find the peak and
        # use only the descent from there.
        peak_i = max(range(len(dists)), key=lambda i: dists[i])
        dists = dists[peak_i:]
        times = times[peak_i:]
        if len(dists) < 4 or dists[0] < 30:
            continue
        # Least-squares fit of ln(d) vs t => slope = -1/tau.
        # Filter out zero/near-zero distances.
        ln_d = []
        ts = []
        t0 = times[0]
        for d, t in zip(dists, times):
            if d > 5:
                ln_d.append(math.log(d))
                ts.append(t - t0)
        if len(ln_d) < 4:
            continue
        # Simple linear regression: ln_d = a + b*t, tau = -1/b.
        n = len(ln_d)
        st = sum(ts)
        sld = sum(ln_d)
        stt = sum(t * t for t in ts)
        stld = sum(t * ld for t, ld in zip(ts, ln_d))
        denom = n * stt - st * st
        if abs(denom) < 1e-9:
            continue
        b = (n * stld - st * sld) / denom
        if b >= -1e-6:
            continue  # not decaying
        tau = -1.0 / b
        if 5.0 < tau < 200.0:
            taus.append(tau)


# ── driver ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Estimate controller parameters from real replays.")
    ap.add_argument("--config", "-c", default=None,
                    help="Path to config YAML (reads data.raw_data_dir)")
    ap.add_argument("--data", "-d", default=None,
                    help="raw_data dir (beatmaps/ + replays/)")
    ap.add_argument("--max-replays", "-n", type=int, default=None,
                    help="cap the number of replays analysed")
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

    data_dir = (args.data
                or cfg.get("data", {}).get("raw_data_dir")
                or "raw_data")

    from src.data.replay_parser import find_replay_pairs, parse_replay
    from src.data.osu_parser import OsuParser, HitCircle, Slider, Spinner

    pairs = find_replay_pairs(data_dir)
    if not pairs:
        print(f"ERROR: no (beatmap, replay) pairs in {data_dir}", file=sys.stderr)
        sys.exit(1)

    speeds: List[float] = []
    accels: List[float] = []
    holds: List[float] = []
    gaps: List[float] = []
    spin_speeds: List[float] = []
    spin_radii: List[float] = []
    taus: List[float] = []
    n_replays = 0

    for osu_path, osr_paths in pairs:
        try:
            bm = OsuParser.parse(osu_path)
        except Exception as e:
            print(f"  skip {osu_path.name}: {e}")
            continue
        if bm.metadata.mode != 0:
            continue

        circles = [o for o in bm.hit_objects if isinstance(o, HitCircle)]
        sliders = [o for o in bm.hit_objects if isinstance(o, Slider)]
        spinners = [o for o in bm.hit_objects if isinstance(o, Spinner)]
        preempt = bm.difficulty.preempt_ms
        # Include slider heads as approach targets too.
        approach_targets = circles + sliders

        for osr_path in osr_paths:
            if args.max_replays is not None and n_replays >= args.max_replays:
                break
            try:
                replay = parse_replay(osr_path)
            except Exception as e:
                print(f"  skip {osr_path.name}: {e}")
                continue
            if len(replay.frames) < 30:
                continue
            n_replays += 1
            print(f"  [{n_replays}] {osu_path.parent.name}/{osr_path.name} "
                  f"({len(replay.frames)} frames)")

            _collect_speed_accel(replay, speeds, accels)
            _collect_tap_stats(replay, holds, gaps)
            _collect_spinner_stats(replay, spinners, spin_speeds, spin_radii)
            _collect_seek_tau(replay, approach_targets, preempt, taus)

        if args.max_replays is not None and n_replays >= args.max_replays:
            break

    if n_replays == 0:
        print("ERROR: no replays analysed", file=sys.stderr)
        sys.exit(1)

    # ── results ────────────────────────────────────────────────────────
    results = {}

    if speeds:
        v99 = _percentile(speeds, 99)
        results["max_speed_osu_pms"] = round(v99, 2)
    if accels:
        a99 = _percentile(accels, 99)
        results["max_accel_osu_pms2"] = round(a99, 3)
    if taus:
        results["seek_tau_ms"] = round(_median(taus), 1)
    if holds:
        results["tap_hold_ms"] = round(_median(holds), 1)
    if gaps:
        g5 = _percentile(gaps, 5)
        results["tap_refractory_ms"] = round(g5, 1)
    if spin_speeds:
        results["spin_speed"] = round(_median(spin_speeds), 4)
    if spin_radii:
        results["spin_radius_osu"] = round(_median(spin_radii), 1)

    # Which params affect the motion-policy dataset (seek_velocity / goal).
    _RETRAIN = {
        "max_speed_osu_pms", "seek_tau_ms", "spin_speed", "spin_radius_osu",
    }
    _RUNTIME_ONLY = {
        "max_accel_osu_pms2", "tap_hold_ms", "tap_refractory_ms",
    }

    print(f"\n── estimated parameters ({n_replays} replays) ─────────")
    print(f"  {'parameter':<24} {'value':>10}  {'samples':>7}  retrain?")
    print(f"  {'─' * 24} {'─' * 10}  {'─' * 7}  {'─' * 8}")
    for k, v in results.items():
        count = {
            "max_speed_osu_pms": len(speeds),
            "max_accel_osu_pms2": len(accels),
            "seek_tau_ms": len(taus),
            "tap_hold_ms": len(holds),
            "tap_refractory_ms": len(gaps),
            "spin_speed": len(spin_speeds),
            "spin_radius_osu": len(spin_radii),
        }.get(k, 0)
        tag = "YES" if k in _RETRAIN else "no"
        print(f"  {k:<24} {v:>10}  {count:>7}  {tag}")

    print(f"\n── YAML snippet ────────────────────────")
    print("# [!] = changing this requires: build_motion_dataset + train_motion")
    for k, v in results.items():
        marker = "  # [!]" if k in _RETRAIN else ""
        print(f"{k}: {v}{marker}")

    print(f"\nPaste into your config YAML or pass with -c to other scripts.")
    if any(k in _RETRAIN for k in results):
        print("\nParameters marked [!] affect the motion-policy dataset.")
        print("If you change them, re-run:")
        print("  python scripts/build_motion_dataset.py -c <config>")
        print("  python scripts/train_motion.py -c <config>")


if __name__ == "__main__":
    main()
