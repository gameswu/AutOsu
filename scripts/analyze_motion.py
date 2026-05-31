#!/usr/bin/env python3
"""
Offline human-motion analysis: extract motion statistics from real replays and
bake them into a `MotionProfile` YAML for the deterministic vision-only
controller.

This runs **entirely offline** on matched (.osu beatmap, .osr replay) pairs.
The runtime player never reads beatmaps or replays — this script merely
distils *how humans move the cursor* into a handful of numbers
(`jitter`, `overshoot`, `follow_alpha`, `tap_lead_ms`) that tune
`src/control/motion.py` and `src/control/controller.py`.

Metrics
-------
* **tap_lead_ms** — how far *before* a circle / slider-head's hit time the
  player presses a key (median over all objects). Positive = taps early.
* **overshoot** — fractional distance the cursor travels *past* a hit target
  on its final approach before settling back (median over real moves).
* **jitter** — RMS amplitude (osu!px) of high-frequency cursor tremor, i.e.
  the residual after subtracting a smoothed path.
* **follow_alpha** — slider-follow correction gain, estimated from how tightly
  the cursor tracks a smoothed (ball-like) path while a slider is held.

Usage::

    python scripts/analyze_motion.py --data raw_data --output configs/motion_profile.yaml
    python scripts/analyze_motion.py --data raw_data --max-replays 50
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

Vec = Tuple[float, float]


# ── small geometry helpers ────────────────────────────────────────────────

def _dist(a: Vec, b: Vec) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _smooth(values: List[float], half: int) -> List[float]:
    """Centred moving average (index-based) with a +/- ``half`` window."""
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = sum(values[lo:hi]) / (hi - lo)
    return out


# ── per-replay metric extraction ──────────────────────────────────────────

def _hit_window_ms(od: float) -> float:
    """osu!std 50-hit window (ms): 200 - 10*OD. Used as the tap-search radius."""
    return max(60.0, 200.0 - 10.0 * od)


def _collect_tap_lead(replay, hit_times: List[int], od: float,
                      out: List[float]) -> None:
    """For each hit object, match the nearest key-down edge and record the lead."""
    frames = replay.frames
    if not frames:
        return
    # Key-down edges (either key going False -> True).
    edges: List[int] = []
    prev_z = prev_x = False
    for f in frames:
        if (f.key_z and not prev_z) or (f.key_x and not prev_x):
            edges.append(f.time_ms)
        prev_z, prev_x = f.key_z, f.key_x
    if not edges:
        return

    window = _hit_window_ms(od)
    j = 0
    for T in hit_times:
        # Advance a pointer to the first edge >= T - window (edges are sorted).
        while j < len(edges) and edges[j] < T - window:
            j += 1
        # Scan the small neighbourhood for the closest edge to T.
        best = None
        k = j
        while k < len(edges) and edges[k] <= T + window:
            d = abs(edges[k] - T)
            if best is None or d < abs(best - T):
                best = edges[k]
            k += 1
        if best is not None:
            out.append(float(T - best))   # >0: pressed before the hit time


def _collect_overshoot(replay, targets: List[Tuple[int, float, float]],
                       out: List[float]) -> None:
    """Measure fractional overshoot on the final approach to each target."""
    for T, tx, ty in targets:
        win = replay.frames_in_range(T - 250, T + 40)
        if len(win) < 4:
            continue
        s = (win[0].x, win[0].y)          # approach start
        g = (tx, ty)                      # target
        L = _dist(s, g)
        if L < 30.0:                      # stacked / no real movement
            continue
        axis = ((g[0] - s[0]) / L, (g[1] - s[1]) / L)
        # Normalised projection onto the start->target axis (target == 1.0).
        peak = 0.0
        for f in win:
            proj = ((f.x - s[0]) * axis[0] + (f.y - s[1]) * axis[1]) / L
            if proj > peak:
                peak = proj
        out.append(max(0.0, peak - 1.0))


def _collect_jitter(replay, out: List[float]) -> None:
    """RMS amplitude of the high-frequency residual of the cursor path."""
    frames = replay.frames
    if len(frames) < 16:
        return
    xs = [f.x for f in frames]
    ys = [f.y for f in frames]
    sx, sy = _smooth(xs, 3), _smooth(ys, 3)
    acc = 0.0
    n = 0
    for i in range(len(frames)):
        rx, ry = xs[i] - sx[i], ys[i] - sy[i]
        acc += rx * rx + ry * ry
        n += 1
    if n:
        out.append((acc / n) ** 0.5)


def _collect_follow_alpha(replay, sliders: List[Tuple[int, int]],
                          out: List[float]) -> None:
    """Estimate the slider-follow correction gain alpha.

    During a held slider the cursor tracks the (smoothly moving) ball. We use a
    heavily smoothed copy of the cursor as a proxy for that ball path and fit
    ``c[t+1]-c[t] ~= alpha*(smoothed[t]-c[t])`` by least squares. This captures
    how tightly the player corrects toward the path, a reasonable proxy for the
    low-pass gain used by HumanMotion.follow().
    """
    for head_t, end_t in sliders:
        win = replay.frames_in_range(head_t + 20, end_t)
        # Require the key to actually be held through the slider.
        held = [f for f in win if f.key_z or f.key_x]
        if len(held) < 8:
            continue
        xs = [f.x for f in held]
        ys = [f.y for f in held]
        sx, sy = _smooth(xs, 4), _smooth(ys, 4)
        num = den = 0.0
        for i in range(len(held) - 1):
            ex, ey = sx[i] - xs[i], sy[i] - ys[i]      # error toward path
            dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]  # step taken
            num += dx * ex + dy * ey
            den += ex * ex + ey * ey
        if den > 1e-6:
            alpha = num / den
            if 0.0 < alpha < 1.5:
                out.append(min(0.9, max(0.2, alpha)))


# ── driver ────────────────────────────────────────────────────────────────

def analyze(data_dir: str, max_replays: Optional[int]) -> dict:
    from src.data.replay_parser import find_replay_pairs, parse_replay
    from src.data.osu_parser import OsuParser, HitCircle, Slider, Spinner

    pairs = find_replay_pairs(data_dir)
    if not pairs:
        print(f"ERROR: no (beatmap, replay) pairs found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    tap_lead: List[float] = []
    overshoot: List[float] = []
    jitter: List[float] = []
    follow_alpha: List[float] = []

    n_replays = 0
    for osu_path, osr_paths in pairs:
        try:
            bm = OsuParser.parse(osu_path)
        except Exception as e:
            print(f"  skip beatmap {osu_path.name}: {e}")
            continue
        od = bm.difficulty.od

        # Tap targets = circles + slider heads (spinners excluded — no tap).
        hit_times: List[int] = []
        approach_targets: List[Tuple[int, float, float]] = []
        sliders: List[Tuple[int, int]] = []
        for o in bm.hit_objects:
            if isinstance(o, HitCircle):
                hit_times.append(o.time)
                approach_targets.append((o.time, float(o.x), float(o.y)))
            elif isinstance(o, Slider):
                hit_times.append(o.time)
                approach_targets.append((o.time, float(o.x), float(o.y)))
                if o.end_time > o.time:
                    sliders.append((o.time, o.end_time))
        hit_times.sort()

        for osr_path in osr_paths:
            if max_replays is not None and n_replays >= max_replays:
                break
            try:
                replay = parse_replay(osr_path)
            except Exception as e:
                print(f"  skip replay {osr_path.name}: {e}")
                continue
            n_replays += 1
            print(f"  [{n_replays}] {osu_path.parent.name}/{osr_path.name} "
                  f"({len(replay.frames)} frames)")

            _collect_tap_lead(replay, hit_times, od, tap_lead)
            _collect_overshoot(replay, approach_targets, overshoot)
            _collect_jitter(replay, jitter)
            _collect_follow_alpha(replay, sliders, follow_alpha)

        if max_replays is not None and n_replays >= max_replays:
            break

    if n_replays == 0:
        print("ERROR: no replays could be parsed", file=sys.stderr)
        sys.exit(1)

    def _median(vals: List[float], default: float) -> float:
        return float(statistics.median(vals)) if vals else default

    profile = {
        "jitter": round(_median(jitter, 1.2), 3),
        "overshoot": round(_median(overshoot, 0.0), 4),
        "follow_alpha": round(_median(follow_alpha, 0.45), 3),
        "tap_lead_ms": round(_median(tap_lead, 0.0), 2),
    }

    print("\n── samples ─────────────────────────────")
    print(f"  replays analysed : {n_replays}")
    print(f"  tap_lead samples : {len(tap_lead)}")
    print(f"  overshoot samples: {len(overshoot)}")
    print(f"  jitter samples   : {len(jitter)}")
    print(f"  follow samples   : {len(follow_alpha)}")
    return profile


def main():
    ap = argparse.ArgumentParser(
        description="Extract a human MotionProfile from real osu! replays.")
    ap.add_argument("--data", "-d", default="raw_data",
                    help="raw_data dir (beatmaps/ + replays/)")
    ap.add_argument("--output", "-o", default="configs/motion_profile.yaml",
                    help="output MotionProfile YAML path")
    ap.add_argument("--max-replays", "-n", type=int, default=None,
                    help="cap the number of replays analysed")
    args = ap.parse_args()

    profile = analyze(args.data, args.max_replays)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        with open(out, "w", encoding="utf-8") as f:
            f.write("# Human motion profile baked from real replays by\n")
            f.write("# scripts/analyze_motion.py. Loaded via motion_profile_path.\n")
            yaml.safe_dump(profile, f, sort_keys=True)
    except ImportError:
        with open(out, "w", encoding="utf-8") as f:
            f.write("# Human motion profile (scripts/analyze_motion.py)\n")
            for k, v in sorted(profile.items()):
                f.write(f"{k}: {v}\n")

    print("\n── motion profile ──────────────────────")
    for k, v in sorted(profile.items()):
        print(f"  {k}: {v}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
