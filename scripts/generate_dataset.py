#!/usr/bin/env python3
"""
Generate training data from raw_data/ (beatmaps + replays).

Produces a YOLO detection dataset (images + labels) with the cursor rendered.
Classes: 0 hitcircle, 1 slider_head, 2 slider_body, 3 slider_end, 4 spinner,
5 approach_circle (ring; box size encodes approach ratio), 6 slider_ball
(moving follow target during an active slider).

Input data structure::

    raw_data/
    +-- beatmaps/          <- .osz files (auto-extracted)
    |   +-- 123456 Artist - Title.osz
    |   +-- ...
    +-- replays/           <- .osr files
        +-- replay1.osr
        +-- ...

Matching is automatic via MD5 hash.

Usage::

    python scripts/generate_dataset.py --data raw_data --skin path/to/skin --output dataset
    python scripts/generate_dataset.py --data raw_data --skin path/to/skin --max-beatmaps 100
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Also ensure lib/ is on sys.path for osr2mp4
_LIB_DIR = str(Path(__file__).resolve().parent.parent / "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

import cv2
import numpy as np


def open_replay_frames(osu_path, osr_path, skin_path, width, height, fps):
    """
    Construct a renderer for one (beatmap, replay) pair and return a per-frame
    generator. Shared by dataset generation and the approach-geo validator.

    The renderer is constructed eagerly, so a bad replay raises here (the caller
    can catch and skip) rather than mid-iteration.

    Returns:
        (renderer, frame_iter) where frame_iter yields dicts with keys:
            frame (HxWx3 BGR uint8), t_ms, frame_idx,
            visible (list of non-fadeout hit objects),
            cursor_x, cursor_y, key_z, key_x,
            time_preempt, radius_osu
    """
    from src.data.renderer import Osr2mp4Renderer
    from osr2mp4.CheckSystem.Judgement import DiffCalculator
    from osr2mp4.EEnum.EReplay import Replays

    renderer = Osr2mp4Renderer(
        osu_path=str(osu_path),
        osr_path=str(osr_path),
        skin_path=str(skin_path),
        width=width,
        height=height,
        fps=fps,
    )

    time_preempt = renderer.time_preempt
    radius_osu = DiffCalculator(renderer.beatmap.diff).max_distance
    replay_data = renderer.replay_info.play_data

    def _gen():
        frame_idx = 0
        while renderer.has_frames():
            frame, t_ms = renderer.render_frame()
            frame_idx += 1

            osr_idx = min(renderer.frame_info.osr_index, len(replay_data) - 1)
            cursor_x = float(replay_data[osr_idx][Replays.CURSOR_X])
            cursor_y = float(replay_data[osr_idx][Replays.CURSOR_Y])
            keys = int(replay_data[osr_idx][Replays.KEYS_PRESSED])
            key_z = float((keys >> 2) & 1)  # K1
            key_x = float((keys >> 3) & 1)  # K2

            visible_raw = renderer.get_visible_objects()
            visible = [obj for obj, is_fadeout in visible_raw if not is_fadeout]

            yield {
                "frame": frame, "t_ms": t_ms, "frame_idx": frame_idx,
                "visible": visible,
                "cursor_x": cursor_x, "cursor_y": cursor_y,
                "key_z": key_z, "key_x": key_x,
                "time_preempt": time_preempt, "radius_osu": radius_osu,
            }

    return renderer, _gen()


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from raw_data (beatmaps + replays)"
    )
    parser.add_argument("--data", "-d", required=True,
                        help="raw_data directory (must contain beatmaps/ and replays/)")
    parser.add_argument("--skin", "-s", required=True,
                        help="Path to the osu! skin directory")
    parser.add_argument("--output", "-o", default="dataset",
                        help="Output directory (default: dataset)")
    parser.add_argument("--width", type=int, default=640,
                        help="Render width (default: 640)")
    parser.add_argument("--height", type=int, default=384,
                        help="Render height (default: 384)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Sampling FPS (default: 30)")
    parser.add_argument("--max-beatmaps", "-n", type=int, default=None,
                        help="Maximum number of beatmaps to use")
    parser.add_argument("--train-ratio", type=float, default=0.85,
                        help="Train/val split ratio (default: 0.85)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-balance", action="store_true",
                        help="Disable class-balancing oversampling of the train split")
    parser.add_argument("--balance-max-factor", type=int, default=8,
                        help="Max number of duplicates per image when balancing (default: 8)")
    args = parser.parse_args()

    from src.data.replay_parser import find_replay_pairs
    from src.data.renderer import Osr2mp4Renderer, PlayfieldTransform

    # Find (osu, [osr]) pairs by MD5 matching
    pairs = find_replay_pairs(args.data)
    if not pairs:
        print(f"ERROR: No (beatmap, replay) pairs found in {args.data}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} matched (beatmap, replay) groups")

    # Subsample if needed
    if args.max_beatmaps and len(pairs) > args.max_beatmaps:
        random.seed(args.seed)
        pairs = random.sample(pairs, args.max_beatmaps)
        print(f"Sampled {args.max_beatmaps} (seed={args.seed})")

    # Setup output dirs
    out = Path(args.output)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Coordinate transform for YOLO labels
    tf = PlayfieldTransform(args.width, args.height)

    random.seed(args.seed)
    total_frames = 0

    for pair_idx, (osu_path, osr_paths) in enumerate(pairs):
        print(f"\n[{pair_idx+1}/{len(pairs)}] {osu_path.parent.name}/{osu_path.name}")
        print(f"  Replays: {len(osr_paths)}")

        for osr_path in osr_paths:
            try:
                renderer, frame_iter = open_replay_frames(
                    osu_path, osr_path, args.skin,
                    args.width, args.height, args.fps,
                )
            except Exception as e:
                print(f"  Skip replay (renderer init error): {osr_path.name}: {e}")
                continue

            print(f"  Processing: {osr_path.name} (~{renderer.total_frames_estimate} frames)")

            # Train/val split per replay
            is_train = random.random() < args.train_ratio
            split = "train" if is_train else "val"

            for fd in frame_iter:
                frame, t_ms, frame_idx = fd["frame"], fd["t_ms"], fd["frame_idx"]
                visible = fd["visible"]
                time_preempt, radius_osu = fd["time_preempt"], fd["radius_osu"]

                if visible:
                    # Generate YOLO labels
                    labels = _generate_labels(
                        visible, t_ms, time_preempt, radius_osu,
                        tf, args.width, args.height,
                    )

                    if labels:
                        name = f"p{pair_idx:04d}_r{osr_paths.index(osr_path):02d}_f{frame_idx:06d}"
                        img_path = out / "images" / split / f"{name}.jpg"
                        lbl_path = out / "labels" / split / f"{name}.txt"

                        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        with open(lbl_path, "w") as f:
                            f.write("\n".join(labels))
                        total_frames += 1

                if frame_idx % 500 == 0:
                    print(f"    frame {frame_idx}, t={t_ms:.0f}ms, "
                          f"imgs={total_frames}")

    # Class-balance the train split by oversampling rare-class images
    if not args.no_balance:
        _balance_train_split(out, max_factor=args.balance_max_factor)

    # Write YOLO data.yaml
    _write_data_yaml(out)

    # Report final class distribution
    _print_class_histogram(out)

    print(f"\n{'=' * 60}")
    print(f"Dataset generation complete:")
    print(f"  Frames: {total_frames}")
    print(f"  Output: {out.resolve()}")


# ── Helpers ──────────────────────────────────────────────────────────────

def _obj_kind(obj):
    """Get simplified type string from osr2mp4 hitobject type list."""
    types = obj["type"]  # list like ['new combo', 'slider']
    if "spinner" in types:
        return "spinner"
    elif "slider" in types:
        return "slider"
    else:
        return "circle"


def _generate_labels(visible, t_ms, time_preempt, radius_osu, tf, w, h):
    """Generate YOLO label lines from visible osr2mp4 hit objects.

    Classes emitted:
        0 hitcircle, 1 slider_head, 2 slider_body, 3 slider_end, 4 spinner,
        5 approach_circle (only while approaching, t_ms < obj time),
        6 slider_ball (only while the slider is actively being followed).
    """
    lines = []
    half_disc_w = (radius_osu * 2 * tf.playfieldscale) / w
    half_disc_h = (radius_osu * 2 * tf.playfieldscale) / h

    for obj in visible:
        kind = _obj_kind(obj)
        ox, oy = obj["x"], obj["y"]

        # Convert osu!pixel to render pixel, then normalize
        rx, ry = tf.osu_to_render(ox, oy)
        cx_n = rx / w
        cy_n = ry / h
        bw_n = half_disc_w
        bh_n = half_disc_h

        if kind == "circle":
            lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")
            _append_approach_ring(lines, obj, t_ms, time_preempt, radius_osu,
                                  rx, ry, tf, w, h)

        elif kind == "slider":
            # Slider head
            lines.append(f"1 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")
            _append_approach_ring(lines, obj, t_ms, time_preempt, radius_osu,
                                  rx, ry, tf, w, h)

            # Slider body bounding box
            slider_c = obj.get("slider_c")
            if slider_c and hasattr(slider_c, "pos") and slider_c.pos:
                xs = [p[0] for p in slider_c.pos]
                ys = [p[1] for p in slider_c.pos]
                min_x = min(xs) - radius_osu
                max_x = max(xs) + radius_osu
                min_y = min(ys) - radius_osu
                max_y = max(ys) + radius_osu
                bcx = ((min_x + max_x) / 2 * tf.playfieldscale + tf.moveright) / w
                bcy = ((min_y + max_y) / 2 * tf.playfieldscale + tf.movedown) / h
                body_w = (max_x - min_x) * tf.playfieldscale / w
                body_h = (max_y - min_y) * tf.playfieldscale / h
                lines.append(f"2 {bcx:.6f} {bcy:.6f} {body_w:.6f} {body_h:.6f}")

            # Slider end
            end_x = obj.get("end x", ox)
            end_y = obj.get("end y", oy)
            erx, ery = tf.osu_to_render(end_x, end_y)
            ecx_n = erx / w
            ecy_n = ery / h
            lines.append(f"3 {ecx_n:.6f} {ecy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

            # Slider ball (moving follow point during active slide)
            ball = _slider_ball_pos(obj, t_ms)
            if ball is not None:
                brx, bry = tf.osu_to_render(ball[0], ball[1])
                lines.append(
                    f"6 {brx / w:.6f} {bry / h:.6f} {half_disc_w:.6f} {half_disc_h:.6f}"
                )

        elif kind == "spinner":
            # Spinner centered at playfield center
            scx, scy = tf.osu_to_render(256, 192)
            lines.append(f"4 {scx / w:.6f} {scy / h:.6f} 0.600000 0.800000")

    return list(dict.fromkeys(lines))


def _append_approach_ring(lines, obj, t_ms, time_preempt, radius_osu,
                          rx, ry, tf, w, h):
    """Emit a class-5 approach_circle label for an approaching hit object.

    The ring shrinks linearly from 4x the disc radius (just appeared) to 1x
    (hit time), matching osu!lazer's ``Scale=4 -> 1`` over TimePreempt. We only
    label it while it is actually larger than the disc (still approaching).
    """
    if time_preempt <= 0:
        return
    appear = obj["time"] - time_preempt
    dt = t_ms - appear
    ratio = dt / time_preempt
    if ratio < 0.0 or ratio >= 1.0:
        return  # not visible yet, or already collapsed onto the disc
    scale = 4.0 - 3.0 * ratio           # 4 -> 1
    ring_w = (radius_osu * 2 * scale * tf.playfieldscale) / w
    ring_h = (radius_osu * 2 * scale * tf.playfieldscale) / h
    lines.append(f"5 {rx / w:.6f} {ry / h:.6f} {ring_w:.6f} {ring_h:.6f}")


def _slider_ball_pos(obj, t_ms):
    """Position [x, y] (osu! coords) of the slider ball at time t_ms, or None.

    Mirrors osr2mp4's slider traversal: the ball walks the curve once per
    repeat, reversing direction on odd repeats (ping-pong).
    """
    if _obj_kind(obj) != "slider":
        return None
    slider_c = obj.get("slider_c")
    duration = obj.get("duration", 0)
    repeated = obj.get("repeated", 1)
    pixel_length = obj.get("pixel length")
    if not slider_c or duration <= 0 or pixel_length is None:
        return None

    start = obj["time"]
    end_time = start + duration * repeated
    if t_ms < start or t_ms > end_time:
        return None

    elapsed = t_ms - start
    repeat_idx = min(int(elapsed // duration), repeated - 1)
    frac = (elapsed - repeat_idx * duration) / duration
    if repeat_idx % 2 == 1:
        frac = 1.0 - frac
    return slider_c.at(frac * pixel_length)


def _write_data_yaml(out: Path):
    import yaml
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 7,
        "names": {
            0: "hitcircle",
            1: "slider_head",
            2: "slider_body",
            3: "slider_end",
            4: "spinner",
            5: "approach_circle",
            6: "slider_ball",
        },
    }
    with open(out / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)


_CLASS_NAMES = [
    "hitcircle", "slider_head", "slider_body", "slider_end",
    "spinner", "approach_circle", "slider_ball",
]


def _label_class_counts(label_path: Path) -> dict:
    """Return {class_id: instance_count} for a single YOLO label file."""
    counts: dict = {}
    try:
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                counts[cid] = counts.get(cid, 0) + 1
    except OSError:
        pass
    return counts


def _scan_class_instances(label_dir: Path):
    """Aggregate per-class instance counts across every label file in a dir."""
    totals: dict = {}
    for lbl in label_dir.glob("*.txt"):
        for cid, n in _label_class_counts(lbl).items():
            totals[cid] = totals.get(cid, 0) + n
    return totals


def _balance_train_split(out: Path, max_factor: int = 8):
    """Oversample rare-class images in the train split until classes balance.

    Detection labels are multi-class per image, so we cannot drop images
    without losing common-class instances. Instead we duplicate (oversample)
    images that contain under-represented classes. Each image is replicated by
    a factor driven by the rarest class it contains::

        factor(image) = min(max_factor,
                            ceil(max over classes c in image of target / count[c]))

    where ``target`` is the most common class's instance count. Duplicated
    files are written with an ``_balN`` suffix so they survive a re-run cleanly
    (existing ``_bal*`` copies are purged first).
    """
    img_dir = out / "images" / "train"
    lbl_dir = out / "labels" / "train"
    if not lbl_dir.exists():
        return

    # Purge previous balancing duplicates (idempotent re-runs)
    for p in img_dir.glob("*_bal*.jpg"):
        p.unlink()
    for p in lbl_dir.glob("*_bal*.txt"):
        p.unlink()

    totals = _scan_class_instances(lbl_dir)
    if not totals:
        return
    target = max(totals.values())

    print(f"\n[balance] pre-balance instances: "
          + ", ".join(f"{_CLASS_NAMES[c]}={totals.get(c, 0)}"
                      for c in range(len(_CLASS_NAMES))))

    import math
    import shutil

    added = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        if "_bal" in lbl.stem:
            continue
        counts = _label_class_counts(lbl)
        if not counts:
            continue
        # Replication driven by the rarest class present in this image
        factor = 1
        for cid in counts:
            c = totals.get(cid, 0)
            if c > 0:
                factor = max(factor, math.ceil(target / c))
        factor = min(factor, max_factor)
        if factor <= 1:
            continue

        img = img_dir / f"{lbl.stem}.jpg"
        if not img.exists():
            continue
        for k in range(1, factor):
            shutil.copyfile(img, img_dir / f"{lbl.stem}_bal{k}.jpg")
            shutil.copyfile(lbl, lbl_dir / f"{lbl.stem}_bal{k}.txt")
            added += 1

    print(f"[balance] added {added} oversampled copies (max_factor={max_factor})")


def _print_class_histogram(out: Path):
    """Print per-class instance counts for the train and val splits."""
    for split in ("train", "val"):
        lbl_dir = out / "labels" / split
        if not lbl_dir.exists():
            continue
        totals = _scan_class_instances(lbl_dir)
        total = sum(totals.values()) or 1
        print(f"\n[{split}] class instances ({total} total):")
        for c in range(len(_CLASS_NAMES)):
            n = totals.get(c, 0)
            print(f"    {c} {_CLASS_NAMES[c]:<16} {n:>8}  ({100*n/total:5.1f}%)")


if __name__ == "__main__":
    main()
