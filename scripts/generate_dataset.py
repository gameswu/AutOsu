#!/usr/bin/env python3
"""
Generate training data from raw_data/ (beatmaps + replays).

Produces:
  1. YOLO detection dataset (images + labels) -- with cursor rendered
  2. Action model sequences (state_vector, action) pairs as .npz

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
    (out / "sequences").mkdir(parents=True, exist_ok=True)

    # Coordinate transform for YOLO labels
    tf = PlayfieldTransform(args.width, args.height)

    random.seed(args.seed)
    total_frames = 0
    total_sequences = 0

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

            # Action sequence buffers
            sequence_states = []
            sequence_actions = []
            prev_cx, prev_cy = 256.0, 192.0
            prev_t_ms = 0.0

            from src.action.state import GameStateVector, ActionVector

            for fd in frame_iter:
                frame, t_ms, frame_idx = fd["frame"], fd["t_ms"], fd["frame_idx"]
                visible = fd["visible"]
                time_preempt, radius_osu = fd["time_preempt"], fd["radius_osu"]
                cursor_x, cursor_y = fd["cursor_x"], fd["cursor_y"]
                key_z, key_x = fd["key_z"], fd["key_x"]

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

                # Build state/action for action model
                dt_ms = t_ms - prev_t_ms
                objects = _build_object_states(visible, t_ms, time_preempt)
                vx = (cursor_x - prev_cx) / max(1, dt_ms) if dt_ms > 0 else 0
                vy = (cursor_y - prev_cy) / max(1, dt_ms) if dt_ms > 0 else 0

                state = GameStateVector(
                    objects=objects,
                    cursor_x=prev_cx, cursor_y=prev_cy,
                    cursor_vx=vx, cursor_vy=vy,
                    time_delta_ms=dt_ms,
                )
                action = ActionVector(
                    dx=cursor_x - prev_cx,
                    dy=cursor_y - prev_cy,
                    key_z=key_z,
                    key_x=key_x,
                )

                sequence_states.append(state.to_numpy())
                sequence_actions.append(action.to_numpy())

                prev_cx, prev_cy = cursor_x, cursor_y
                prev_t_ms = t_ms

                if frame_idx % 500 == 0:
                    print(f"    frame {frame_idx}, t={t_ms:.0f}ms, "
                          f"imgs={total_frames}")

            # Save sequence
            if sequence_states:
                seq_name = f"seq_{pair_idx:04d}_{osr_paths.index(osr_path):02d}.npz"
                np.savez_compressed(
                    str(out / "sequences" / seq_name),
                    states=np.array(sequence_states, dtype=np.float32),
                    actions=np.array(sequence_actions, dtype=np.float32),
                )
                total_sequences += 1

    # Write YOLO data.yaml
    _write_data_yaml(out)

    print(f"\n{'=' * 60}")
    print(f"Dataset generation complete:")
    print(f"  Frames: {total_frames}")
    print(f"  Sequences: {total_sequences}")
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
    """Generate YOLO label lines from visible osr2mp4 hit objects."""
    lines = []
    for obj in visible:
        kind = _obj_kind(obj)
        ox, oy = obj["x"], obj["y"]

        # Convert osu!pixel to render pixel, then normalize
        rx, ry = tf.osu_to_render(ox, oy)
        cx_n = rx / w
        cy_n = ry / h
        bw_n = (radius_osu * 2 * tf.playfieldscale) / w
        bh_n = (radius_osu * 2 * tf.playfieldscale) / h

        if kind == "circle":
            lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

        elif kind == "slider":
            # Slider head
            lines.append(f"1 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

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

        elif kind == "spinner":
            # Spinner centered at playfield center
            scx, scy = tf.osu_to_render(256, 192)
            lines.append(f"4 {scx / w:.6f} {scy / h:.6f} 0.600000 0.800000")

    return list(dict.fromkeys(lines))


def _approach_ratio(parts, t_ms, visible, time_preempt):
    """Compute approach_ratio for a YOLO label by matching to visible objects."""
    cls_id = int(parts[0])
    for obj in visible:
        kind = _obj_kind(obj)
        if cls_id == 0 and kind == "circle":
            dt = t_ms - (obj["time"] - time_preempt)
            return min(1.0, max(0.0, dt / time_preempt)) if time_preempt > 0 else 1.0
        elif cls_id == 1 and kind == "slider":
            dt = t_ms - (obj["time"] - time_preempt)
            return min(1.0, max(0.0, dt / time_preempt)) if time_preempt > 0 else 1.0
    return 0.5


def _build_object_states(visible, t_ms, time_preempt):
    """Convert visible osr2mp4 hit objects to ObjectState list."""
    from src.action.state import ObjectState

    objects = []
    for obj in visible:
        dt = t_ms - (obj["time"] - time_preempt)
        ratio = min(1.0, max(0.0, dt / time_preempt)) if time_preempt > 0 else 1.0

        kind = _obj_kind(obj)
        if kind == "circle":
            objects.append(ObjectState(class_id=0, x=obj["x"], y=obj["y"],
                                       approach_ratio=ratio))
        elif kind == "slider":
            objects.append(ObjectState(class_id=1, x=obj["x"], y=obj["y"],
                                       approach_ratio=ratio))
        elif kind == "spinner":
            objects.append(ObjectState(class_id=4, x=256, y=192,
                                       approach_ratio=ratio))

    return objects


def _write_data_yaml(out: Path):
    import yaml
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 5,
        "names": {
            0: "hitcircle",
            1: "slider_head",
            2: "slider_body",
            3: "slider_end",
            4: "spinner",
        },
    }
    with open(out / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)


if __name__ == "__main__":
    main()
