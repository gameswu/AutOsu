#!/usr/bin/env python3
"""
Visualization / debug tool for osr2mp4-based rendering.

Renders a beatmap with its matched replay using osr2mp4-core's rendering
pipeline, producing pixel-perfect osu! frames. Use this to verify rendering
output and inspect frame-by-frame.

Controls:
    Space        — pause / resume playback
    → (right)    — next frame (when paused)
    + / =        — speed up (2x)
    - / _        — slow down (0.5x)
    S            — save current frame as PNG
    Q / Esc      — quit

Usage::

    # Auto-match from raw_data (interactive selection):
    python scripts/preview.py --data raw_data --skin path/to/skin

    # Preview a specific .osz (auto-finds matching replays):
    python scripts/preview.py --osz raw_data/beatmaps/123456.osz --skin path/to/skin \\
        --replays raw_data/replays

    # Export as video file (no GUI):
    python scripts/preview.py --data raw_data --skin path/to/skin --export output.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Visual preview / debug tool for synthetic osu! rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input source ─────────────────────────────────────────────────
    src = parser.add_argument_group("input source (pick one)")
    src.add_argument("--data", default=None,
                     help="raw_data directory (beatmaps/ + replays/); auto-matches via MD5")
    src.add_argument("--osz", default=None,
                     help="Single .osz file; use with --replays to find matching .osr")
    src.add_argument("--replays", default=None,
                     help="Directory of .osr files to search for matches (used with --osz)")

    # ── Common options ───────────────────────────────────────────────
    parser.add_argument("--skin", required=True, help="Path to osu! skin directory")
    parser.add_argument("--width", type=int, default=640, help="Render width (default: 640)")
    parser.add_argument("--height", type=int, default=384, help="Render height (default: 384)")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--export", default=None, help="Export to video file (e.g. output.mp4)")
    parser.add_argument("--export-fps", type=int, default=None,
                        help="Video export FPS (default: same as --fps)")
    args = parser.parse_args()

    # Validate: exactly one input source
    if args.data is None and args.osz is None:
        parser.error("Provide one of --data or --osz")
    if args.data is not None and args.osz is not None:
        parser.error("--data and --osz are mutually exclusive")

    # ── Resolve .osu and .osr paths ──────────────────────────────────
    osu_path, osr_path = _resolve_input(args)

    # ── Render ───────────────────────────────────────────────────────
    _run_preview(osu_path, osr_path, args)


# =====================================================================
#  Input resolution
# =====================================================================

def _resolve_input(args) -> Tuple[Path, Optional[Path]]:
    """
    Resolve the actual .osu and .osr paths from the CLI arguments.

    Two modes:
      1. --data  : full raw_data directory, auto-match via MD5
      2. --osz   : single .osz file, extract + match
    """
    from src.data.replay_parser import (
        extract_osz,
        extract_all_osz,
        build_beatmap_index,
        scan_replays,
    )

    # ── Mode 1: --data (full raw_data directory) ─────────────────────
    if args.data is not None:
        data_dir = Path(args.data)
        beatmaps_dir = data_dir / "beatmaps"
        replays_dir = data_dir / "replays"

        if not beatmaps_dir.is_dir():
            print(f"ERROR: {beatmaps_dir} not found", file=sys.stderr)
            sys.exit(1)
        if not replays_dir.is_dir():
            print(f"ERROR: {replays_dir} not found", file=sys.stderr)
            sys.exit(1)

        # Extract .osz files
        n = extract_all_osz(beatmaps_dir)
        if n:
            print(f"  Extracted {n} .osz files")

        # Build index + scan replays
        print("  Building beatmap MD5 index...")
        index = build_beatmap_index(beatmaps_dir)
        print(f"  Indexed {len(index)} .osu files")

        print("  Scanning replays...")
        replay_entries = scan_replays(replays_dir)
        print(f"  Found {len(replay_entries)} replays")

        # Build match groups: osu_path -> [osr_paths]
        matches: Dict[Path, List[Path]] = {}
        for osr_p, bh in replay_entries:
            osu_p = index.get(bh)
            if osu_p is not None:
                matches.setdefault(osu_p, []).append(osr_p)

        if not matches:
            print("ERROR: No replay matched any beatmap.", file=sys.stderr)
            sys.exit(1)

        return _interactive_select(matches)

    # ── Mode 2: --osz (single .osz file) ─────────────────────────────
    osz_path = Path(args.osz)
    if not osz_path.exists():
        print(f"ERROR: .osz not found: {osz_path}", file=sys.stderr)
        sys.exit(1)

    print(f"  Extracting: {osz_path.name}")
    extracted = extract_osz(osz_path)

    # Build index for just this one folder
    index = build_beatmap_index(extracted)
    if not index:
        print("ERROR: No .osu files found in .osz", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(index)} difficulties")

    # Find replays to match against
    replay_entries: List[Tuple[Path, str]] = []
    if args.replays:
        replays_dir = Path(args.replays)
        if replays_dir.is_dir():
            replay_entries = scan_replays(replays_dir)
            print(f"  Scanned {len(replay_entries)} replays from {replays_dir}")
    else:
        # Try raw_data/replays/ as default
        default_replays = osz_path.parent.parent / "replays"
        if default_replays.is_dir():
            replay_entries = scan_replays(default_replays)
            print(f"  Scanned {len(replay_entries)} replays from {default_replays}")

    # Match
    matches = {}
    for osr_p, bh in replay_entries:
        osu_p = index.get(bh)
        if osu_p is not None:
            matches.setdefault(osu_p, []).append(osr_p)

    if matches:
        return _interactive_select(matches)

    # No replay matches — let user pick a difficulty without replay
    print("  No matching replays found. Preview without cursor.")
    osu_files = sorted(index.values(), key=lambda p: p.name)
    if len(osu_files) == 1:
        print(f"  Using: {osu_files[0].name}")
        return osu_files[0], None

    return _select_difficulty(osu_files), None


def _interactive_select(
    matches: Dict[Path, List[Path]],
) -> Tuple[Path, Optional[Path]]:
    """
    Let the user pick a beatmap difficulty and replay from the matches.

    If there's only one option at each step, it's auto-selected.
    """
    items = sorted(matches.items(), key=lambda kv: kv[0].name)

    # ── Select difficulty ────────────────────────────────────────────
    if len(items) == 1:
        osu_path, osr_list = items[0]
        print(f"\n  Auto-selected: {osu_path.name}")
    else:
        print(f"\n  Available beatmaps with matching replays ({len(items)}):\n")
        for i, (osu_p, osr_list) in enumerate(items):
            parent = osu_p.parent.name
            diff = osu_p.stem
            info = _quick_info(osu_p)
            print(f"    [{i}] {parent}/{diff}")
            if info:
                print(f"        {info}  ({len(osr_list)} replay{'s' if len(osr_list) != 1 else ''})")

        idx = _prompt_int(f"\n  Select beatmap [0-{len(items)-1}]: ", 0, len(items) - 1)
        osu_path, osr_list = items[idx]

    # ── Select replay ────────────────────────────────────────────────
    if len(osr_list) == 1:
        osr_path = osr_list[0]
        print(f"  Auto-selected replay: {osr_path.name}")
    else:
        print(f"\n  Matching replays ({len(osr_list)}):\n")
        for i, osr_p in enumerate(sorted(osr_list)):
            print(f"    [{i}] {osr_p.name}")

        idx = _prompt_int(f"\n  Select replay [0-{len(osr_list)-1}]: ", 0, len(osr_list) - 1)
        osr_path = sorted(osr_list)[idx]

    return osu_path, osr_path


def _select_difficulty(osu_files: List[Path]) -> Path:
    """Let user pick a difficulty when there are no replays."""
    print(f"\n  Available difficulties ({len(osu_files)}):\n")
    for i, p in enumerate(osu_files):
        info = _quick_info(p)
        print(f"    [{i}] {p.stem}")
        if info:
            print(f"        {info}")
    idx = _prompt_int(f"\n  Select difficulty [0-{len(osu_files)-1}]: ", 0, len(osu_files) - 1)
    return osu_files[idx]


def _quick_info(osu_path: Path) -> str:
    """Read AR/CS/OD from a .osu file without full parsing (fast)."""
    ar = cs = od = ""
    try:
        with open(osu_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ApproachRate:"):
                    ar = line.split(":")[1].strip()
                elif line.startswith("CircleSize:"):
                    cs = line.split(":")[1].strip()
                elif line.startswith("OverallDifficulty:"):
                    od = line.split(":")[1].strip()
                elif line.startswith("[HitObjects]"):
                    break
    except Exception:
        return ""
    if ar or cs or od:
        return f"AR={ar} CS={cs} OD={od}"
    return ""


def _prompt_int(prompt: str, lo: int, hi: int) -> int:
    """Prompt user for an integer in [lo, hi]."""
    while True:
        try:
            val = int(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  Please enter a number between {lo} and {hi}")
        except (ValueError, EOFError):
            print(f"  Please enter a number between {lo} and {hi}")


# =====================================================================
#  Preview / playback
# =====================================================================

def _run_preview(osu_path: Path, osr_path: Optional[Path], args):
    """Set up osr2mp4 renderer and run interactive or export playback."""
    from src.data.renderer import Osr2mp4Renderer

    if osr_path is None:
        print("ERROR: osr2mp4 renderer requires a replay file.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Beatmap: {osu_path.name}")
    print(f"  Replay:  {osr_path.name}")
    print(f"  Skin:    {args.skin}")
    print(f"  Resolution: {args.width}x{args.height} @ {args.fps}fps")

    print("  Building renderer (parsing, checking, loading skin)...")
    renderer = Osr2mp4Renderer(
        osu_path=str(osu_path),
        osr_path=str(osr_path),
        skin_path=str(args.skin),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(f"  Ready. ~{renderer.total_frames_estimate} frames")

    # ── Export mode ──────────────────────────────────────────────────
    if args.export:
        _run_export(args, renderer)
        return

    # ── Interactive mode ─────────────────────────────────────────────
    _run_interactive(args, renderer, osu_path)


def _run_export(args, renderer):
    """Export all frames to a video file."""
    export_path = Path(args.export)
    export_fps = args.export_fps or args.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(export_path), fourcc, export_fps,
                             (args.width, args.height))
    if not writer.isOpened():
        print(f"ERROR: Cannot open video writer: {export_path}")
        sys.exit(1)

    print(f"\n  Exporting to: {export_path} ({export_fps} fps)")
    frame_idx = 0
    total = renderer.total_frames_estimate
    while renderer.has_frames():
        frame, t_ms = renderer.render_frame()
        _draw_hud(frame, int(t_ms), total * 1000 // max(1, args.fps),
                  frame_idx, total, 1.0, False, True)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / max(1, total) * 100
            print(f"\r  Progress: {pct:.1f}% ({frame_idx}/{total})", end="")

    writer.release()
    print(f"\n  Done: {export_path} ({frame_idx} frames)")


def _run_interactive(args, renderer, osu_path):
    """Run interactive OpenCV playback (forward-only, with pause/step)."""
    print("\n  Controls: Space=pause  ->step  +/-=speed  S=save  Q=quit")
    print("  NOTE: Backward stepping not supported (osr2mp4 is forward-only).")
    print("  Starting playback...\n")

    window_name = f"AutOsu Preview - {osu_path.stem}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    paused = False
    speed = 1.0
    frame_idx = 0
    total = renderer.total_frames_estimate
    interval_ms = 1000.0 / args.fps

    # Pre-render first frame
    if not renderer.has_frames():
        print("  No frames to render.")
        return
    current_frame, current_time = renderer.render_frame()
    frame_idx = 1

    while True:
        # Draw HUD
        display = current_frame.copy()
        _draw_hud(display, int(current_time), total * 1000 // max(1, args.fps),
                  frame_idx, total, speed, False, True)
        cv2.imshow(window_name, display)

        # Wait / handle input
        wait_ms = 0 if paused else max(1, int(interval_ms / speed))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord(' '):
            paused = not paused
            print(f"  {'Paused' if paused else 'Playing'}")
        elif key == 83 or key == ord('d'):  # Right arrow or D
            if paused and renderer.has_frames():
                current_frame, current_time = renderer.render_frame()
                frame_idx += 1
        elif key == ord('+') or key == ord('='):
            speed = min(8.0, speed * 2.0)
            print(f"  Speed: {speed}x")
        elif key == ord('-') or key == ord('_'):
            speed = max(0.125, speed / 2.0)
            print(f"  Speed: {speed}x")
        elif key == ord('s'):
            save_path = f"preview_frame_{frame_idx:06d}_t{int(current_time)}.png"
            cv2.imwrite(save_path, current_frame)
            print(f"  Saved: {save_path}")
        else:
            if not paused and renderer.has_frames():
                current_frame, current_time = renderer.render_frame()
                frame_idx += 1
            elif not renderer.has_frames():
                if not paused:
                    print("  End of replay reached. Paused.")
                    paused = True

    cv2.destroyAllWindows()
    print("  Preview closed.")


# =====================================================================
#  Drawing helpers
# =====================================================================


def _draw_hud(
    frame: np.ndarray,
    t_ms: int,
    end_ms: int,
    frame_idx: int,
    total_frames: int,
    speed: float,
    labels_on: bool,
    trail_on: bool,
):
    """Draw HUD overlay (time, frame number, speed, status indicators)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Time bar at bottom
    progress = t_ms / max(1, end_ms)
    bar_y = h - 8
    bar_w = int(w * progress)
    cv2.rectangle(frame, (0, bar_y), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, bar_y), (bar_w, h), (0, 200, 255), -1)

    # Top-left: time
    minutes = t_ms // 60000
    seconds = (t_ms % 60000) // 1000
    millis = t_ms % 1000
    time_str = f"{minutes:02d}:{seconds:02d}.{millis:03d}"
    cv2.putText(frame, time_str, (8, 18), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Frame / total
    frame_str = f"F:{frame_idx}/{total_frames}"
    cv2.putText(frame, frame_str, (8, 36), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # Top-right: speed + toggles
    speed_str = f"{speed:.2g}x"
    (tw, _), _ = cv2.getTextSize(speed_str, font, 0.4, 1)
    cv2.putText(frame, speed_str, (w - tw - 8, 18), font, 0.4,
                (0, 255, 200), 1, cv2.LINE_AA)

    indicators = []
    if labels_on:
        indicators.append("L")
    if trail_on:
        indicators.append("T")
    if indicators:
        ind_str = " ".join(indicators)
        (tw2, _), _ = cv2.getTextSize(ind_str, font, 0.35, 1)
        cv2.putText(frame, ind_str, (w - tw2 - 8, 34), font, 0.35,
                    (150, 255, 150), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
