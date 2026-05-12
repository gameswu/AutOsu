#!/usr/bin/env python3
"""
Prepare raw data: extract .osz files and verify replay matching.

This is a diagnostic / preparation step before running generate_dataset.py.
It extracts .osz archives, builds the beatmap MD5 index, and shows which
replays match which beatmaps.

Usage::

    # First time: extract .osz files and check matches
    python scripts/prepare_data.py --data raw_data

    # Just show match summary (skip extraction)
    python scripts/prepare_data.py --data raw_data --no-extract

    # Show detailed match info
    python scripts/prepare_data.py --data raw_data --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.xxx` works
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Prepare raw_data: extract .osz + verify matches")
    parser.add_argument("--data", "-d", default="raw_data",
                        help="Path to raw_data directory (default: raw_data)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip .osz extraction")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed match info")
    args = parser.parse_args()

    from src.data.replay_parser import (
        extract_all_osz,
        build_beatmap_index,
        scan_replays,
    )
    from src.data.osu_parser import OsuParser

    data_dir = Path(args.data)
    beatmaps_dir = data_dir / "beatmaps"
    replays_dir = data_dir / "replays"

    # Validate structure
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        print(f"\nCreate the following structure:")
        print(f"  {data_dir}/")
        print(f"  ├── beatmaps/   ← put .osz files here")
        print(f"  └── replays/    ← put .osr files here")
        sys.exit(1)

    if not beatmaps_dir.exists():
        beatmaps_dir.mkdir(parents=True)
        print(f"Created: {beatmaps_dir}")
    if not replays_dir.exists():
        replays_dir.mkdir(parents=True)
        print(f"Created: {replays_dir}")

    # Count raw files
    osz_files = list(beatmaps_dir.glob("*.osz"))
    osr_files = list(replays_dir.rglob("*.osr"))
    print(f"\nraw_data/")
    print(f"  beatmaps/  {len(osz_files)} .osz files")
    print(f"  replays/   {len(osr_files)} .osr files")

    if not osz_files and not list(beatmaps_dir.rglob("*.osu")):
        print("\nNo beatmaps found. Put .osz files in raw_data/beatmaps/")
        sys.exit(0)

    # 1. Extract .osz
    if not args.no_extract and osz_files:
        print(f"\nExtracting {len(osz_files)} .osz files...")
        n = extract_all_osz(beatmaps_dir)
        print(f"  Extracted: {n}")

    # 2. Build MD5 index
    print("\nBuilding beatmap index...")
    index = build_beatmap_index(beatmaps_dir)
    print(f"  {len(index)} .osu files indexed")

    # Show beatmap summary
    if args.verbose:
        # Group by folder
        folders = {}
        for md5, osu_path in index.items():
            folder = osu_path.parent.name
            folders.setdefault(folder, []).append((md5, osu_path))

        print(f"\n  Beatmap folders ({len(folders)}):")
        for folder, entries in sorted(folders.items()):
            print(f"    {folder}/")
            for md5, p in entries:
                try:
                    bm = OsuParser.parse(p)
                    mode_str = ["std", "taiko", "catch", "mania"][bm.metadata.mode]
                    print(f"      [{mode_str}] {p.name}  (md5={md5[:12]}...)")
                except Exception:
                    print(f"      [?] {p.name}  (md5={md5[:12]}...)")

    # 3. Scan replays
    if not osr_files:
        print("\nNo replays found. Put .osr files in raw_data/replays/")
        sys.exit(0)

    print(f"\nScanning {len(osr_files)} replay files...")
    replay_entries = scan_replays(replays_dir)
    print(f"  {len(replay_entries)} valid replays parsed")

    # 4. Match
    matched = {}
    unmatched_replays = []
    for osr_path, bh in replay_entries:
        osu_path = index.get(bh)
        if osu_path is not None:
            matched.setdefault(osu_path, []).append(osr_path)
        else:
            unmatched_replays.append((osr_path, bh))

    # Results
    total_matched_replays = sum(len(v) for v in matched.values())
    print(f"\n{'='*60}")
    print(f"Match results:")
    print(f"  Beatmaps with replays: {len(matched)}")
    print(f"  Total matched replays: {total_matched_replays}")
    print(f"  Unmatched replays:     {len(unmatched_replays)}")

    if args.verbose and matched:
        print(f"\n  Matched pairs:")
        for osu_path, osr_paths in sorted(matched.items()):
            try:
                bm = OsuParser.parse(osu_path)
                mode_str = ["std", "taiko", "catch", "mania"][bm.metadata.mode]
                title = f"{bm.metadata.artist} - {bm.metadata.title} [{bm.metadata.version}]"
            except Exception:
                mode_str = "?"
                title = osu_path.name
            print(f"    [{mode_str}] {title}")
            for osr in osr_paths:
                print(f"         <- {osr.name}")

    if unmatched_replays and args.verbose:
        print(f"\n  Unmatched replays (beatmap not in beatmaps/):")
        for osr_path, bh in unmatched_replays[:10]:
            print(f"    {osr_path.name}  (needs md5={bh[:16]}...)")
        if len(unmatched_replays) > 10:
            print(f"    ... and {len(unmatched_replays) - 10} more")

    # Filter std-only
    std_count = 0
    for osu_path in matched:
        try:
            bm = OsuParser.parse(osu_path)
            if bm.metadata.mode == 0:
                std_count += 1
        except Exception:
            pass

    print(f"\n  osu!std beatmaps ready: {std_count}")
    if std_count > 0:
        print(f"\n  Ready to generate dataset:")
        print(f"    python scripts/generate_dataset.py --data {data_dir} "
              f"--skin <skin_path> --output dataset")


if __name__ == "__main__":
    main()
