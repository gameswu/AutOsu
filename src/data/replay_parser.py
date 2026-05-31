"""
Replay parser and frame-level data extractor.

Parses .osr replay files (via osrparse) and aligns cursor/key data with
beatmap timestamps to produce absolute-time cursor/key frames. Used offline by
dataset generation (cursor rendering) and by `scripts/analyze_motion.py`
(human-motion statistics). Not used at runtime — the live player is vision-only.

Data organisation::

    raw_data/
    ├── beatmaps/          ← .osz files (or extracted folders)
    │   ├── 123456 Artist - Title.osz
    │   ├── 123456 Artist - Title/    ← auto-extracted
    │   │   ├── Artist - Title (Mapper) [Easy].osu
    │   │   ├── Artist - Title (Mapper) [Hard].osu
    │   │   ├── bg.jpg
    │   │   └── audio.mp3
    │   └── ...
    └── replays/           ← .osr files (flat or nested)
        ├── replay1.osr
        ├── replay2.osr
        └── ...

Matching is fully automatic via MD5: each .osr contains the beatmap's
MD5 hash, which is compared against the MD5 of each .osu file.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import osrparse
except ImportError:
    osrparse = None


@dataclass
class ReplayFrame:
    """A single frame of replay data, aligned to absolute time."""
    time_ms: int          # absolute beatmap time
    x: float              # cursor osu!x (0-512)
    y: float              # cursor osu!y (0-384)
    key_z: bool           # key1 / Z pressed
    key_x: bool           # key2 / X pressed
    # Derived from consecutive frames
    dx: float = 0.0       # cursor delta x since last frame
    dy: float = 0.0       # cursor delta y since last frame
    dt_ms: float = 0.0    # time delta since last frame


@dataclass
class ParsedReplay:
    """A fully parsed replay aligned to beatmap time."""
    player_name: str
    mods: int
    frames: List[ReplayFrame]
    # Metadata
    beatmap_hash: str = ""
    total_score: int = 0
    max_combo: int = 0

    @property
    def duration_ms(self) -> int:
        if not self.frames:
            return 0
        return self.frames[-1].time_ms - self.frames[0].time_ms

    def frames_in_range(self, start_ms: int, end_ms: int) -> List[ReplayFrame]:
        """Get frames within a time range (inclusive)."""
        return [f for f in self.frames if start_ms <= f.time_ms <= end_ms]


def parse_replay(osr_path: str | Path) -> ParsedReplay:
    """
    Parse a .osr replay file into frame-level data.

    Requires osrparse to be installed.
    """
    if osrparse is None:
        raise ImportError("osrparse is required: pip install osrparse")

    osr_path = Path(osr_path)
    replay = osrparse.Replay.from_path(str(osr_path))

    # Extract metadata
    player_name = replay.username or "unknown"
    mods = int(replay.mods) if replay.mods else 0
    beatmap_hash = replay.beatmap_hash or ""
    total_score = replay.score or 0
    max_combo = replay.max_combo or 0

    # Convert replay events to absolute-time frames
    frames: List[ReplayFrame] = []
    abs_time_ms = 0
    prev_x, prev_y = 256.0, 192.0

    for event in replay.replay_data:
        abs_time_ms += event.time_delta

        # Skip seed frames (time_delta = -12345)
        if event.time_delta < 0:
            continue

        x = float(event.x)
        y = float(event.y)

        # Key state decoding (osr format):
        # bit 0 = M1 (left click), bit 1 = M2 (right click)
        # bit 2 = K1 (Z), bit 3 = K2 (X)
        # In practice, key1=Z is bit 2 or bit 0, key2=X is bit 3 or bit 1
        keys = int(event.keys) if event.keys else 0
        key_z = bool(keys & (1 | 4))    # M1 or K1
        key_x = bool(keys & (2 | 8))    # M2 or K2

        dx = x - prev_x
        dy = y - prev_y
        dt_ms = float(event.time_delta)

        frames.append(ReplayFrame(
            time_ms=int(abs_time_ms),
            x=x, y=y,
            key_z=key_z, key_x=key_x,
            dx=dx, dy=dy, dt_ms=dt_ms,
        ))

        prev_x, prev_y = x, y

    return ParsedReplay(
        player_name=player_name,
        mods=mods,
        frames=frames,
        beatmap_hash=beatmap_hash,
        total_score=total_score,
        max_combo=max_combo,
    )


# ── .osz extraction ──────────────────────────────────────────────────────

def extract_osz(osz_path: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Extract a .osz file (zip) into a folder alongside it.

    Returns the path to the extracted folder.  Skips if already extracted.
    """
    osz_path = Path(osz_path)
    if output_dir is None:
        output_dir = osz_path.parent / osz_path.stem

    if output_dir.exists() and any(output_dir.glob("*.osu")):
        return output_dir  # already extracted

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(osz_path, "r") as zf:
        zf.extractall(output_dir)

    return output_dir


def extract_all_osz(beatmaps_dir: Path) -> int:
    """Extract all .osz files in a directory. Returns count extracted."""
    count = 0
    for osz in beatmaps_dir.glob("*.osz"):
        try:
            extract_osz(osz)
            count += 1
        except zipfile.BadZipFile:
            print(f"  WARNING: Bad zip: {osz.name}")
    return count


# ── MD5 index ────────────────────────────────────────────────────────────

def _md5_file(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_beatmap_index(beatmaps_dir: Path) -> Dict[str, Path]:
    """
    Scan *beatmaps_dir* recursively for .osu files and build
    a {md5_hash: osu_path} index.
    """
    index: Dict[str, Path] = {}
    for osu_path in beatmaps_dir.rglob("*.osu"):
        md5 = _md5_file(osu_path)
        index[md5] = osu_path
    return index


def scan_replays(replays_dir: Path) -> List[Tuple[Path, str]]:
    """
    Scan *replays_dir* for .osr files and extract their beatmap hashes.

    Returns list of (osr_path, beatmap_hash).
    Skips unparseable files silently.
    """
    results: List[Tuple[Path, str]] = []
    for osr_path in replays_dir.rglob("*.osr"):
        try:
            replay = osrparse.Replay.from_path(str(osr_path))
            bh = replay.beatmap_hash or ""
            if bh:
                results.append((osr_path, bh))
        except Exception:
            continue
    return results


# ── Main matching function ───────────────────────────────────────────────

def find_replay_pairs(
    data_dir: str | Path,
) -> List[Tuple[Path, List[Path]]]:
    """
    Scan raw_data/ for beatmap/replay pairs using MD5 matching.

    Expected structure::

        data_dir/
        ├── beatmaps/          ← .osz files and/or extracted folders
        │   ├── 123.osz
        │   ├── 123/           ← auto-extracted
        │   │   ├── map [Easy].osu
        │   │   ├── map [Hard].osu
        │   │   └── bg.jpg
        │   └── ...
        └── replays/           ← .osr files
            ├── replay1.osr
            └── ...

    Returns list of (osu_path, [osr_paths]) tuples.
    """
    data_dir = Path(data_dir)
    beatmaps_dir = data_dir / "beatmaps"
    replays_dir = data_dir / "replays"

    if not beatmaps_dir.is_dir():
        raise FileNotFoundError(f"beatmaps directory not found: {beatmaps_dir}")
    if not replays_dir.is_dir():
        raise FileNotFoundError(f"replays directory not found: {replays_dir}")

    if osrparse is None:
        raise ImportError("osrparse is required for replay matching")

    # 1. Extract any .osz files
    n_extracted = extract_all_osz(beatmaps_dir)
    if n_extracted:
        print(f"  Extracted {n_extracted} .osz files")

    # 2. Build {md5: osu_path} index
    print("  Building beatmap MD5 index...")
    index = build_beatmap_index(beatmaps_dir)
    print(f"  Indexed {len(index)} .osu files")

    # 3. Scan replays
    print("  Scanning replays...")
    replay_entries = scan_replays(replays_dir)
    print(f"  Found {len(replay_entries)} valid .osr files")

    # 4. Match by MD5
    matches: Dict[Path, List[Path]] = {}
    unmatched = 0
    for osr_path, bh in replay_entries:
        osu_path = index.get(bh)
        if osu_path is not None:
            matches.setdefault(osu_path, []).append(osr_path)
        else:
            unmatched += 1

    if unmatched:
        print(f"  WARNING: {unmatched} replays had no matching beatmap")

    pairs = [(osu, sorted(osrs)) for osu, osrs in sorted(matches.items())]
    print(f"  Matched: {len(pairs)} beatmaps with {sum(len(o) for _, o in pairs)} replays")
    return pairs
