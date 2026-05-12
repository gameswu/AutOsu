"""
Parser for osu! beatmap (.osu) file format v14.

Extracts all information needed for synthetic frame rendering:
hit objects, timing points, difficulty settings, combo colours, metadata.

Reference: https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import List, Optional, Tuple


# ── Constants ────────────────────────────────────────────────────────────

class HitObjectType(IntFlag):
    CIRCLE  = 1
    SLIDER  = 2
    NEW_COMBO = 4
    SPINNER = 8
    MANIA_HOLD = 128


class CurveType(IntEnum):
    BEZIER  = 0
    CATMULL = 1
    LINEAR  = 2
    PERFECT = 3

    @classmethod
    def from_char(cls, c: str) -> "CurveType":
        return {"B": cls.BEZIER, "C": cls.CATMULL, "L": cls.LINEAR, "P": cls.PERFECT}[c]


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class TimingPoint:
    time: float           # ms
    beat_length: float    # ms per beat (uninherited) or SV multiplier encoding (inherited)
    meter: int
    sample_set: int
    sample_index: int
    volume: int
    uninherited: bool
    kiai: bool

    @property
    def bpm(self) -> float:
        if self.uninherited and self.beat_length > 0:
            return 60_000.0 / self.beat_length
        return 0.0

    @property
    def sv_multiplier(self) -> float:
        """For inherited points, return the slider velocity multiplier."""
        if not self.uninherited and self.beat_length < 0:
            return -100.0 / self.beat_length
        return 1.0


@dataclass
class HitCircle:
    x: int
    y: int
    time: int             # ms
    new_combo: bool
    combo_colour_skip: int
    hit_sound: int
    # assigned after parsing
    combo_colour_index: int = 0
    combo_number: int = 1


@dataclass
class SliderPoint:
    x: float
    y: float


@dataclass
class Slider:
    x: int
    y: int
    time: int
    new_combo: bool
    combo_colour_skip: int
    hit_sound: int
    curve_type: CurveType
    control_points: List[SliderPoint]   # includes starting x,y as first point
    slides: int                         # number of traversals
    pixel_length: float
    edge_sounds: List[int] = field(default_factory=list)
    # computed
    combo_colour_index: int = 0
    combo_number: int = 1
    duration_ms: float = 0.0            # total slider duration including repeats
    end_time: int = 0


@dataclass
class Spinner:
    x: int
    y: int
    time: int
    end_time: int
    new_combo: bool
    combo_colour_skip: int
    hit_sound: int
    combo_colour_index: int = 0
    combo_number: int = 1


@dataclass
class DifficultySettings:
    hp: float = 5.0
    cs: float = 4.0
    od: float = 8.0
    ar: float = 9.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0

    @property
    def circle_radius_osu(self) -> float:
        """Circle radius in osu!pixels.

        Matches osr2mp4-core: cs = (54.4 - 4.48 * CS).
        """
        return 54.4 - 4.48 * self.cs

    @property
    def preempt_ms(self) -> float:
        """Approach circle preempt time in ms (time before hit that object appears)."""
        if self.ar < 5:
            return 1200.0 + 120.0 * (5.0 - self.ar)
        elif self.ar == 5:
            return 1200.0
        else:
            return 1200.0 - 150.0 * (self.ar - 5.0)

    @property
    def fade_in_ms(self) -> float:
        """Fade-in duration: objects reach full opacity after 2/3 of preempt time.

        osu! wiki: "full opacity at 2/3 of preempt time before hit".
        Verified against stable client formulas:
          AR 0: preempt=1800, fade_in=1200  (2/3 * 1800)
          AR 5: preempt=1200, fade_in=800   (2/3 * 1200)
          AR10: preempt=450,  fade_in=300   (2/3 * 450)
        """
        return self.preempt_ms * 2.0 / 3.0


@dataclass
class BeatmapMetadata:
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version: str = ""
    beatmap_id: int = 0
    beatmap_set_id: int = -1
    audio_filename: str = ""
    mode: int = 0                   # 0 = osu!std


@dataclass
class Beatmap:
    metadata: BeatmapMetadata = field(default_factory=BeatmapMetadata)
    difficulty: DifficultySettings = field(default_factory=DifficultySettings)
    timing_points: List[TimingPoint] = field(default_factory=list)
    combo_colours: List[Tuple[int, int, int]] = field(default_factory=list)
    hit_objects: list = field(default_factory=list)   # mixed: HitCircle | Slider | Spinner
    # extracted from [Events]
    background_filename: str = ""

    @property
    def hit_circles(self) -> List[HitCircle]:
        return [o for o in self.hit_objects if isinstance(o, HitCircle)]

    @property
    def sliders(self) -> List[Slider]:
        return [o for o in self.hit_objects if isinstance(o, Slider)]

    @property
    def spinners(self) -> List[Spinner]:
        return [o for o in self.hit_objects if isinstance(o, Spinner)]

    @property
    def total_length_ms(self) -> int:
        if not self.hit_objects:
            return 0
        last = self.hit_objects[-1]
        if isinstance(last, Spinner):
            return last.end_time
        if isinstance(last, Slider):
            return last.end_time
        return last.time


# ── Parser ───────────────────────────────────────────────────────────────

class OsuParser:
    """
    Parse a single .osu file into a Beatmap object.

    Usage::

        beatmap = OsuParser.parse("path/to/file.osu")
    """

    @staticmethod
    def parse(path: str | Path) -> Beatmap:
        path = Path(path)
        text = path.read_text(encoding="utf-8-sig")   # BOM-safe
        return OsuParser._parse_text(text, parent_dir=path.parent)

    @staticmethod
    def _parse_text(text: str, parent_dir: Path | None = None) -> Beatmap:
        bm = Beatmap()
        sections: dict[str, list[str]] = {}
        current_section: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            # section header
            m = re.match(r"^\[(\w+)\]$", line)
            if m:
                current_section = m.group(1)
                sections.setdefault(current_section, [])
                continue
            if current_section is not None:
                sections[current_section].append(line)

        OsuParser._parse_general(sections.get("General", []), bm)
        OsuParser._parse_metadata(sections.get("Metadata", []), bm)
        OsuParser._parse_difficulty(sections.get("Difficulty", []), bm)
        OsuParser._parse_events(sections.get("Events", []), bm)
        OsuParser._parse_timing_points(sections.get("TimingPoints", []), bm)
        OsuParser._parse_colours(sections.get("Colours", []), bm)
        OsuParser._parse_hit_objects(sections.get("HitObjects", []), bm)

        # Assign combo colours and numbers
        OsuParser._assign_combos(bm)

        return bm

    # ── Section parsers ──────────────────────────────────────────────

    @staticmethod
    def _kv(line: str, with_space: bool = True) -> Tuple[str, str]:
        sep = ": " if with_space else ":"
        idx = line.find(sep)
        if idx == -1:
            # fallback: try both styles
            idx = line.find(":")
            if idx == -1:
                return line, ""
            return line[:idx].strip(), line[idx + 1:].strip()
        return line[:idx].strip(), line[idx + len(sep):].strip()

    @staticmethod
    def _parse_general(lines: list[str], bm: Beatmap):
        for line in lines:
            k, v = OsuParser._kv(line)
            if k == "AudioFilename":
                bm.metadata.audio_filename = v
            elif k == "Mode":
                bm.metadata.mode = int(v)

    @staticmethod
    def _parse_metadata(lines: list[str], bm: Beatmap):
        for line in lines:
            k, v = OsuParser._kv(line, with_space=False)
            m = bm.metadata
            if k == "Title":
                m.title = v
            elif k == "TitleUnicode":
                m.title_unicode = v
            elif k == "Artist":
                m.artist = v
            elif k == "ArtistUnicode":
                m.artist_unicode = v
            elif k == "Creator":
                m.creator = v
            elif k == "Version":
                m.version = v
            elif k == "BeatmapID":
                m.beatmap_id = int(v) if v else 0
            elif k == "BeatmapSetID":
                m.beatmap_set_id = int(v) if v else -1

    @staticmethod
    def _parse_difficulty(lines: list[str], bm: Beatmap):
        d = bm.difficulty
        for line in lines:
            k, v = OsuParser._kv(line)
            val = float(v)
            if k == "HPDrainRate":
                d.hp = val
            elif k == "CircleSize":
                d.cs = val
            elif k == "OverallDifficulty":
                d.od = val
            elif k == "ApproachRate":
                d.ar = val
            elif k == "SliderMultiplier":
                d.slider_multiplier = val
            elif k == "SliderTickRate":
                d.slider_tick_rate = val

    @staticmethod
    def _parse_events(lines: list[str], bm: Beatmap):
        for line in lines:
            parts = line.split(",")
            # Background: 0,0,"filename",xoffset,yoffset
            if len(parts) >= 3 and parts[0].strip() == "0" and parts[1].strip() == "0":
                bg = parts[2].strip().strip('"')
                bm.background_filename = bg

    @staticmethod
    def _parse_timing_points(lines: list[str], bm: Beatmap):
        for line in lines:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            time = float(parts[0])
            beat_length = float(parts[1])
            meter = int(parts[2]) if len(parts) > 2 else 4
            sample_set = int(parts[3]) if len(parts) > 3 else 0
            sample_index = int(parts[4]) if len(parts) > 4 else 0
            volume = int(parts[5]) if len(parts) > 5 else 100
            uninherited = bool(int(parts[6])) if len(parts) > 6 else True
            effects = int(parts[7]) if len(parts) > 7 else 0
            kiai = bool(effects & 1)

            bm.timing_points.append(TimingPoint(
                time=time,
                beat_length=beat_length,
                meter=meter,
                sample_set=sample_set,
                sample_index=sample_index,
                volume=volume,
                uninherited=uninherited,
                kiai=kiai,
            ))

    @staticmethod
    def _parse_colours(lines: list[str], bm: Beatmap):
        for line in lines:
            k, v = OsuParser._kv(line)
            if k.startswith("Combo"):
                rgb = tuple(int(c.strip()) for c in v.split(","))
                bm.combo_colours.append(rgb)  # type: ignore[arg-type]

    @staticmethod
    def _parse_hit_objects(lines: list[str], bm: Beatmap):
        for line in lines:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            x = int(parts[0])
            y = int(parts[1])
            time = int(parts[2])
            type_bits = int(parts[3])
            hit_sound = int(parts[4])

            new_combo = bool(type_bits & HitObjectType.NEW_COMBO)
            combo_skip = (type_bits >> 4) & 7

            if type_bits & HitObjectType.CIRCLE:
                bm.hit_objects.append(HitCircle(
                    x=x, y=y, time=time,
                    new_combo=new_combo,
                    combo_colour_skip=combo_skip,
                    hit_sound=hit_sound,
                ))

            elif type_bits & HitObjectType.SLIDER:
                OsuParser._parse_slider(parts, x, y, time, new_combo, combo_skip, hit_sound, bm)

            elif type_bits & HitObjectType.SPINNER:
                end_time = int(parts[5]) if len(parts) > 5 else time
                bm.hit_objects.append(Spinner(
                    x=256, y=192, time=time, end_time=end_time,
                    new_combo=new_combo,
                    combo_colour_skip=combo_skip,
                    hit_sound=hit_sound,
                ))

    @staticmethod
    def _parse_slider(
        parts: list[str],
        x: int, y: int, time: int,
        new_combo: bool, combo_skip: int, hit_sound: int,
        bm: Beatmap,
    ):
        # parts[5] = curveType|controlPoints
        curve_data = parts[5].split("|")
        curve_char = curve_data[0].strip()
        curve_type = CurveType.from_char(curve_char)

        control_points = [SliderPoint(float(x), float(y))]  # starting point
        for cp_str in curve_data[1:]:
            cxy = cp_str.split(":")
            if len(cxy) == 2:
                control_points.append(SliderPoint(float(cxy[0]), float(cxy[1])))

        slides = int(parts[6]) if len(parts) > 6 else 1
        pixel_length = float(parts[7]) if len(parts) > 7 else 0.0

        edge_sounds: list[int] = []
        if len(parts) > 8 and parts[8]:
            edge_sounds = [int(s) for s in parts[8].split("|") if s]

        slider = Slider(
            x=x, y=y, time=time,
            new_combo=new_combo,
            combo_colour_skip=combo_skip,
            hit_sound=hit_sound,
            curve_type=curve_type,
            control_points=control_points,
            slides=slides,
            pixel_length=pixel_length,
            edge_sounds=edge_sounds,
        )

        # Compute duration
        tp_beat, tp_sv = OsuParser._active_timing(bm.timing_points, time)
        if tp_beat > 0:
            sv = bm.difficulty.slider_multiplier * 100.0 * tp_sv
            one_slide_ms = pixel_length / sv * tp_beat if sv else 0
            slider.duration_ms = one_slide_ms * slides
            slider.end_time = time + int(slider.duration_ms)

        bm.hit_objects.append(slider)

    @staticmethod
    def _active_timing(
        timing_points: list[TimingPoint], time: float
    ) -> Tuple[float, float]:
        """
        Return (beat_length_ms, sv_multiplier) active at *time*.
        """
        beat_length = 500.0   # default 120 BPM
        sv = 1.0

        for tp in timing_points:
            if tp.time > time + 1:   # small tolerance
                break
            if tp.uninherited:
                beat_length = tp.beat_length
                sv = 1.0             # reset SV on new red line
            else:
                sv = tp.sv_multiplier

        return beat_length, sv

    @staticmethod
    def _assign_combos(bm: Beatmap):
        """Walk through hit objects and assign combo colour index + combo number."""
        if not bm.combo_colours:
            # fallback colours if beatmap has none
            bm.combo_colours = [(255, 192, 0), (0, 202, 0), (18, 124, 255), (242, 24, 57)]

        colour_idx = 0
        combo_num = 0

        for obj in bm.hit_objects:
            is_new = obj.new_combo or combo_num == 0
            # Spinners always start a new combo
            if isinstance(obj, Spinner):
                is_new = True

            if is_new:
                skip = obj.combo_colour_skip
                colour_idx = (colour_idx + 1 + skip) % len(bm.combo_colours)
                combo_num = 1
            else:
                combo_num += 1

            obj.combo_colour_index = colour_idx
            obj.combo_number = combo_num
