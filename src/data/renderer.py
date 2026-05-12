"""
osr2mp4-core rendering wrapper for AutOsu.

Uses the extracted osr2mp4 library (lib/osr2mp4/) to render beatmap
frames that exactly match osr2mp4-core output.

Usage:
    renderer = Osr2mp4Renderer(
        osu_path="path/to/beatmap.osu",
        osr_path="path/to/replay.osr",
        skin_path="path/to/skin/",
        width=640, height=384, fps=30,
    )
    while renderer.has_frames():
        frame_bgr, cur_time = renderer.render_frame()
        # frame_bgr is a numpy (H, W, 3) uint8 BGR array
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# ── Ensure lib/ is on sys.path so osr2mp4 + recordclass shim can import ──
_LIB_DIR = str(Path(__file__).resolve().parent.parent.parent / "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# ── Also ensure project root is on sys.path (for src.data.slider_path) ──
_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# ── Now import osr2mp4 modules ──
from osr2mp4.global_var import Settings, defaultsettings
from osr2mp4.Parser.osuparser import read_file
from osr2mp4.Parser.osrparser import setup_replay, add_useless_shits
from osr2mp4.Parser.skinparser import Skin
from osr2mp4.Utils.Resolution import get_screensize
from osr2mp4.CheckSystem.checkmain import checkmain
from osr2mp4.CheckSystem.Judgement import DiffCalculator
from osr2mp4.CheckSystem.mathhelper import getunstablerate
from osr2mp4.VideoProcess.AFrames import PreparedFrames, FrameObjects
from osr2mp4.VideoProcess.Draw import Drawer
from osr2mp4.VideoProcess.Setup import FrameInfo, CursorEvent
from osr2mp4.EEnum.EReplay import Replays
from osr2mp4.osrparse.replay import Replay
from osr2mp4.Utils.skip import skip
from osr2mp4.VideoProcess.smoothing import smoothcursor
from osr2mp4.VideoProcess.calc import (
    check_break, check_key, add_followpoints, add_hitobjects, nearer,
)


def _make_settings(
    osu_path: str,
    skin_path: str,
    width: int,
    height: int,
    fps: int,
) -> Settings:
    """Construct an osr2mp4 Settings object from our parameters."""
    settings = Settings()

    # Path to the lib/osr2mp4 package (for res/default/)
    settings.path = str(Path(__file__).resolve().parent.parent.parent / "lib" / "osr2mp4")
    if not settings.path.endswith(("/", "\\")):
        settings.path += "/"

    # Resolution
    pfs, pfw, pfh, scale, mr, md = get_screensize(width, height)
    settings.width = width
    settings.height = height
    settings.scale = scale
    settings.playfieldscale = pfs
    settings.playfieldwidth = pfw
    settings.playfieldheight = pfh
    settings.moveright = mr
    settings.movedown = md
    settings.fps = fps
    settings.timeframe = 1000  # NoMod

    # Paths
    settings.beatmap = str(Path(osu_path).parent)
    settings.osu = None
    settings.output = "output.avi"  # unused but needed
    settings.ffmpeg = "ffmpeg"  # unused

    # Temp dir
    settings.temp = str(Path(__file__).resolve().parent.parent.parent / "tmp_osr2mp4") + "/"
    os.makedirs(settings.temp, exist_ok=True)

    # Skin
    default_path = os.path.join(settings.path, "res/default/")
    skin = Skin(skin_path, default_path)
    settings.skin_path = skin_path
    settings.default_path = default_path
    settings.skin_ini = skin
    settings.default_skin_ini = skin

    # Gameplay settings (disable all UI chrome we don't need)
    gameplay = dict(defaultsettings)
    gameplay["In-game interface"] = True   # needed so background renders properly
    gameplay["Background dim"] = 80        # match typical osu! default (0-100)
    gameplay["Show scoreboard"] = False
    gameplay["Enable PP counter"] = False
    gameplay["Enable Strain Graph"] = False
    gameplay["Always show key overlay"] = False
    gameplay["Show mods icon"] = False
    gameplay["Show score meter"] = False
    gameplay["Global leaderboard"] = False
    settings.settings = gameplay

    return settings


class Osr2mp4Renderer:
    """
    Renders beatmap frames using osr2mp4-core's rendering pipeline.

    Produces pixel-perfect output matching osr2mp4's VideoProcess/Draw.py.
    """

    def __init__(
        self,
        osu_path: str,
        osr_path: str,
        skin_path: str,
        width: int = 640,
        height: int = 384,
        fps: int = 30,
    ):
        self.width = width
        self.height = height
        self.fps = fps

        # 1. Build Settings
        self.settings = _make_settings(osu_path, skin_path, width, height, fps)

        # 2. Parse replay using osr2mp4's parser
        self.replay_info = Replay.from_path(osr_path)

        # 3. Parse beatmap using osr2mp4's parser
        self.beatmap = read_file(
            osu_path,
            self.settings.playfieldscale,
            self.settings.skin_ini.colours,
            mods=self.replay_info.mod_combination,
            lazy=False,
        )

        # 4. Setup replay data (convert to list format, add padding)
        self.replay_data, self.start_time = setup_replay(
            osr_path, self.beatmap
        )
        self.replay_info.play_data = self.replay_data

        # 5. Run hit checking to get resultinfo (needed for visual state transitions)
        self.resultinfo = checkmain(self.beatmap, self.replay_info, self.settings)

        # 6. Compute start/end indices
        self.start_index = 0
        self.end_index = len(self.replay_data) - 3

        # 7. Create the image buffer (replaces shared memory)
        self._buffer = bytearray(width * height * 4)  # RGBA
        self._np_img = np.frombuffer(self._buffer, dtype=np.uint8).reshape(
            (height, width, 4)
        )
        self._pil_img = Image.frombuffer(
            "RGBA", (width, height), self._buffer, "raw", "RGBA", 0, 1
        )
        self._pil_img.readonly = False

        # 8. Create PreparedFrames (loads all skin textures)
        ur = getunstablerate(self.resultinfo)
        self.frames = PreparedFrames(
            self.settings,
            self.beatmap.diff,
            self.replay_info.mod_combination,
            ur=ur,
            bg=self.beatmap.bg,
            loadranking=False,
        )

        # 9. Create Drawer (using our simplified buffer)
        self._setup_drawer()

    def _setup_drawer(self):
        """Set up the drawer state machine (simplified from Drawer.__init__)."""
        replay_event = self.replay_info.play_data

        old_cursor_x = int(
            replay_event[0][Replays.CURSOR_X] * self.settings.playfieldscale
        ) + self.settings.moveright
        old_cursor_y = int(
            replay_event[0][Replays.CURSOR_Y] * self.settings.playfieldscale
        ) + self.settings.movedown

        diffcalculator = DiffCalculator(self.beatmap.diff)
        self.time_preempt = diffcalculator.ar()

        from osr2mp4.CheckSystem.Health import HealthProcessor
        healthproc = HealthProcessor(self.beatmap, self.beatmap.health_processor.drain_rate)

        map_time = (self.beatmap.start_time, self.beatmap.end_time)
        light_replay_info = Replay()
        light_replay_info.set(self.replay_info.get())

        self.component = FrameObjects(
            self.frames, self.settings, self.beatmap.diff,
            light_replay_info, self.beatmap.meta, self.beatmap.hash, map_time,
        )
        self.component.scorebar.set_healthproc(healthproc)
        self.component.cursor_trail.set_cursor(
            old_cursor_x, old_cursor_y, replay_event[0][Replays.TIMES]
        )
        self.component.flashlight.set_pos(old_cursor_x, old_cursor_y)

        self.preempt_followpoint = 800

        from osr2mp4.InfoProcessor import Updater
        self.updater = Updater(
            self.resultinfo, self.component, self.settings,
            self.replay_info.mod_combination, self.beatmap.path,
        )

        # Skip to start
        to_time = replay_event[self.start_index][Replays.TIMES]
        self.frame_info = FrameInfo(
            *skip(
                to_time, self.resultinfo, replay_event,
                self.beatmap, self.time_preempt, self.component,
            )
        )

        self.cursor_event = CursorEvent(
            replay_event[self.frame_info.osr_index],
            old_cursor_x, old_cursor_y,
        )
        self.updater.info_index = self.frame_info.info_index

        self.key_queue = []
        self._started = False

    def has_frames(self) -> bool:
        """Check if there are more frames to render."""
        return self.frame_info.osr_index < self.end_index

    def render_frame(self) -> Tuple[np.ndarray, float]:
        """
        Advance one frame and return (bgr_image, current_time_ms).

        Returns
        -------
        bgr_image : np.ndarray, shape (H, W, 3), dtype uint8
        current_time_ms : float
        """
        replay_event = self.replay_info.play_data
        import copy

        # Activate buffer
        if not self._started:
            self._started = True

        cur_time = self.frame_info.cur_time

        # Process break periods
        in_break = check_break(
            self.beatmap, self.component, self.frame_info,
            self.updater, self.settings,
        )

        # Process key events
        if self.key_queue:
            cur_key = self.key_queue.pop(0)
            check_key(self.component, cur_key, self.frame_info.cur_time, in_break)

        # Add new followpoints and hit objects
        add_followpoints(
            self.beatmap, self.component, self.frame_info,
            self.preempt_followpoint,
        )
        add_hitobjects(
            self.beatmap, self.component, self.frame_info,
            self.time_preempt, self.settings,
        )

        # Update visual state (hit results, fadeouts, slider follows)
        self.updater.update(self.frame_info.cur_time)

        # Cursor interpolation
        cx, cy = smoothcursor(
            replay_event, self.frame_info.osr_index, self.frame_info.cur_time,
        )
        cursor_x = int(cx * self.settings.playfieldscale) + self.settings.moveright
        cursor_y = int(cy * self.settings.playfieldscale) + self.settings.movedown

        # ── DRAW (matches Draw.py:106-143 order exactly) ──
        img = self._pil_img

        # Background (also clears frame)
        self.component.background.add_to_frame(
            img, self._np_img, self.frame_info.cur_time, in_break,
        )

        # Hit results layer 1 (under hit objects)
        self.component.hitresult.add_to_frame(img)

        # Follow points
        self.component.followpoints.add_to_frame(img, self.frame_info.cur_time)

        # Hit objects (circles + sliders + spinners)
        self.component.hitobjmanager.add_to_frame(img, self.frame_info.cur_time)

        # Hit results layer 2 (over hit objects)
        self.component.hitresult.add_to_frame(img)

        # Cursor trail + cursor + cursor middle
        self.component.cursor_trail.add_to_frame(
            img, cursor_x, cursor_y, self.frame_info.cur_time,
        )
        self.component.cursor.add_to_frame(img, cursor_x, cursor_y)
        self.component.cursormiddle.add_to_frame(img, cursor_x, cursor_y)

        # Advance time
        self.frame_info.cur_time += self.settings.timeframe / self.settings.fps

        # Advance replay index
        tt, keys = nearer(
            self.frame_info.cur_time, self.replay_info, self.frame_info.osr_index,
        )
        if self.key_queue:
            while keys and keys[0] == self.key_queue[-1]:
                keys = keys[1:]
        self.key_queue.extend(keys)
        self.frame_info.osr_index += tt
        self.cursor_event.event = copy.copy(
            replay_event[self.frame_info.osr_index]
        )

        # Convert RGBA PIL image to BGR numpy array
        rgba = np.array(img)
        bgr = rgba[:, :, :3][:, :, ::-1].copy()  # RGBA -> BGR

        return bgr, cur_time

    def get_visible_objects(self):
        """Return the list of beatmap hitobjects currently being rendered.

        Reads directly from the HitObjectManager's internal state so the
        result is perfectly synchronised with the rendered frame.  Must be
        called **after** ``render_frame()`` for the same frame.

        Each returned element is ``(hitobject_dict, is_fadeout)`` where
        *is_fadeout* is True when the circle is in its post-hit fade-out
        animation (already hit/missed and disappearing).

        Sliders have both a circle key (``{id}c``) and a slider key
        (``{id}s``) in the manager; we deduplicate by object id.
        """
        mgr = self.component.hitobjmanager
        hitobjects = self.beatmap.hitobjects
        visible = []
        seen_ids: set = set()
        for key in mgr.objtime:
            # key format: "{id}c" / "{id}s" / "{id}o"
            obj_id = int(key[:-1])
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            if obj_id < 0 or obj_id >= len(hitobjects):
                continue
            obj = hitobjects[obj_id]

            # Check if the circle part is in fadeout (already hit/missed)
            is_fadeout = False
            circle_key = str(obj_id) + "c"
            if circle_key in mgr.circle_manager.circles:
                is_fadeout = bool(mgr.circle_manager.circles[circle_key].is_fadeout)

            visible.append((obj, is_fadeout))
        return visible

    def render_all(self):
        """Generator yielding all frames as (bgr_image, time_ms) tuples."""
        while self.has_frames():
            yield self.render_frame()

    @property
    def total_frames_estimate(self) -> int:
        """Rough estimate of total frames."""
        if not self.replay_info.play_data:
            return 0
        total_time = (
            self.replay_info.play_data[self.end_index][Replays.TIMES]
            - self.replay_info.play_data[self.start_index][Replays.TIMES]
        )
        return max(1, int(total_time * self.fps / self.settings.timeframe))


# ── Convenience: PlayfieldTransform for coordinate math ──


class PlayfieldTransform:
    """Maps osu! pixel coordinates to render-canvas pixel coordinates.

    Uses osr2mp4-core's Resolution.py formula exactly.
    """

    def __init__(self, render_w: int, render_h: int):
        pfs, pfw, pfh, scale, mr, md = get_screensize(render_w, render_h)
        self.render_w = render_w
        self.render_h = render_h
        self.playfieldscale = pfs
        self.scale = scale
        self.moveright = mr
        self.movedown = md

    def osu_to_render(self, osu_x: float, osu_y: float) -> Tuple[int, int]:
        """Convert osu!pixel (512x384) to render pixel coords."""
        rx = int(osu_x * self.playfieldscale) + self.moveright
        ry = int(osu_y * self.playfieldscale) + self.movedown
        return rx, ry
