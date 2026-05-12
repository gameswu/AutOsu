"""
Dynamic osu! window detection and coordinate mapping.

Uses Win32 API (via ctypes) to find the osu! window at runtime and compute
the playfield ↔ screen ↔ model-input coordinate transforms.

No resolution values are hard-coded; everything is derived from the
actual window geometry at call time.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import platform
from dataclasses import dataclass
from typing import Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────

# osu! internal playfield (always the same)
_PLAYFIELD_W = 512
_PLAYFIELD_H = 384
_BASE_W = 640
_BASE_H = 480

# ── Win32 helpers ────────────────────────────────────────────────────────

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    _user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    _shcore = None
    try:
        _shcore = ctypes.windll.shcore  # type: ignore[attr-defined]
    except OSError:
        pass

    # Make this process DPI-aware so GetWindowRect returns real pixels
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
    except Exception:
        pass


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def _is_valid_game_window(hwnd: int) -> bool:
    """Check if hwnd is a visible window with a reasonable client area."""
    if not _user32.IsWindowVisible(hwnd):
        return False
    rc = _RECT()
    _user32.GetClientRect(hwnd, ctypes.byref(rc))
    w = rc.right - rc.left
    h = rc.bottom - rc.top
    return w >= 100 and h >= 100


def _find_osu_hwnd() -> int:
    """Find the osu! game window handle.

    osu! creates multiple windows (main, helper, tray, etc.).
    We enumerate ALL windows with "osu!" in the title and pick the
    one that is visible and has the largest client area.
    """
    if not _IS_WINDOWS:
        return 0

    candidates = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _enum_cb(h, _lp):
        buf = ctypes.create_unicode_buffer(256)
        _user32.GetWindowTextW(h, buf, 256)
        title = buf.value
        if "osu!" in title:
            candidates.append(int(h))  # ensure plain int
        return True  # continue enumeration

    _user32.EnumWindows(_enum_cb, 0)

    if not candidates:
        return 0

    # Pick the largest visible window
    best_hwnd = 0
    best_area = 0
    for h in candidates:
        if not _user32.IsWindowVisible(h):
            continue
        rc = _RECT()
        _user32.GetClientRect(h, ctypes.byref(rc))
        w = rc.right - rc.left
        h_px = rc.bottom - rc.top
        area = w * h_px
        if area > best_area:
            best_area = area
            best_hwnd = h

    # Fallback: if no visible candidate, return first one found
    return best_hwnd if best_hwnd else candidates[0]


def _get_client_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """Return (client_x, client_y, client_w, client_h) in screen pixels."""
    if not _IS_WINDOWS:
        raise RuntimeError("Window detection is only available on Windows.")

    # Get client area size
    rc_client = _RECT()
    _user32.GetClientRect(hwnd, ctypes.byref(rc_client))
    client_w = rc_client.right - rc_client.left
    client_h = rc_client.bottom - rc_client.top

    # Convert client (0,0) to screen coordinates
    pt = wt.POINT(0, 0)
    _user32.ClientToScreen(hwnd, ctypes.byref(pt))

    return pt.x, pt.y, client_w, client_h


# ── Coordinate mapping ───────────────────────────────────────────────────

@dataclass
class PlayfieldMapping:
    """
    All values needed to convert between osu! pixels, screen pixels,
    and model-input pixels.

    Derived dynamically from the window's client rect.
    """

    # Window client area position and size (screen pixels)
    client_x: int
    client_y: int
    client_w: int
    client_h: int

    # Computed scale factor  (screen pixels per osu!-base pixel)
    scale: float

    # Playfield bounding box in screen-pixel coordinates
    playfield_x: int       # left edge (screen px)
    playfield_y: int       # top edge  (screen px)
    playfield_w: int       # width  (screen px)
    playfield_h: int       # height (screen px)

    # Model input dimensions (for the YOLO detector)
    model_input_w: int = 640
    model_input_h: int = 384

    # ── Transforms ───────────────────────────────────────────────────

    def osu_to_screen(self, osu_x: float, osu_y: float) -> Tuple[int, int]:
        """Convert osu!pixel → absolute screen pixel.

        Formula: screen = window_centre + (osu - playfield_centre) * scale
        with the 8-osu!px downward shift baked into playfield_y.
        """
        sx = int(osu_x * self.scale + self.playfield_x)
        sy = int(osu_y * self.scale + self.playfield_y)
        return sx, sy

    def screen_to_osu(self, sx: int, sy: int) -> Tuple[float, float]:
        """Convert absolute screen pixel → osu!pixel."""
        ox = (sx - self.playfield_x) / self.scale
        oy = (sy - self.playfield_y) / self.scale
        return ox, oy

    def osu_to_model(self, osu_x: float, osu_y: float) -> Tuple[float, float]:
        """Convert osu!pixel → model input pixel (float).

        Chain: osu! → screen → client-relative → model pixel.
        This correctly handles any client aspect ratio (4:3, 16:9, etc.).
        """
        sx, sy = self.osu_to_screen(osu_x, osu_y)
        mx = (sx - self.client_x) * self.model_input_w / self.client_w
        my = (sy - self.client_y) * self.model_input_h / self.client_h
        return mx, my

    def model_to_osu(self, mx: float, my: float) -> Tuple[float, float]:
        """Convert model input pixel → osu!pixel.

        Chain: model pixel → client-relative → screen → osu!.
        Inverse of osu_to_model().
        """
        sx = mx * self.client_w / self.model_input_w + self.client_x
        sy = my * self.client_h / self.model_input_h + self.client_y
        return self.screen_to_osu(sx, sy)

    def osu_radius_to_model(self, radius_osu: float) -> float:
        """Convert a radius in osu!pixels to model-input pixels.

        Uses vertical scale (preserves playfield proportion).
        """
        return radius_osu * self.scale * self.model_input_h / self.client_h

    def osu_radius_to_screen(self, radius_osu: float) -> float:
        """Convert a radius in osu!pixels to screen pixels."""
        return radius_osu * self.scale

    @property
    def capture_region(self) -> Tuple[int, int, int, int]:
        """
        Full client area (left, top, right, bottom) in screen pixels.

        We capture the entire window (not just the playfield) because
        the model is trained on full-window frames.
        """
        return (
            self.client_x,
            self.client_y,
            self.client_x + self.client_w,
            self.client_y + self.client_h,
        )


def get_playfield_mapping(
    model_input_w: int = 640,
    model_input_h: int = 384,
    hwnd: int | None = None,
) -> PlayfieldMapping:
    """
    Detect the osu! window and compute all coordinate mappings.

    Raises RuntimeError if osu! is not running or not on Windows.
    """
    if hwnd is None:
        hwnd = _find_osu_hwnd()
    if not hwnd:
        raise RuntimeError(
            "Could not find the osu! window. Is osu! running?"
        )

    cx, cy, cw, ch = _get_client_rect(hwnd)

    if cw < 100 or ch < 100:
        raise RuntimeError(
            f"osu! window client area is too small ({cw}x{ch}). "
            f"Is osu! minimised or still loading? "
            f"Make sure the osu! window is visible on screen."
        )

    # osu! maps its 640×480 base resolution to the full client area
    # while keeping 4:3 aspect ratio.
    # scale = client_height / 480
    scale = ch / _BASE_H

    # Playfield origin inside the client area:
    # horizontal centre  – half the playfield width
    # vertical centre    – half the playfield height + 8 * scale (osu! offset)
    pf_w = int(_PLAYFIELD_W * scale)
    pf_h = int(_PLAYFIELD_H * scale)
    pf_x = cx + (cw - pf_w) // 2
    pf_y = cy + (ch - pf_h) // 2 + int(8 * scale)

    return PlayfieldMapping(
        client_x=cx, client_y=cy, client_w=cw, client_h=ch,
        scale=scale,
        playfield_x=pf_x, playfield_y=pf_y,
        playfield_w=pf_w, playfield_h=pf_h,
        model_input_w=model_input_w,
        model_input_h=model_input_h,
    )


def make_offline_mapping(
    client_w: int = 1920,
    client_h: int = 1080,
    model_input_w: int = 640,
    model_input_h: int = 384,
) -> PlayfieldMapping:
    """
    Create a PlayfieldMapping without an actual osu! window.

    Useful for offline rendering / dataset generation where no window exists.
    The client area is treated as starting at (0, 0).
    """
    scale = client_h / _BASE_H
    pf_w = int(_PLAYFIELD_W * scale)
    pf_h = int(_PLAYFIELD_H * scale)
    pf_x = (client_w - pf_w) // 2
    pf_y = (client_h - pf_h) // 2 + int(8 * scale)

    return PlayfieldMapping(
        client_x=0, client_y=0, client_w=client_w, client_h=client_h,
        scale=scale,
        playfield_x=pf_x, playfield_y=pf_y,
        playfield_w=pf_w, playfield_h=pf_h,
        model_input_w=model_input_w,
        model_input_h=model_input_h,
    )
