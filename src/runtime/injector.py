"""
Input injection for osu! using Win32 SendInput.

Provides precise cursor movement and key presses at 1000Hz.
All input goes through the standard Windows input pipeline
(no direct memory manipulation).
"""

from __future__ import annotations

import ctypes
import platform
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    import ctypes.wintypes as wt

    # ── DPI awareness ────────────────────────────────────────────────
    # Must be set BEFORE any GetSystemMetrics call so we get physical
    # pixel values, not DPI-scaled logical values.
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    # SendInput structures
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_ABSOLUTE = 0x8000
    MOUSEEVENTF_VIRTUALDESK = 0x4000

    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008

    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1

    # Virtual key codes for osu! keys
    VK_Z = 0x5A
    VK_X = 0x58

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", ctypes.c_ulong),
            ("wParamL", ctypes.c_ushort),
            ("wParamH", ctypes.c_ushort),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("union", _INPUT_UNION),
        ]

    _user32 = ctypes.windll.user32
    _SendInput = _user32.SendInput
    _SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int]
    _SendInput.restype = ctypes.c_uint

    # GetCursorPos for reading actual cursor position
    _GetCursorPos = _user32.GetCursorPos
    _GetCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
    _GetCursorPos.restype = ctypes.c_bool

    # ── Virtual desktop metrics (for multi-monitor + VIRTUALDESK) ────
    # SM_XVIRTUALSCREEN / SM_YVIRTUALSCREEN = origin (can be negative)
    # SM_CXVIRTUALSCREEN / SM_CYVIRTUALSCREEN = total size
    _SM_XVIRTUALSCREEN = 76
    _SM_YVIRTUALSCREEN = 77
    _SM_CXVIRTUALSCREEN = 78
    _SM_CYVIRTUALSCREEN = 79
    _virt_x = _user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
    _virt_y = _user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
    _virt_w = _user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
    _virt_h = _user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)


@dataclass
class InputEvent:
    """A single input event to be executed."""
    timestamp_ms: float
    # Mouse move
    move_x: Optional[int] = None    # screen pixels
    move_y: Optional[int] = None
    # Key press
    key_down: Optional[int] = None  # virtual key code
    key_up: Optional[int] = None


class InputInjector:
    """
    High-frequency input injection using Win32 SendInput.

    Runs a dedicated 1000Hz thread that processes queued input events
    at precise timestamps.
    """

    def __init__(self, polling_rate_hz: int = 1000):
        """
        Args:
            polling_rate_hz: Rate at which to process input events
        """
        if not _IS_WINDOWS:
            raise NotImplementedError(
                "InputInjector requires Windows. "
                "Use MockInjector for testing/observe mode."
            )

        self.interval_s = 1.0 / polling_rate_hz
        self._queue: Deque[InputEvent] = deque()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._key_states = {}  # track pressed keys

    def start(self):
        """Start the input processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop and release all held keys."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # Release any held keys
        for vk in list(self._key_states.keys()):
            if self._key_states[vk]:
                self._send_key_up(vk)
        self._key_states.clear()

    def move_to(self, screen_x: int, screen_y: int):
        """Queue an absolute cursor move."""
        self._queue.append(InputEvent(
            timestamp_ms=time.perf_counter() * 1000,
            move_x=screen_x,
            move_y=screen_y,
        ))

    def move_to_immediate(self, screen_x: int, screen_y: int):
        """Move cursor immediately (bypass queue)."""
        self._send_mouse_move(screen_x, screen_y)

    def key_down(self, vk: int = VK_Z):
        """Queue a key press."""
        self._queue.append(InputEvent(
            timestamp_ms=time.perf_counter() * 1000,
            key_down=vk,
        ))

    def key_up(self, vk: int = VK_Z):
        """Queue a key release."""
        self._queue.append(InputEvent(
            timestamp_ms=time.perf_counter() * 1000,
            key_up=vk,
        ))

    def tap(self, vk: int = VK_Z, hold_ms: float = 30.0):
        """Queue a key tap (down then up after hold_ms)."""
        now = time.perf_counter() * 1000
        self._queue.append(InputEvent(timestamp_ms=now, key_down=vk))
        self._queue.append(InputEvent(timestamp_ms=now + hold_ms, key_up=vk))

    def queue_trajectory(
        self,
        points: list,
        screen_transform: callable,
        base_time_ms: float,
    ):
        """
        Queue a full trajectory of mouse moves.

        Args:
            points: List of TrajectoryPoint
            screen_transform: Function (osu_x, osu_y) -> (screen_x, screen_y)
            base_time_ms: Absolute timestamp of trajectory start
        """
        for pt in points:
            sx, sy = screen_transform(pt.x, pt.y)
            self._queue.append(InputEvent(
                timestamp_ms=base_time_ms + pt.t,
                move_x=int(sx),
                move_y=int(sy),
            ))

    def _loop(self):
        """Process input events at high frequency."""
        while self._running:
            now_ms = time.perf_counter() * 1000

            # Process all events due
            while self._queue:
                event = self._queue[0]
                if event.timestamp_ms > now_ms:
                    break
                self._queue.popleft()
                self._execute(event)

            time.sleep(self.interval_s)

    def _execute(self, event: InputEvent):
        """Execute a single input event via SendInput."""
        if event.move_x is not None and event.move_y is not None:
            self._send_mouse_move(event.move_x, event.move_y)
        if event.key_down is not None:
            self._send_key_down(event.key_down)
        if event.key_up is not None:
            self._send_key_up(event.key_up)

    def _send_mouse_move(self, sx: int, sy: int):
        """Send absolute mouse move via SendInput.

        Uses MOUSEEVENTF_VIRTUALDESK so the cursor is positioned correctly
        regardless of which monitor osu! is on.  Coordinates are normalised
        to the virtual-desktop extent (not just the primary monitor).
        """
        # Map screen coords → 0-65535 over the virtual desktop
        abs_x = int((sx - _virt_x) * 65535 / max(1, _virt_w - 1))
        abs_y = int((sy - _virt_y) * 65535 / max(1, _virt_h - 1))
        abs_x = max(0, min(65535, abs_x))
        abs_y = max(0, min(65535, abs_y))

        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = abs_x
        inp.union.mi.dy = abs_y
        inp.union.mi.dwFlags = (
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        )
        _SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _send_key_down(self, vk: int):
        """Send key down event."""
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        inp.union.ki.wScan = 0
        inp.union.ki.dwFlags = 0
        _SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        self._key_states[vk] = True

    def _send_key_up(self, vk: int):
        """Send key up event."""
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        inp.union.ki.wScan = 0
        inp.union.ki.dwFlags = KEYEVENTF_KEYUP
        _SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        self._key_states[vk] = False

    def get_cursor_pos(self) -> Tuple[int, int]:
        """Read actual cursor position from Windows."""
        pt = wt.POINT()
        _GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y


class MockInjector:
    """Mock input injector for observe/testing mode (no actual input sent)."""

    def __init__(self, **kwargs):
        self.events: List[InputEvent] = []
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def move_to(self, x: int, y: int):
        self.events.append(InputEvent(
            timestamp_ms=time.perf_counter() * 1000, move_x=x, move_y=y
        ))

    def move_to_immediate(self, x: int, y: int):
        self.move_to(x, y)

    def key_down(self, vk: int = 0x5A):
        self.events.append(InputEvent(
            timestamp_ms=time.perf_counter() * 1000, key_down=vk
        ))

    def key_up(self, vk: int = 0x5A):
        self.events.append(InputEvent(
            timestamp_ms=time.perf_counter() * 1000, key_up=vk
        ))

    def tap(self, vk: int = 0x5A, hold_ms: float = 30.0):
        self.key_down(vk)
        self.key_up(vk)

    def queue_trajectory(self, points, screen_transform, base_time_ms):
        for pt in points:
            sx, sy = screen_transform(pt.x, pt.y)
            self.events.append(InputEvent(
                timestamp_ms=base_time_ms + pt.t,
                move_x=int(sx), move_y=int(sy),
            ))

    def get_cursor_pos(self) -> Tuple[int, int]:
        """Return last known cursor position (mock)."""
        for event in reversed(self.events):
            if event.move_x is not None and event.move_y is not None:
                return event.move_x, event.move_y
        return 0, 0
