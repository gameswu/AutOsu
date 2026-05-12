"""
Screen capture using DXcam (DXGI Desktop Duplication API).

Provides sub-5ms frame capture from the osu! window. Falls back to
mss (GDI) on systems where DXcam is unavailable.

This module requires Windows.
"""

from __future__ import annotations

import platform
import time
from typing import Optional, Tuple

import cv2
import numpy as np

_IS_WINDOWS = platform.system() == "Windows"


class ScreenCapture:
    """
    Capture frames from the osu! window region.

    Uses DXcam for fastest capture (DXGI Desktop Duplication).
    Falls back to mss if DXcam is not available.
    """

    def __init__(
        self,
        region: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (640, 384),
        use_dxcam: bool = True,
    ):
        """
        Args:
            region: (left, top, right, bottom) in screen pixels
            target_size: (width, height) to resize captured frames to
            use_dxcam: prefer DXcam over mss if available
        """
        self.region = region
        self.target_w, self.target_h = target_size
        self._camera = None
        self._mss = None
        self._backend = "none"

        if not _IS_WINDOWS:
            raise NotImplementedError(
                "ScreenCapture requires Windows (DXcam/mss)."
            )

        if use_dxcam:
            try:
                import dxcam
                self._camera = dxcam.create(output_color="BGR")
                self._backend = "dxcam"
            except (ImportError, Exception):
                pass

        if self._camera is None:
            try:
                import mss
                self._mss = mss.mss()
                self._backend = "mss"
            except ImportError:
                raise RuntimeError(
                    "Neither dxcam nor mss available. "
                    "Install one: pip install dxcam  OR  pip install mss"
                )

    @property
    def backend(self) -> str:
        return self._backend

    def grab(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the specified region.

        Returns:
            BGR numpy array of shape (target_h, target_w, 3), or None on failure.
        """
        frame = None

        if self._backend == "dxcam":
            frame = self._camera.grab(region=self.region)
            if frame is None:
                # DXcam may return None if no new frame is available
                return None

        elif self._backend == "mss":
            left, top, right, bottom = self.region
            monitor = {
                "left": left, "top": top,
                "width": right - left, "height": bottom - top,
            }
            shot = self._mss.grab(monitor)
            frame = np.array(shot)[:, :, :3]  # BGRA -> BGR

        if frame is None:
            return None

        # Resize to model input size
        h, w = frame.shape[:2]
        if h != self.target_h or w != self.target_w:
            frame = cv2.resize(
                frame, (self.target_w, self.target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        return frame

    def start_continuous(self, fps: int = 120):
        """Start continuous capture (DXcam only). For highest throughput."""
        if self._backend == "dxcam" and self._camera is not None:
            self._camera.start(
                region=self.region,
                target_fps=fps,
                video_mode=True,
            )

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from continuous capture mode."""
        if self._backend != "dxcam" or self._camera is None:
            return self.grab()

        frame = self._camera.get_latest_frame()
        if frame is None:
            return None

        h, w = frame.shape[:2]
        if h != self.target_h or w != self.target_w:
            frame = cv2.resize(
                frame, (self.target_w, self.target_h),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    def stop(self):
        """Stop continuous capture and release resources."""
        if self._backend == "dxcam" and self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass

    def __del__(self):
        self.stop()
