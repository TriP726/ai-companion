"""Camera service — manual capture only, no automatic recording."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

from ai_companion.infrastructure.base_service import BaseService


class CameraService(BaseService):
    """Manages camera access for manual frame capture.

    Security constraints:
    - NO automatic recording
    - NO continuous streaming to disk
    - Only captures a single frame when explicitly requested
    - Camera is opened on-demand and closed immediately after capture
    """

    service_name = "Camera"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._camera: Any = None
        self._camera_open = False
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._camera_open

    def start(self) -> None:
        super().start()
        # Camera is NOT opened on service start — opened on demand only

    def stop(self) -> None:
        self.close_camera()
        super().stop()

    def open_camera(self) -> bool:
        """Open the camera device for capture."""
        with self._lock:
            if self._camera_open:
                return True

            try:
                import cv2
                device = self._config.camera.device_index
                self._camera = cv2.VideoCapture(device)
                if not self._camera.isOpened():
                    self.emit_error(f"Cannot open camera device {device}")
                    self._camera = None
                    return False

                # Set resolution
                self._camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    self._config.camera.resolution_width,
                )
                self._camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    self._config.camera.resolution_height,
                )

                self._camera_open = True
                self._signal_bus.camera.camera_opened.emit()
                return True

            except ImportError:
                self.emit_error("opencv-python not installed")
                return False
            except Exception as e:
                self.emit_error(f"Camera error: {e}")
                return False

    def close_camera(self) -> None:
        """Close the camera device."""
        with self._lock:
            if self._camera is not None:
                try:
                    self._camera.release()
                except Exception:
                    pass
                self._camera = None
                self._camera_open = False
                self._signal_bus.camera.camera_closed.emit()

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera.

        Opens the camera if needed, captures one frame, then closes it.
        Returns the frame as a numpy array (BGR), or None on failure.
        """
        # Guard: auto_record must be False
        if self._config.camera.auto_record:
            self.emit_error("Camera auto_record is enabled (security violation)")
            return None

        opened_here = False
        if not self._camera_open:
            if not self.open_camera():
                return None
            opened_here = True

        try:
            with self._lock:
                if self._camera is None:
                    return None
                ret, frame = self._camera.read()
                if not ret:
                    self.emit_error("Failed to capture frame")
                    return None

            self._signal_bus.camera.frame_captured.emit(frame)
            return frame

        finally:
            # Always close camera after capture (no continuous recording)
            if opened_here:
                self.close_camera()

    def analyze_current_frame(self) -> Optional[str]:
        """Capture a frame and return a basic analysis.

        For a full implementation, this would send the frame to the LLM
        for description. Currently returns basic image stats.
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        analysis_id = str(int(time.time()))
        h, w = frame.shape[:2]
        mean_brightness = float(frame.mean())

        result = (
            f"Frame captured: {w}x{h}, "
            f"mean brightness: {mean_brightness:.1f}/255"
        )
        self._signal_bus.camera.analysis_complete.emit(analysis_id, result)
        return result
