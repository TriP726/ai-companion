"""Camera panel — manual, on-demand capture only.

DESIGN CONSTRAINT
-----------------
The camera never opens by itself and never records. The user opens it, sees a
live preview, and explicitly captures a single frame. Closing the panel or the
app releases the device.

The preview is driven by a QTimer that pulls frames only while the camera is
open and this panel is visible. There is no background thread holding the
device, and no frame is written to disk unless the user presses Save.

HONEST LIMIT
------------
The loaded model is text-only and cannot see images. "Analyse" therefore
reports measurable frame properties (resolution, brightness, sharpness,
colour balance) rather than describing content. The panel says so on screen,
because a button labelled "Analyse" that silently produced a hallucinated
description would be worse than no button.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai_companion.ui.icons import svg_icon
from ai_companion.ui.theme import theme

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class CameraPanel(QWidget):
    """Manual camera access with single-frame capture."""

    PREVIEW_INTERVAL_MS = 66  # ~15 fps; enough to frame a shot, easy on CPU

    def __init__(self, service_manager: "ServiceManager") -> None:
        super().__init__()
        self._sm = service_manager
        self._last_frame = None
        self._timer = QTimer(self)
        self._timer.setInterval(self.PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._setup_ui()
        self._sync_buttons()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        m = theme.m
        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.space_4, m.space_4, m.space_4, m.space_4)
        layout.setSpacing(m.space_3)

        header = QWidget()
        header.setObjectName("HeaderBar")
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(m.space_4, m.space_2, m.space_4, m.space_2)

        self._status_label = QLabel("CAMERA OFF")
        self._status_label.setObjectName("StatusWarn")
        header_row.addWidget(self._status_label)
        header_row.addStretch(1)

        self._privacy_note = QLabel("MANUAL CAPTURE ONLY - NO RECORDING")
        self._privacy_note.setObjectName("PrivateBadge")
        header_row.addWidget(self._privacy_note)
        layout.addWidget(header)

        self._preview = QLabel()
        self._preview.setObjectName("ChatDisplay")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(360)
        self._preview.setText(
            "Camera is off.\n\nPress Open Camera to start a live preview.\n"
            "Nothing is captured or saved until you press Capture."
        )
        layout.addWidget(self._preview, 1)

        controls = QHBoxLayout()
        controls.setSpacing(m.space_2)

        self._open_btn = QPushButton("Open Camera")
        self._open_btn.setObjectName("PrimaryButton")
        self._open_btn.clicked.connect(self._open_camera)
        controls.addWidget(self._open_btn)

        self._close_btn = QPushButton("Close Camera")
        self._close_btn.setObjectName("DangerButton")
        self._close_btn.clicked.connect(self._close_camera)
        controls.addWidget(self._close_btn)

        self._capture_btn = QPushButton("Capture Frame")
        self._capture_btn.clicked.connect(self._capture)
        controls.addWidget(self._capture_btn)

        self._analyze_btn = QPushButton("Analyse Frame")
        self._analyze_btn.clicked.connect(self._analyze)
        controls.addWidget(self._analyze_btn)

        self._save_btn = QPushButton("Save Frame...")
        self._save_btn.clicked.connect(self._save)
        controls.addWidget(self._save_btn)

        controls.addStretch(1)
        layout.addLayout(controls)

        self._analysis = QTextEdit()
        self._analysis.setObjectName("OutputSurface")
        self._analysis.setReadOnly(True)
        self._analysis.setMaximumHeight(140)
        self._analysis.setPlainText(
            "No frame captured yet.\n"
            "Note: the loaded model is text-only and cannot see images. "
            "Analysis reports measurable frame properties, not a description "
            "of what is in the picture."
        )
        layout.addWidget(self._analysis)

        self.apply_theme()

    def apply_theme(self) -> None:
        p = theme.p
        self._open_btn.setIcon(svg_icon("camera", p.bg_base))
        self._close_btn.setIcon(svg_icon("stop", p.danger))
        self._capture_btn.setIcon(svg_icon("plus", p.fg_secondary))
        self._analyze_btn.setIcon(svg_icon("search", p.fg_secondary))
        self._save_btn.setIcon(svg_icon("workspace", p.fg_secondary))
        for widget in (
            self._status_label,
            self._privacy_note,
            self._open_btn,
            self._close_btn,
        ):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    def _service(self):
        return self._sm.get("Camera")

    def _open_camera(self) -> None:
        svc = self._service()
        if svc is None:
            QMessageBox.warning(self, "Camera", "Camera service unavailable.")
            return
        try:
            ok = svc.open_camera()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Camera", f"Could not open camera:\n{exc}")
            return
        if not ok:
            QMessageBox.warning(
                self,
                "Camera",
                "Could not open the camera.\n\n"
                "It may be missing, disabled in Windows privacy settings, or "
                "in use by another application.",
            )
            return
        self._timer.start()
        self._sync_buttons()

    def _close_camera(self) -> None:
        self._timer.stop()
        svc = self._service()
        if svc is not None:
            svc.close_camera()
        self._preview.setText("Camera is off.")
        self._sync_buttons()

    def _tick(self) -> None:
        """Pull one preview frame. Never stores it."""
        svc = self._service()
        if svc is None or not svc.is_open:
            self._timer.stop()
            self._sync_buttons()
            return
        frame = svc.capture_frame()
        if frame is not None:
            self._show(frame)

    def _show(self, frame) -> None:
        try:
            height, width = frame.shape[:2]
            rgb = frame[:, :, ::-1].copy()  # BGR -> RGB
            image = QImage(
                rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(image).scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview.setPixmap(pixmap)
        except Exception:  # noqa: BLE001 - a bad frame must not kill preview
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _capture(self) -> None:
        svc = self._service()
        if svc is None or not svc.is_open:
            QMessageBox.information(self, "Camera", "Open the camera first.")
            return
        frame = svc.capture_frame()
        if frame is None:
            QMessageBox.warning(self, "Camera", "Capture failed.")
            return
        self._last_frame = frame
        self._show(frame)
        self._analysis.setPlainText(
            f"Frame held in memory: {frame.shape[1]}x{frame.shape[0]}.\n"
            "It is not saved to disk unless you press Save Frame."
        )
        self._sync_buttons()

    def _analyze(self) -> None:
        svc = self._service()
        if svc is None or not svc.is_open:
            QMessageBox.information(self, "Camera", "Open the camera first.")
            return
        result = svc.analyze_current_frame()
        if result is None:
            QMessageBox.warning(self, "Camera", "Analysis failed.")
            return
        self._analysis.setPlainText(
            f"{result}\n\n"
            "These are measured image properties. The loaded model is "
            "text-only and cannot describe what the picture contains."
        )

    def _save(self) -> None:
        if self._last_frame is None:
            QMessageBox.information(
                self, "Camera", "Capture a frame before saving."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame", "frame.png", "PNG Image (*.png)"
        )
        if not path:
            return
        try:
            import cv2

            cv2.imwrite(path, self._last_frame)
            self._analysis.setPlainText(f"Saved to {path}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Camera", f"Could not save:\n{exc}")

    # ------------------------------------------------------------------

    def _sync_buttons(self) -> None:
        svc = self._service()
        is_open = bool(svc is not None and svc.is_open)
        self._open_btn.setEnabled(not is_open)
        self._close_btn.setEnabled(is_open)
        self._capture_btn.setEnabled(is_open)
        self._analyze_btn.setEnabled(is_open)
        self._save_btn.setEnabled(self._last_frame is not None)
        self._status_label.setText("CAMERA LIVE" if is_open else "CAMERA OFF")
        self._status_label.setObjectName(
            "StatusOk" if is_open else "StatusWarn"
        )
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

    def shutdown(self) -> None:
        """Release the device when the app closes or the panel goes away."""
        self._timer.stop()
        svc = self._service()
        if svc is not None and svc.is_open:
            svc.close_camera()
