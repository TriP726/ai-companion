"""HUD boot sequence.

DESIGN RULE: REPORT, DO NOT PRETEND
-----------------------------------
A progress bar that counts to 100% on a timer while the app is already idle is
theatre, and it makes startup *slower* for no information. This screen instead
reports work that genuinely happens:

    * each service is started while its line is on screen
    * the model load - by far the slowest step, several seconds for a 7B on
      CPU - is reported honestly instead of hidden behind a spinner

So the boot screen costs almost nothing beyond the startup that was already
occurring, and every line corresponds to something real. A step that fails
says FAIL in red and the sequence continues, because a cosmetic screen must
never be able to prevent the app from opening.

Skippable at any time with a click or any key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from ai_companion.ui.theme import theme


@dataclass
class BootStep:
    """One line of the boot log.

    `action` is the real work. If it raises, the line is marked FAIL and the
    sequence continues to the next step.
    """

    label: str
    action: Optional[Callable[[], None]] = None


class BootScreen(QWidget):
    """Full-window boot overlay with a live log and a spinning reactor."""

    finished = Signal()

    LINE_MS = 90          # minimum dwell per line, so text is readable
    FADE_MS = 260

    def __init__(
        self,
        steps: list[BootStep],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._steps = steps
        self._results: list[tuple[str, str]] = []  # (label, state)
        self._index = 0
        self._angle = 0.0
        self._done = False
        self._skipped = False

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # One timer drives the ring; it stops the moment the sequence ends.
        self._spin = QTimer(self)
        self._spin.setInterval(50)
        self._spin.timeout.connect(self._tick_spin)

        self._advance = QTimer(self)
        self._advance.setSingleShot(True)
        self._advance.timeout.connect(self._run_next)

    # ------------------------------------------------------------------

    def start(self) -> None:
        self._spin.start()
        self._advance.start(60)

    def _tick_spin(self) -> None:
        import shiboken6

        if not shiboken6.isValid(self):
            return
        self._angle = (self._angle + 7.0) % 360.0
        self.update()

    def _run_next(self) -> None:
        import shiboken6

        if not shiboken6.isValid(self):
            return
        if self._skipped:
            return
        if self._index >= len(self._steps):
            self._complete()
            return

        step = self._steps[self._index]
        state = "OK"
        if step.action is not None:
            try:
                step.action()
            except Exception:  # noqa: BLE001 - a boot screen must never block
                state = "FAIL"
        self._results.append((step.label, state))
        self._index += 1
        self.update()
        self._advance.start(self.LINE_MS)

    def _complete(self) -> None:
        if self._done:
            return
        self._done = True
        self._spin.stop()
        self._advance.stop()
        self.finished.emit()

    def skip(self) -> None:
        """Run any remaining work immediately, then close.

        Skipping must not skip the WORK - only the waiting. Otherwise a user
        who clicks early gets an app with unstarted services.
        """
        if self._done:
            return
        self._skipped = True
        self._advance.stop()
        while self._index < len(self._steps):
            step = self._steps[self._index]
            if step.action is not None:
                try:
                    step.action()
                except Exception:  # noqa: BLE001
                    pass
            self._index += 1
        self._complete()

    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.skip()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        self.skip()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._spin.stop()
        self._advance.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Stop every timer before the widget goes away.

        A QTimer still armed when its target is destroyed will fire into
        freed memory and take the whole interpreter down with a bus error.
        """
        self._spin.stop()
        self._advance.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = theme.p
        m = theme.m
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(p.bg_base))

        # faint grid, same language as the main window
        painter.setPen(QPen(QColor(p.grid), 1))
        for x in range(0, self.width(), 34):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 34):
            painter.drawLine(0, y, self.width(), y)

        cx = self.width() / 2
        cy = self.height() / 2 - 70

        self._paint_reactor(painter, cx, cy)
        self._paint_title(painter, cx, cy)
        self._paint_log(painter, cx, cy)
        self._paint_hint(painter)
        painter.end()

    def _paint_reactor(self, painter: QPainter, cx: float, cy: float) -> None:
        p = theme.p
        radius = 54.0
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        accent = QColor(p.accent)

        glow = QRadialGradient(rect.center(), radius * 1.6)
        glow.setColorAt(0.0, QColor(accent.red(), accent.green(),
                                    accent.blue(), 120))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(rect.adjusted(-40, -40, 40, 40))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(p.border_strong), 1.5))
        painter.drawEllipse(rect)
        painter.drawEllipse(rect.adjusted(12, 12, -12, -12))

        painter.setPen(QPen(accent, 2.5))
        for offset in (0, 120, 240):
            painter.drawArc(
                rect.adjusted(6, 6, -6, -6),
                int((self._angle + offset) * 16),
                58 * 16,
            )
        painter.setPen(QPen(accent, 1.5))
        for offset in (60, 180, 300):
            painter.drawArc(
                rect.adjusted(20, 20, -20, -20),
                int((-self._angle * 1.4 + offset) * 16),
                40 * 16,
            )

        core = rect.adjusted(
            radius * 0.62, radius * 0.62, -radius * 0.62, -radius * 0.62
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(core)

    def _paint_title(self, painter: QPainter, cx: float, cy: float) -> None:
        p = theme.p
        m = theme.m
        mono = m.font_mono.split(",")[0].strip("'\" ")

        font = QFont(mono)
        font.setPixelSize(20)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 7)
        painter.setFont(font)
        painter.setPen(QPen(QColor(p.fg_primary)))
        painter.drawText(
            QRectF(0, cy + 74, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter,
            "A.I. COMPANION",
        )

        font.setPixelSize(m.text_xs)
        font.setBold(False)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        painter.setFont(font)
        painter.setPen(QPen(QColor(p.fg_muted)))
        painter.drawText(
            QRectF(0, cy + 104, self.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            "LOCAL  ·  OFFLINE  ·  PRIVATE",
        )

    def _paint_log(self, painter: QPainter, cx: float, cy: float) -> None:
        p = theme.p
        m = theme.m
        mono = m.font_mono.split(",")[0].strip("'\" ")
        font = QFont(mono)
        font.setPixelSize(m.text_sm)
        painter.setFont(font)

        # Show a trailing window so the block never grows off-screen.
        visible = self._results[-8:]
        width = 420.0
        left = cx - width / 2
        top = cy + 150

        for row, (label, state) in enumerate(visible):
            y = top + row * 19
            painter.setPen(QPen(QColor(p.fg_secondary)))
            painter.drawText(
                QRectF(left, y, width - 60, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            colour = p.success if state == "OK" else p.danger
            painter.setPen(QPen(QColor(colour)))
            painter.drawText(
                QRectF(left, y, width, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                state,
            )

    def _paint_hint(self, painter: QPainter) -> None:
        p = theme.p
        m = theme.m
        mono = m.font_mono.split(",")[0].strip("'\" ")
        font = QFont(mono)
        font.setPixelSize(m.text_xs)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        painter.setFont(font)
        painter.setPen(QPen(QColor(p.fg_muted)))
        painter.drawText(
            QRectF(0, self.height() - 42, self.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            "CLICK OR PRESS ANY KEY TO SKIP",
        )


def build_steps(service_manager, config) -> list[BootStep]:
    """Boot steps that start the real services, in order.

    Each service is started as its line appears, so the log reflects actual
    progress rather than a scripted animation.
    """
    steps: list[BootStep] = [
        BootStep("INITIALISING CORE", None),
    ]

    labels = {
        "Vault": "MOUNTING VAULT",
        "LLM": "LOADING LANGUAGE MODEL",
        "Memory": "RESTORING MEMORY STORE",
        "Graph": "BUILDING KNOWLEDGE GRAPH",
        "Camera": "REGISTERING CAMERA",
        "ImageGen": "REGISTERING IMAGE WORKER",
        "Speech": "REGISTERING SPEECH",
        "FileWorkspace": "OPENING FILE WORKSPACE",
        "CodeWorkspace": "OPENING CODE SANDBOX",
        "Sandbox3D": "OPENING 3D SANDBOX",
    }

    for name, service in service_manager.services.items():
        label = labels.get(name, f"STARTING {name.upper()}")
        steps.append(BootStep(label, _starter(service)))

    steps.append(BootStep("INTERFACE ONLINE", None))
    return steps


def _starter(service) -> Callable[[], None]:
    def run() -> None:
        if not getattr(service, "is_running", False):
            service.start()

    return run
