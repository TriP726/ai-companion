"""Custom-painted HUD widgets for the holographic interface.

PERFORMANCE CONTRACT
--------------------
This machine runs CPU inference on an integrated-graphics APU. Every frame
painted here competes with token generation, so the rule throughout is:

    nothing animates unless something is actually happening.

Idle cost is zero. `ReactorCore` and `PulseBar` only run their timers while
explicitly activated, and they stop themselves the moment they are hidden.
Static depth comes from gradients, corner brackets and border layering, all
of which cost one paint and then nothing.
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    Property,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ai_companion.ui.theme import theme


# ----------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------


class HudPanel(QFrame):
    """A panel with angled corners and bracket accents.

    Drawn rather than styled because Qt stylesheets cannot cut corners off a
    rectangle; a chamfered outline is what separates 'HUD' from 'dark theme'.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        chamfer: int = 14,
        brackets: bool = True,
        filled: bool = True,
    ) -> None:
        super().__init__(parent)
        self._chamfer = chamfer
        self._brackets = brackets
        self._filled = filled
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def _outline(self) -> QPainterPath:
        c = self._chamfer
        w = self.width() - 1
        h = self.height() - 1
        path = QPainterPath()
        path.moveTo(c, 0)
        path.lineTo(w - c, 0)
        path.lineTo(w, c)
        path.lineTo(w, h - c)
        path.lineTo(w - c, h)
        path.lineTo(c, h)
        path.lineTo(0, h - c)
        path.lineTo(0, c)
        path.closeSubpath()
        return path

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        p = theme.p
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = self._outline()

        if self._filled:
            grad = QLinearGradient(0, 0, 0, self.height())
            grad.setColorAt(0.0, QColor(p.bg_surface))
            grad.setColorAt(1.0, QColor(p.bg_base))
            painter.fillPath(path, QBrush(grad))

        painter.setPen(QPen(QColor(p.border_strong), 1))
        painter.drawPath(path)

        if self._brackets:
            self._draw_brackets(painter)
        painter.end()

    def _draw_brackets(self, painter: QPainter) -> None:
        """Short accent strokes at each corner - the classic HUD tell."""
        p = theme.p
        length = 18
        c = self._chamfer
        w = self.width() - 1
        h = self.height() - 1
        pen = QPen(QColor(p.accent), 2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        # top-left
        painter.drawLine(c, 0, c + length, 0)
        painter.drawLine(0, c, 0, c + length)
        # top-right
        painter.drawLine(w - c, 0, w - c - length, 0)
        painter.drawLine(w, c, w, c + length)
        # bottom-left
        painter.drawLine(c, h, c + length, h)
        painter.drawLine(0, h - c, 0, h - c - length)
        # bottom-right
        painter.drawLine(w - c, h, w - c - length, h)
        painter.drawLine(w, h - c, w, h - c - length)


class GridBackground(QWidget):
    """Faint static grid with a radial vignette. Painted once per resize."""

    def __init__(self, parent: Optional[QWidget] = None, step: int = 34) -> None:
        super().__init__(parent)
        self._step = step
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = theme.p
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(p.bg_base))

        pen = QPen(QColor(p.grid), 1)
        painter.setPen(pen)
        step = self._step
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)

        # Vignette: pulls the eye to the centre and hides grid tiling at edges.
        glow = QRadialGradient(
            QPointF(self.width() / 2, self.height() / 2),
            max(self.width(), self.height()) * 0.75,
        )
        glow.setColorAt(0.0, QColor(0, 0, 0, 0))
        glow.setColorAt(1.0, QColor(0, 0, 0, 170))
        painter.fillRect(self.rect(), QBrush(glow))
        painter.end()


# ----------------------------------------------------------------------
# Reactor core
# ----------------------------------------------------------------------


class ReactorCore(QWidget):
    """Concentric arc-reactor rings.

    Static when idle - the rings are drawn once and left alone. Only spins
    while `set_active(True)`, which the chat panel drives from generation
    start/stop, so the cost is paid only while the model is working.
    """

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None, size: int = 46) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0.0
        self._active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._timer = QTimer(self)
        self._timer.setInterval(50)  # 20fps is plenty for a spinner
        self._timer.timeout.connect(self._tick)

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        if active:
            self._timer.start()
        else:
            self._timer.stop()
            self._angle = 0.0
        self.update()

    def _tick(self) -> None:
        self._angle = (self._angle + 6.0) % 360.0
        self.update()

    def hideEvent(self, event) -> None:  # noqa: N802
        # Never leave a timer running behind a hidden widget.
        self._timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        if self._active:
            self._timer.start()
        super().showEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = theme.p
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        centre = rect.center()

        # Core glow
        glow = QRadialGradient(centre, rect.width() / 2)
        colour = QColor(p.accent)
        glow.setColorAt(0.0, QColor(colour.red(), colour.green(),
                                    colour.blue(), 200))
        glow.setColorAt(0.55, QColor(colour.red(), colour.green(),
                                     colour.blue(), 60))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(rect)

        # Outer ring
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(p.border_strong), 1.5))
        painter.drawEllipse(rect)

        # Rotating arc segments
        painter.setPen(QPen(QColor(p.accent), 2))
        inner = rect.adjusted(5, 5, -5, -5)
        for offset in (0, 120, 240):
            start = int((self._angle + offset) * 16)
            painter.drawArc(inner, start, 60 * 16)

        # Inner core
        core = rect.adjusted(
            rect.width() * 0.33, rect.height() * 0.33,
            -rect.width() * 0.33, -rect.height() * 0.33,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(p.accent)))
        painter.drawEllipse(core)
        painter.end()


# ----------------------------------------------------------------------
# Status strip
# ----------------------------------------------------------------------


class PulseBar(QWidget):
    """Thin activity bar. A sweeping highlight only while active."""

    def __init__(self, parent: Optional[QWidget] = None, height: int = 2) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._pos = 0.0
        self._active = False
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        if active:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._pos = (self._pos + 0.018) % 1.0
        self.update()

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        if self._active:
            self._timer.start()
        super().showEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = theme.p
        painter = QPainter(self)
        if not self._active:
            # Idle: blend into the header's bottom edge instead of drawing a
            # visible purple band across the window.
            painter.fillRect(self.rect(), QColor(p.bg_base))
            painter.end()
            return
        painter.fillRect(self.rect(), QColor(p.bg_base))
        accent = QColor(p.accent)
        grad = QLinearGradient(0, 0, self.width(), 0)
        left = max(0.0, self._pos - 0.12)
        right = min(1.0, self._pos + 0.12)
        grad.setColorAt(0.0, QColor(accent.red(), accent.green(),
                                    accent.blue(), 0))
        grad.setColorAt(left, QColor(accent.red(), accent.green(),
                                     accent.blue(), 0))
        grad.setColorAt(self._pos, accent)
        grad.setColorAt(right, QColor(accent.red(), accent.green(),
                                      accent.blue(), 0))
        grad.setColorAt(1.0, QColor(accent.red(), accent.green(),
                                    accent.blue(), 0))
        painter.fillRect(self.rect(), QBrush(grad))
        painter.end()


class TelemetryStrip(QWidget):
    """Monospace key/value readouts, HUD style."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items: dict[str, QLabel] = {}
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(theme.m.space_4)

    def add_field(self, key: str, label: str, value: str = "--") -> None:
        holder = QWidget()
        box = QHBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.m.space_1)

        caption = QLabel(label.upper())
        caption.setObjectName("TelemetryKey")
        box.addWidget(caption)

        val = QLabel(value)
        val.setObjectName("TelemetryValue")
        box.addWidget(val)

        self._items[key] = val
        self._row.addWidget(holder)

    def set_value(self, key: str, value: str) -> None:
        if key in self._items:
            self._items[key].setText(value)

    def add_stretch(self) -> None:
        self._row.addStretch(1)


def fade_in(widget: QWidget, duration: int = 140) -> Optional[QPropertyAnimation]:
    """Fade a widget in once, on an explicit user action.

    Starts at 0.55 rather than 0.0. A full fade from transparent looks like
    the panel is broken mid-flight (and a screenshot taken during the
    animation shows a half-rendered UI), while a short lift from mostly-opaque
    reads as a HUD refresh without ever looking unfinished.

    Any in-flight animation on the same widget is stopped first, so rapid tab
    switching cannot leave a panel stranded at partial opacity.
    """
    existing = widget.findChild(QPropertyAnimation, "hud_fade")
    if existing is not None:
        existing.stop()
        widget.setGraphicsEffect(None)

    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setObjectName("hud_fade")
    anim.setDuration(duration)
    anim.setStartValue(0.55)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _cleanup() -> None:
        # A lingering QGraphicsOpacityEffect forces every later repaint of
        # this widget through an offscreen buffer - real cost on an iGPU.
        widget.setGraphicsEffect(None)

    anim.finished.connect(_cleanup)
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    return anim
