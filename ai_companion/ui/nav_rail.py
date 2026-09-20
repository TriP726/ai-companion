"""Vertical HUD navigation rail.

Each destination is a custom-painted hexagonal cell rather than a QPushButton,
because the selected state needs a glow, a leading accent bar and a chamfered
outline that a stylesheet cannot express.

Motion policy: hover and selection are instant repaints, not animations. The
only moving part in the rail is the reactor at the top, and it moves only
while the model is generating.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ai_companion.ui.icons import svg_icon
from ai_companion.ui.theme import theme


class NavCell(QWidget):
    """One hexagonal rail entry: icon over a short caption."""

    activated = Signal()

    def __init__(
        self,
        icon_name: str,
        label: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.icon_name = icon_name
        self.label = label
        self._selected = False
        self._hover = False
        self.setFixedHeight(66)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self.update()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.activated.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = theme.p
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        inset = 6
        cut = 9
        body = QPainterPath()
        body.moveTo(inset + cut, 3)
        body.lineTo(w - inset, 3)
        body.lineTo(w - inset, h - 3)
        body.lineTo(inset + cut, h - 3)
        body.lineTo(inset, h - 3 - cut)
        body.lineTo(inset, 3 + cut)
        body.closeSubpath()

        if self._selected:
            fill = QColor(p.bg_active)
            painter.fillPath(body, QBrush(fill))
            painter.setPen(QPen(QColor(p.accent), 1.4))
            painter.drawPath(body)
            # Leading accent bar - the primary "you are here" signal.
            painter.fillRect(0, 10, 3, h - 20, QColor(p.accent))
        elif self._hover:
            painter.fillPath(body, QBrush(QColor(p.bg_hover)))
            painter.setPen(QPen(QColor(p.border_strong), 1))
            painter.drawPath(body)

        colour = p.accent if self._selected else (
            p.fg_secondary if self._hover else p.fg_muted
        )
        icon = svg_icon(self.icon_name, colour, 22)
        icon_x = int((w - 22) / 2) + 2
        icon.paint(painter, icon_x, 10, 22, 22)

        font = QFont(theme.m.font_mono.split(",")[0].strip("'\" "))
        font.setPixelSize(theme.m.text_xs)
        font.setBold(self._selected)
        painter.setFont(font)
        painter.setPen(QPen(QColor(colour)))
        painter.drawText(
            QRectF(0, 38, w, 18),
            Qt.AlignmentFlag.AlignCenter,
            self.label,
        )
        painter.end()


class NavRail(QWidget):
    """Vertical rail of destinations plus pinned actions."""

    navigated = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("NavRail")
        self.setFixedWidth(theme.m.rail_width)
        self._cells: list[NavCell] = []
        self._actions: list[NavCell] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, theme.m.space_3, 0, theme.m.space_3)
        self._layout.setSpacing(theme.m.space_1)

    # -- construction -------------------------------------------------

    def add_destination(self, icon_name: str, label: str) -> None:
        index = len(self._cells)
        cell = NavCell(icon_name, label)
        cell.activated.connect(lambda i=index: self._on_activated(i))
        self._cells.append(cell)
        self._layout.addWidget(cell)

    def add_action(
        self, icon_name: str, label: str, callback: Callable[[], None]
    ) -> None:
        cell = NavCell(icon_name, label)
        cell.activated.connect(callback)
        self._actions.append(cell)
        self._layout.addWidget(cell)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def add_widget(self, widget: QWidget) -> None:
        """Pin an arbitrary widget (used for the reactor core)."""
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, theme.m.space_2)
        box.addWidget(widget, 0, Qt.AlignmentFlag.AlignHCenter)
        self._layout.addWidget(holder)

    # -- state --------------------------------------------------------

    def _on_activated(self, index: int) -> None:
        self.set_current(index)
        self.navigated.emit(index)

    def set_current(self, index: int) -> None:
        for i, cell in enumerate(self._cells):
            cell.set_selected(i == index)

    def refresh_icons(self) -> None:
        for cell in self._cells + self._actions:
            cell.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = theme.p
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(p.bg_surface))
        painter.setPen(QPen(QColor(p.border), 1))
        painter.drawLine(
            self.width() - 1, 0, self.width() - 1, self.height()
        )
        painter.end()
