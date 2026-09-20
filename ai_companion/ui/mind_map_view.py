"""Mind Map graph view: concept bubbles linked by shared memories.

Rendering rules, all derived from the data rather than decoration:

  * bubble AREA grows with memories absorbed (sqrt of count, so a 10-memory
    concept does not look ten times more important than a 1-memory one)
  * link opacity and width grow with shared-memory count
  * category sets the hue, kept subtle so the graph reads as one system
  * selecting a concept dims everything it is not connected to

Performance: static. Nothing animates, nothing runs on a timer. Repaints
happen on interaction only, which matters on an integrated GPU already
sharing memory bandwidth with CPU inference.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from ai_companion.services.mind_map import MindMap
from ai_companion.ui.theme import theme

MAX_LABEL = 16


class MindMapView(QWidget):
    """Pan/zoom canvas of concept bubbles."""

    concept_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._map = MindMap()
        self._positions: dict[str, tuple[float, float]] = {}
        self._selected: str = ""
        self._hover: str = ""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_from: Optional[QPointF] = None
        self.setMouseTracking(True)
        self.setMinimumHeight(320)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # -- data ---------------------------------------------------------

    def set_map(self, mind_map: MindMap, positions: dict) -> None:
        self._map = mind_map
        self._positions = dict(positions)
        if self._selected not in mind_map.concepts:
            self._selected = ""
        self.fit()

    @property
    def selected(self) -> str:
        return self._selected

    def fit(self) -> None:
        """Zoom and centre so every bubble is visible."""
        if not self._positions:
            self._zoom, self._pan = 1.0, QPointF(0.0, 0.0)
            self.update()
            return

        max_weight = self._map.max_weight
        radii = {
            k: c.radius(max_weight=max_weight)
            for k, c in self._map.concepts.items()
        }
        pad = 48.0
        xs = [p[0] for p in self._positions.values()]
        ys = [p[1] for p in self._positions.values()]
        biggest = max(radii.values()) if radii else 20.0
        min_x, max_x = min(xs) - biggest - pad, max(xs) + biggest + pad
        min_y, max_y = min(ys) - biggest - pad, max(ys) + biggest + pad

        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        # Never magnify past 1:1 - a three-bubble map blown up looks broken.
        self._zoom = min(self.width() / width, self.height() / height, 1.0)
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        self._pan = QPointF(
            self.width() / 2 - cx * self._zoom,
            self.height() / 2 - cy * self._zoom,
        )
        self.update()

    # -- interaction --------------------------------------------------

    def _to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(x * self._zoom + self._pan.x(),
                       y * self._zoom + self._pan.y())

    def _hit_test(self, point: QPointF) -> str:
        max_weight = self._map.max_weight
        # Reverse order so the visually topmost bubble wins.
        for concept in reversed(self._map.ordered()):
            pos = self._positions.get(concept.key)
            if pos is None:
                continue
            centre = self._to_screen(*pos)
            radius = concept.radius(max_weight=max_weight) * self._zoom
            dx = point.x() - centre.x()
            dy = point.y() - centre.y()
            if dx * dx + dy * dy <= radius * radius:
                return concept.key
        return ""

    def mousePressEvent(self, event) -> None:  # noqa: N802
        hit = self._hit_test(event.position())
        if hit:
            self._selected = "" if hit == self._selected else hit
            self.concept_selected.emit(self._selected)
            self.update()
        else:
            self._drag_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None:
            delta = event.position() - self._drag_from
            self._pan += delta
            self._drag_from = event.position()
            self.update()
            return
        hit = self._hit_test(event.position())
        if hit != self._hover:
            self._hover = hit
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if hit
                else Qt.CursorShape.OpenHandCursor
            )
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_from = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom about the cursor, so the point under it stays put."""
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 1.12 ** steps
        new_zoom = max(0.2, min(3.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        cursor = event.position()
        world_x = (cursor.x() - self._pan.x()) / self._zoom
        world_y = (cursor.y() - self._pan.y()) / self._zoom
        self._zoom = new_zoom
        self._pan = QPointF(
            cursor.x() - world_x * self._zoom,
            cursor.y() - world_y * self._zoom,
        )
        self.update()

    # -- painting -----------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = theme.p
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(p.bg_base))

        if not self._map.concepts:
            self._paint_empty(painter)
            painter.end()
            return

        connected = self._connected_keys()
        self._paint_links(painter, connected)
        self._paint_bubbles(painter, connected)
        painter.end()

    def _connected_keys(self) -> Optional[set]:
        """Keys to keep bright, or None when nothing is selected."""
        if not self._selected:
            return None
        keys = {self._selected}
        for link in self._map.links_for(self._selected):
            keys.add(link.source)
            keys.add(link.target)
        return keys

    def _paint_empty(self, painter: QPainter) -> None:
        p, m = theme.p, theme.m
        font = QFont(m.font_mono.split(",")[0].strip("'\" "))
        font.setPixelSize(m.text_sm)
        painter.setFont(font)
        painter.setPen(QPen(QColor(p.fg_muted)))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "No concepts yet.\n\n"
            "Tag your memories in the MEM tab, then press Rebuild.\n"
            "Concepts form from tags and repeated names.",
        )

    def _paint_links(self, painter: QPainter, connected) -> None:
        p = theme.p
        for link in self._map.links:
            a = self._positions.get(link.source)
            b = self._positions.get(link.target)
            if a is None or b is None:
                continue
            dim = connected is not None and not (
                link.source in connected and link.target in connected
            )
            highlight = (
                self._selected
                and self._selected in (link.source, link.target)
            )
            colour = QColor(p.accent if highlight else p.link)
            # Opacity carries link strength, but from a readable FLOOR.
            # The old 26..146 range rendered the weakest links at 1.08:1
            # contrast against the background - measurably invisible. The
            # floor is what makes a single shared memory visible at all;
            # the range above it still communicates strength.
            alpha = 170 + int(85 * link.strength)
            if highlight:
                alpha = 255
            if dim:
                # Dimming must still leave the link traceable, otherwise
                # selecting a concept appears to delete the rest of the graph.
                alpha = max(80, alpha // 3)
            colour.setAlpha(alpha)
            width = (1.1 + 1.9 * link.strength) * self._zoom
            painter.setPen(QPen(colour, max(0.9, width)))
            painter.drawLine(self._to_screen(*a), self._to_screen(*b))

    def _paint_bubbles(self, painter: QPainter, connected) -> None:
        p, m = theme.p, theme.m
        max_weight = self._map.max_weight
        mono = m.font_mono.split(",")[0].strip("'\" ")

        for concept in self._map.ordered():
            pos = self._positions.get(concept.key)
            if pos is None:
                continue
            centre = self._to_screen(*pos)
            radius = concept.radius(max_weight=max_weight) * self._zoom
            if radius < 1.0:
                continue

            dim = connected is not None and concept.key not in connected
            is_selected = concept.key == self._selected
            base = QColor(concept.color)

            rect = QRectF(
                centre.x() - radius, centre.y() - radius,
                radius * 2, radius * 2,
            )

            fill = QRadialGradient(
                QPointF(centre.x(), centre.y() - radius * 0.35), radius * 1.4
            )
            top = QColor(base)
            top.setAlpha(40 if dim else 130)
            bottom = QColor(p.bg_elevated)
            bottom.setAlpha(60 if dim else 230)
            fill.setColorAt(0.0, top)
            fill.setColorAt(1.0, bottom)

            painter.setBrush(QBrush(fill))
            edge = QColor(base)
            edge.setAlpha(60 if dim else 235)
            painter.setPen(QPen(edge, 2.4 if is_selected else 1.4))
            painter.drawEllipse(rect)

            if is_selected:
                halo = QColor(base)
                halo.setAlpha(90)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(halo, 3.0))
                painter.drawEllipse(rect.adjusted(-5, -5, 5, 5))

            # Label must fit INSIDE the bubble. Elide against the actual
            # pixel width rather than a fixed character count, otherwise
            # "engineering" spills past the circle at small radii.
            font = QFont(mono)
            font.setPixelSize(max(7, int(min(12, radius * 0.40))))
            font.setBold(is_selected or concept.weight >= max_weight * 0.6)
            painter.setFont(font)
            from PySide6.QtGui import QFontMetrics

            metrics = QFontMetrics(font)
            usable = int(radius * 1.72)
            label = metrics.elidedText(
                concept.label, Qt.TextElideMode.ElideRight, max(usable, 12)
            )
            text_colour = QColor(p.fg_muted if dim else p.fg_primary)
            painter.setPen(QPen(text_colour))
            shift = -radius * 0.12 if (concept.weight > 1 and radius > 18) else 0
            painter.drawText(
                rect.adjusted(0, shift, 0, shift),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

            if not dim and concept.weight > 1 and radius > 18:
                small = QFont(mono)
                small.setPixelSize(max(6, int(radius * 0.3)))
                painter.setFont(small)
                painter.setPen(QPen(QColor(p.fg_muted)))
                painter.drawText(
                    QRectF(rect.left(), rect.bottom() - radius * 0.62,
                           rect.width(), radius * 0.5),
                    Qt.AlignmentFlag.AlignCenter,
                    str(concept.weight),
                )
