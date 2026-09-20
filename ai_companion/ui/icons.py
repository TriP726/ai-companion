"""Inline SVG icons rendered to QIcon at the current accent colour.

No image files and no emoji: emoji render inconsistently across Windows font
fallbacks and immediately read as amateur. These are hand-written 24x24 paths
recoloured at runtime, so they follow the active theme automatically.
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

# 24x24 viewBox, stroke-based so a single colour token drives everything.
_PATHS: dict[str, str] = {
    "chat": (
        '<path d="M4 5h16v11H9l-5 4V5z" fill="none" stroke="{c}" '
        'stroke-width="1.6" stroke-linejoin="round"/>'
        '<path d="M8 9h8M8 12h5" stroke="{c}" stroke-width="1.6" '
        'stroke-linecap="round"/>'
    ),
    "memory": (
        '<circle cx="12" cy="12" r="3" fill="none" stroke="{c}" '
        'stroke-width="1.6"/>'
        '<path d="M12 4v5M12 15v5M4 12h5M15 12h5M6.5 6.5l3.2 3.2'
        'M14.3 14.3l3.2 3.2M17.5 6.5l-3.2 3.2M9.7 14.3l-3.2 3.2" '
        'stroke="{c}" stroke-width="1.6" stroke-linecap="round"/>'
    ),
    "graph": (
        '<circle cx="6" cy="7" r="2.3" fill="none" stroke="{c}" '
        'stroke-width="1.6"/>'
        '<circle cx="18" cy="7" r="2.3" fill="none" stroke="{c}" '
        'stroke-width="1.6"/>'
        '<circle cx="12" cy="17" r="2.3" fill="none" stroke="{c}" '
        'stroke-width="1.6"/>'
        '<path d="M8 8.3l2.5 6.8M16 8.3l-2.5 6.8M8.3 7h7.4" '
        'stroke="{c}" stroke-width="1.6"/>'
    ),
    "workspace": (
        '<path d="M3 7a1 1 0 011-1h5l2 2h9a1 1 0 011 1v9a1 1 0 01-1 1H4'
        'a1 1 0 01-1-1V7z" fill="none" stroke="{c}" stroke-width="1.6" '
        'stroke-linejoin="round"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3" fill="none" stroke="{c}" '
        'stroke-width="1.6"/>'
        '<path d="M12 3v2.5M12 18.5V21M3 12h2.5M18.5 12H21'
        'M5.6 5.6l1.8 1.8M16.6 16.6l1.8 1.8M18.4 5.6l-1.8 1.8'
        'M7.4 16.6l-1.8 1.8" stroke="{c}" stroke-width="1.6" '
        'stroke-linecap="round"/>'
    ),
    "send": (
        '<path d="M4 12l16-8-6 8 6 8-16-8z" fill="none" stroke="{c}" '
        'stroke-width="1.6" stroke-linejoin="round"/>'
    ),
    "attach": (
        '<path d="M16 7l-6.5 6.5a2.5 2.5 0 103.5 3.5L19 10a4.5 4.5 0 '
        '10-6.4-6.4L6 10.2" fill="none" stroke="{c}" stroke-width="1.6" '
        'stroke-linecap="round"/>'
    ),
    "plus": (
        '<path d="M12 5v14M5 12h14" stroke="{c}" stroke-width="1.8" '
        'stroke-linecap="round"/>'
    ),
    "trash": (
        '<path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13M10 11v6M14 11v6" '
        'fill="none" stroke="{c}" stroke-width="1.6" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    ),
    "stop": (
        '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="none" '
        'stroke="{c}" stroke-width="1.6"/>'
    ),
    "theme": (
        '<path d="M12 4a8 8 0 100 16 6 6 0 010-16z" fill="none" '
        'stroke="{c}" stroke-width="1.6" stroke-linejoin="round"/>'
    ),
    "lock": (
        '<rect x="5" y="10.5" width="14" height="9.5" rx="1.5" fill="none" '
        'stroke="{c}" stroke-width="1.6"/>'
        '<path d="M8 10.5V7.5a4 4 0 118 0v3" fill="none" stroke="{c}" '
        'stroke-width="1.6"/>'
    ),
    "camera": (
        '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4'
        'l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>'
        '<circle cx="12" cy="13" r="4"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="6" fill="none" stroke="{c}" '
        'stroke-width="1.6"/>'
        '<path d="M15.5 15.5L20 20" stroke="{c}" stroke-width="1.6" '
        'stroke-linecap="round"/>'
    ),
}


def svg_icon(name: str, color: str, size: int = 24) -> QIcon:
    """Render a named icon in `color`. Unknown names give an empty QIcon."""
    path = _PATHS.get(name)
    if path is None:
        return QIcon()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size}" viewBox="0 0 24 24">{path.format(c=color)}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


def available_icons() -> list[str]:
    return sorted(_PATHS)
