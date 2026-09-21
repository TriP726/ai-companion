"""Graph panel — visual knowledge graph with search and layout."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class GraphView(QGraphicsView):
    """Custom graphics view for the knowledge graph."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        # Styling comes from the global stylesheet so the light/dark toggle
        # reaches this widget too.
        self._node_items: dict[str, QGraphicsEllipseItem] = {}
        self._edge_items: dict[str, QGraphicsLineItem] = {}

    def fit_contents(self) -> None:
        """Zoom so the whole graph is visible.

        The layout canvas grows with node count, so without this a larger
        graph simply runs off the edge of the viewport.
        """
        rect = self._scene.itemsBoundingRect()
        if rect.isEmpty():
            return
        margin = 40
        rect.adjust(-margin, -margin, margin, margin)
        self._scene.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        # Never magnify past 1:1 - a two-node graph blown up to fill the
        # window looks broken.
        if self.transform().m11() > 1.0:
            self.resetTransform()
            self.centerOn(rect.center())

    def clear_graph(self) -> None:
        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()

    MAX_LABEL_CHARS = 26

    def add_node(
        self,
        node_id: str,
        label: str,
        x: float,
        y: float,
        color: str = "",
        radius: float = 25.0,
        is_tag: bool = False,
    ) -> None:
        """Add a node to the scene.

        Tag nodes are drawn smaller and dimmer than memory nodes. Without a
        visual hierarchy every circle looks equally important and the graph
        reads as noise rather than structure.
        """
        from ai_companion.ui.theme import theme

        color = color or theme.p.accent
        if is_tag:
            radius = radius * 0.62

        fill = QColor(color)
        fill.setAlpha(70 if is_tag else 190)
        pen_width = 1.4 if is_tag else 2.2

        ellipse = self._scene.addEllipse(
            x - radius, y - radius,
            radius * 2, radius * 2,
            QPen(QColor(color), pen_width),
            QBrush(fill),
        )
        ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        ellipse.setData(0, node_id)
        ellipse.setZValue(1 if is_tag else 2)
        ellipse.setToolTip(label)
        self._node_items[node_id] = ellipse

        # Labels are elided: a full memory summary under every circle
        # collides with its neighbours and the graph becomes unreadable.
        shown = label
        if len(shown) > self.MAX_LABEL_CHARS:
            shown = shown[: self.MAX_LABEL_CHARS - 1].rstrip() + "\u2026"

        font = QFont("Arial", 7 if is_tag else 8)
        font.setBold(not is_tag)
        text = self._scene.addText(shown, font)
        text.setDefaultTextColor(
            QColor(theme.p.fg_muted if is_tag else theme.p.fg_primary)
        )
        text.setPos(x - text.boundingRect().width() / 2, y + radius + 2)
        text.setData(0, f"label_{node_id}")
        text.setZValue(3)
        text.setToolTip(label)

    def add_edge(
        self,
        edge_id: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "",
        label: str = "",
    ) -> None:
        """Add an edge line between two positions."""
        from ai_companion.ui.theme import theme

        # Edges use the dedicated link token, not border_strong: at 2.07:1
        # against the background a structural border is fine for a 1px panel
        # divider but effectively invisible as a line across open space.
        color = color or theme.p.link
        pen = QPen(QColor(color), 1.6, Qt.PenStyle.DashLine)
        line = self._scene.addLine(x1, y1, x2, y2, pen)
        line.setData(0, edge_id)
        line.setZValue(-1)
        self._edge_items[edge_id] = line

        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            text = self._scene.addText(label, QFont("Arial", 7))
            text.setDefaultTextColor(QColor(theme.p.fg_secondary))
            text.setPos(mid_x, mid_y)

    def highlight_node(self, node_id: str) -> None:
        """Highlight a node by making it larger."""
        item = self._node_items.get(node_id)
        if item:
            self.centerOn(item)


class GraphPanel(QWidget):
    """UI for the visual knowledge graph."""

    def __init__(self, service_manager: ServiceManager) -> None:
        super().__init__()
        self._sm = service_manager
        self._setup_ui()
        self._connect_signals()
        # Services start before this panel exists, so node_added/edge_added
        # have already fired with nothing connected. Load current state
        # directly rather than waiting for a signal that will not come.
        self._refresh_graph()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Toolbar
        toolbar = QHBoxLayout()

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search graph...")
        self._search_input.returnPressed.connect(self._on_search)
        toolbar.addWidget(self._search_input)

        self._type_filter = QComboBox()
        self._type_filter.addItems([
            "All Types", "Memory", "Chat", "File", "Person",
            "Project", "Idea", "Tag", "Topic",
        ])
        self._type_filter.currentTextChanged.connect(self._on_search)
        toolbar.addWidget(self._type_filter)

        self._sync_btn = QPushButton("Sync from Memories")
        self._sync_btn.setObjectName("PrimaryButton")
        self._sync_btn.setToolTip(
            "Rebuild the graph from your approved memories and their tags."
        )
        self._sync_btn.clicked.connect(self._sync_from_memories)
        toolbar.addWidget(self._sync_btn)

        self._layout_btn = QPushButton("Auto Layout")
        self._layout_btn.clicked.connect(self._auto_layout)
        toolbar.addWidget(self._layout_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_graph)
        toolbar.addWidget(self._refresh_btn)

        self._toggle3d_btn = QPushButton("3D View")
        self._toggle3d_btn.setCheckable(True)
        self._toggle3d_btn.setToolTip(
            "True-3D graph: drag to orbit, wheel to zoom, "
            "right-drag to pan, click a node to select, "
            "double-click empty space to reframe."
        )
        self._toggle3d_btn.clicked.connect(self._on_toggle_3d)
        toolbar.addWidget(self._toggle3d_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_graph)
        toolbar.addWidget(self._clear_btn)

        layout.addLayout(toolbar)

        # Graph views — 2D canvas and true-3D OpenGL, stacked. The 3D view
        # degrades gracefully: if QtOpenGL itself can't even be imported or
        # the widget reports GL failure at first paint, the toggle disables
        # and 2D stays. Never let a driver problem kill the graph tab.
        self._graph_view = GraphView()
        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self._graph_view)  # index 0 = 2D

        self._graph3d_view = None
        self._last_nodes: dict[str, object] = {}
        try:
            from ai_companion.ui.graph3d_view import Graph3DView

            self._graph3d_view = Graph3DView()
            self._graph3d_view.gl_unavailable.connect(self._on_3d_unavailable)
            self._graph3d_view.node_selected.connect(self._on_3d_node_selected)
            self._view_stack.addWidget(self._graph3d_view)  # index 1 = 3D
        except Exception as exc:  # noqa: BLE001 - import/env failure only
            self._toggle3d_btn.setEnabled(False)
            self._toggle3d_btn.setToolTip(f"3D view unavailable: {exc}")
        layout.addWidget(self._view_stack)

        # Stats
        self._stats_label = QLabel("No graph data")
        self._stats_label.setObjectName("MonoText")
        layout.addWidget(self._stats_label)

    def _connect_signals(self) -> None:
        sb = self._sm.signal_bus
        sb.graph.node_added.connect(lambda _: self._refresh_graph())
        sb.graph.node_updated.connect(lambda _: self._refresh_graph())
        sb.graph.node_deleted.connect(lambda _: self._refresh_graph())
        sb.graph.edge_added.connect(lambda _: self._refresh_graph())
        sb.graph.edge_deleted.connect(lambda _: self._refresh_graph())
        sb.graph.graph_cleared.connect(self._on_graph_cleared)

    def _refresh_graph(self) -> None:
        graph_svc = self._sm.get("Graph")
        if not graph_svc:
            return

        self._graph_view.clear_graph()

        # Get all nodes and edges
        result = graph_svc.search()
        nodes = result["nodes"]
        edges = result["edges"]

        # Compute layout
        positions = graph_svc.compute_layout()

        # Draw edges first
        for edge in edges:
            src_pos = positions.get(edge.source_id)
            tgt_pos = positions.get(edge.target_id)
            if src_pos and tgt_pos:
                self._graph_view.add_edge(
                    edge.id,
                    src_pos[0], src_pos[1],
                    tgt_pos[0], tgt_pos[1],
                    label=edge.label or edge.edge_type.value,
                )

        # Draw nodes
        for node in nodes:
            pos = positions.get(node.id, (400, 300))
            self._graph_view.add_node(
                node.id,
                node.label,
                pos[0], pos[1],
                color=node.color,
                is_tag=node.node_type.value == "tag",
            )

        # Feed the 3D view the same graph (it does its own 3D layout).
        self._last_nodes = {n.id: n for n in nodes}
        if self._graph3d_view is not None:
            self._graph3d_view.set_graph(
                [
                    {
                        "id": n.id,
                        "label": n.label,
                        "color": n.color,
                        "is_tag": n.node_type.value == "tag",
                    }
                    for n in nodes
                ],
                [(e.source_id, e.target_id) for e in edges],
            )

        # Update stats
        stats = graph_svc.get_stats()
        if stats["total_nodes"] == 0:
            self._update_empty_state()
            return
        # Render the type breakdown readably. This used to interpolate the
        # raw dict, printing "<NodeType.PERSON: 'person'>: 1" at the user.
        types = stats.get("node_types") or {}
        parts = []
        for key, count in sorted(types.items(), key=lambda kv: -kv[1]):
            name = getattr(key, "value", str(key))
            parts.append(f"{name} {count}")
        self._graph_view.fit_contents()
        breakdown = "  ".join(parts) if parts else "none"
        self._stats_label.setText(
            f"Nodes: {stats['total_nodes']} | "
            f"Edges: {stats['total_edges']} | "
            f"{breakdown}"
        )

    def _on_search(self) -> None:
        self._refresh_graph()

    def _auto_layout(self) -> None:
        self._refresh_graph()

    def _clear_graph(self) -> None:
        graph_svc = self._sm.get("Graph")
        if graph_svc:
            graph_svc.clear()

    def _sync_from_memories(self) -> None:
        """Project approved memories and their tags into the graph."""
        from ai_companion.services.graph_sync import sync_memories_to_graph

        mem_svc = self._sm.get("Memory")
        graph_svc = self._sm.get("Graph")
        if mem_svc is None or graph_svc is None:
            return

        try:
            result = sync_memories_to_graph(mem_svc.search(), graph_svc)
        except Exception as exc:  # noqa: BLE001 - surfaced, never crashes
            self._stats_label.setText(f"Sync failed: {exc}")
            return

        self._refresh_graph()
        if result.total_changes:
            self._auto_layout()
        self._stats_label.setText(result.summary())

    def _update_empty_state(self) -> None:
        """Explain what this tab is for instead of showing a blank canvas."""
        graph_svc = self._sm.get("Graph")
        if graph_svc is None:
            return
        if graph_svc.get_stats()["total_nodes"]:
            return
        mem_svc = self._sm.get("Memory")
        approved = 0
        if mem_svc is not None:
            try:
                approved = sum(
                    1 for m in mem_svc.search()
                    if m.status.value == "approved" and not m.private
                )
            except Exception:  # noqa: BLE001
                approved = 0
        if approved:
            self._stats_label.setText(
                f"Empty. Press 'Sync from Memories' to build a graph from "
                f"your {approved} approved "
                f"{'memory' if approved == 1 else 'memories'}."
            )
        else:
            self._stats_label.setText(
                "Empty. Add tagged memories in the MEM tab, then press "
                "'Sync from Memories'."
            )

    def _on_graph_cleared(self) -> None:
        self._graph_view.clear_graph()
        if self._graph3d_view is not None:
            self._graph3d_view.clear_graph()
        self._stats_label.setText("Graph cleared")

    def _on_toggle_3d(self, checked: bool) -> None:
        if checked and self._graph3d_view is not None:
            self._view_stack.setCurrentIndex(1)
            self._toggle3d_btn.setText("2D View")
            self._graph3d_view.reset_camera()
        else:
            self._view_stack.setCurrentIndex(0)
            self._toggle3d_btn.setText("3D View")

    def _on_3d_unavailable(self, reason: str) -> None:
        """GL failed at first paint on this machine — fall back to 2D and
        say so, rather than leaving a dead widget in the stack."""
        self._view_stack.setCurrentIndex(0)
        self._toggle3d_btn.setChecked(False)
        self._toggle3d_btn.setText("3D View")
        self._toggle3d_btn.setEnabled(False)
        self._toggle3d_btn.setToolTip(f"3D unavailable: {reason}")
        self._stats_label.setText(
            f"3D view needs working OpenGL ({reason}) — showing 2D."
        )

    def _on_3d_node_selected(self, node_id: str) -> None:
        node = self._last_nodes.get(node_id)
        if node is not None:
            self._stats_label.setText(
                f"Selected: {node.label}  ({node.node_type.value})"
            )
