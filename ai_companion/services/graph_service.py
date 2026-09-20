"""Knowledge graph service — entities and relationships."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.privacy import PrivacyMixin, save_public_only
from ai_companion.models.graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphQuery,
    NodeType,
)


class GraphService(PrivacyMixin, BaseService):
    """Manages a visual knowledge graph connecting memories, chats, files,
    people, projects, and ideas.

    Stores nodes and edges in JSON. Provides search, filtering, and
    layout computation for visualization.
    """

    service_name = "Graph"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nodes_store: Optional[JsonStore] = None
        self._edges_store: Optional[JsonStore] = None

    def start(self) -> None:
        super().start()
        self._nodes_store = JsonStore(self._config.graph.store_path)
        self._nodes_store.load()
        self._edges_store = JsonStore(
            self._config.graph.store_path.replace(".json", "_edges.json")
        )
        self._edges_store.load()
        self.emit_status(
            f"Graph loaded: {self._nodes_store.count()} nodes, "
            f"{self._edges_store.count()} edges"
        )

    def stop(self) -> None:
        self._persist_nodes()
        self._persist_edges()
        super().stop()

    # --- Node operations ---

    def _persist_nodes(self) -> None:
        """Write nodes to disk, excluding anything created in Private mode."""
        if self._nodes_store:
            save_public_only(self._nodes_store, dict(self._nodes_store.items()))

    def _persist_edges(self) -> None:
        """Write edges to disk, excluding anything created in Private mode."""
        if self._edges_store:
            save_public_only(self._edges_store, dict(self._edges_store.items()))

    def add_node(
        self,
        node_type: NodeType,
        label: str,
        description: str = "",
        metadata: Optional[dict] = None,
        x: float = 0.0,
        y: float = 0.0,
    ) -> GraphNode:
        """Add a node to the graph."""
        if self._nodes_store.count() >= self._config.graph.max_nodes:
            raise ValueError("Maximum node limit reached")

        color = self._type_color(node_type)
        node = GraphNode(
            node_type=node_type,
            label=label,
            description=description,
            metadata=metadata or {},
            x=x,
            y=y,
            color=color,
        )
        node.private = self.privacy_stamp()
        self._nodes_store.set(node.id, node.model_dump())
        self._persist_nodes()
        self._signal_bus.graph.node_added.emit(node.id)
        return node

    def update_node(self, node_id: str, **updates: Any) -> Optional[GraphNode]:
        """Update a node's fields."""
        data = self._nodes_store.get(node_id)
        if not data:
            return None
        data.update(updates)
        node = GraphNode(**data)
        self._nodes_store.set(node_id, node.model_dump())
        self._persist_nodes()
        self._signal_bus.graph.node_updated.emit(node_id)
        return node

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connected edges."""
        if not self._nodes_store.delete(node_id):
            return False
        # Remove connected edges
        edges_to_remove = []
        for eid, edata in self._edges_store.items():
            if edata.get("source_id") == node_id or edata.get("target_id") == node_id:
                edges_to_remove.append(eid)
        for eid in edges_to_remove:
            self._edges_store.delete(eid)
        self._persist_nodes()
        self._persist_edges()
        self._signal_bus.graph.node_deleted.emit(node_id)
        return True

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        data = self._nodes_store.get(node_id)
        if data:
            return GraphNode(**data)
        return None

    # --- Edge operations ---

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.RELATED_TO,
        label: str = "",
        weight: float = 1.0,
    ) -> Optional[GraphEdge]:
        """Add an edge between two nodes."""
        # Validate endpoints exist
        if not self._nodes_store.get(source_id):
            self.emit_error(f"Source node {source_id} not found")
            return None
        if not self._nodes_store.get(target_id):
            self.emit_error(f"Target node {target_id} not found")
            return None
        if source_id == target_id:
            self.emit_error("Cannot create self-loop")
            return None

        if self._edges_store.count() >= self._config.graph.max_edges:
            raise ValueError("Maximum edge limit reached")

        # Check for duplicate
        for _, edata in self._edges_store.items():
            if (edata.get("source_id") == source_id and
                    edata.get("target_id") == target_id and
                    edata.get("edge_type") == edge_type.value):
                return None  # Already exists

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=label,
            weight=weight,
        )
        edge.private = self.privacy_stamp()
        self._edges_store.set(edge.id, edge.model_dump())
        self._persist_edges()
        self._signal_bus.graph.edge_added.emit(edge.id)
        return edge

    def delete_edge(self, edge_id: str) -> bool:
        if self._edges_store.delete(edge_id):
            self._persist_edges()
            self._signal_bus.graph.edge_deleted.emit(edge_id)
            return True
        return False

    def get_edges_for_node(self, node_id: str) -> list[GraphEdge]:
        """Get all edges connected to a node."""
        edges = []
        for _, edata in self._edges_store.items():
            if edata.get("source_id") == node_id or edata.get("target_id") == node_id:
                edges.append(GraphEdge(**edata))
        return edges

    # --- Query / Search ---

    def search(self, query: Optional[GraphQuery] = None) -> dict[str, list]:
        """Search the graph. Returns {nodes: [...], edges: [...]}."""
        if query is None:
            query = GraphQuery()

        nodes = []
        for _, ndata in self._nodes_store.items():
            node = GraphNode(**ndata)

            # Filter by type
            if query.node_types and node.node_type not in query.node_types:
                continue
            # Filter by text
            if query.text:
                q = query.text.lower()
                if (q not in node.label.lower() and
                        q not in node.description.lower()):
                    continue
            nodes.append(node)

            if len(nodes) >= query.limit:
                break

        # Get edges between found nodes
        node_ids = {n.id for n in nodes}
        edges = []
        for _, edata in self._edges_store.items():
            edge = GraphEdge(**edata)
            if edge.source_id in node_ids and edge.target_id in node_ids:
                if not query.edge_types or edge.edge_type in query.edge_types:
                    edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    def get_neighbors(
        self, node_id: str, max_depth: int = 1
    ) -> dict[str, list]:
        """Get neighboring nodes up to max_depth hops."""
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()
        frontier = {node_id}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)

                for _, edata in self._edges_store.items():
                    edge = GraphEdge(**edata)
                    if edge.source_id == nid:
                        next_frontier.add(edge.target_id)
                        visited_edges.add(edge.id)
                    elif edge.target_id == nid:
                        next_frontier.add(edge.source_id)
                        visited_edges.add(edge.id)

            frontier = next_frontier - visited_nodes

        nodes = []
        for nid in visited_nodes:
            data = self._nodes_store.get(nid)
            if data:
                nodes.append(GraphNode(**data))

        edges = []
        for eid in visited_edges:
            data = self._edges_store.get(eid)
            if data:
                edges.append(GraphEdge(**data))

        return {"nodes": nodes, "edges": edges}

    def compute_layout(self) -> dict[str, tuple[float, float]]:
        """Compute a force-directed layout for all nodes.

        Returns {node_id: (x, y)}.
        """
        nodes = []
        for _, ndata in self._nodes_store.items():
            nodes.append(GraphNode(**ndata))

        if not nodes:
            return {}

        # Simple force-directed layout
        positions: dict[str, list[float]] = {
            n.id: [n.x if n.x else 0.0, n.y if n.y else 0.0]
            for n in nodes
        }

        # Initialize with circular placement if all at origin
        if all(p[0] == 0 and p[1] == 0 for p in positions.values()):
            for i, node in enumerate(nodes):
                angle = 2 * math.pi * i / len(nodes)
                spread = max(300.0, 60.0 * math.sqrt(len(nodes)))
                positions[node.id] = [
                    spread * math.cos(angle) + 400,
                    spread * math.sin(angle) + 300,
                ]

        edges = []
        for _, edata in self._edges_store.items():
            edges.append(GraphEdge(**edata))

        # Run iterations.
        #
        # k is the target node separation. It was 100 against a 700x500 canvas,
        # which packed labels on top of each other as soon as the graph had
        # more than a handful of nodes. Scale it with node count so a bigger
        # graph spreads out instead of compressing.
        k = max(150.0, 52.0 * math.sqrt(max(len(nodes), 1)))
        iterations = 140
        # Cooling: large early moves settle into small late ones. Without it
        # the simulation oscillates and never converges, which is why nodes
        # were still landing on top of each other.
        temperature = k * 0.55
        for step in range(iterations):
            cooling = temperature * (1.0 - step / iterations)
            disp: dict[str, list[float]] = {n.id: [0.0, 0.0] for n in nodes}
            # Repulsive forces between all pairs
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1:]:
                    dx = positions[n1.id][0] - positions[n2.id][0]
                    dy = positions[n1.id][1] - positions[n2.id][1]
                    dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
                    force = (k * k) / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    disp[n1.id][0] += fx
                    disp[n1.id][1] += fy
                    disp[n2.id][0] -= fx
                    disp[n2.id][1] -= fy

            # Attractive forces along edges
            for edge in edges:
                if edge.source_id in positions and edge.target_id in positions:
                    dx = positions[edge.source_id][0] - positions[edge.target_id][0]
                    dy = positions[edge.source_id][1] - positions[edge.target_id][1]
                    dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
                    force = (dist * dist) / k
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    disp[edge.source_id][0] -= fx
                    disp[edge.source_id][1] -= fy
                    disp[edge.target_id][0] += fx
                    disp[edge.target_id][1] += fy

            # Apply displacement, capped by the cooling schedule.
            for node in nodes:
                dx, dy = disp[node.id]
                length = max(math.sqrt(dx * dx + dy * dy), 0.01)
                limit = min(length, cooling)
                positions[node.id][0] += (dx / length) * limit
                positions[node.id][1] += (dy / length) * limit

        # Order matters. Clamping to a box AFTER separating nodes piles them
        # onto the boundary - that is what produced coincident nodes at the
        # corners. Instead: normalise the cloud into the canvas by scaling,
        # THEN separate, so de-overlap is the last word.
        span = max(1.0, math.sqrt(max(len(nodes), 1)))
        width = max(820.0, 230.0 * span)
        height = max(600.0, 165.0 * span)

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 1.0)
        range_y = max(max_y - min_y, 1.0)
        for pid, p in positions.items():
            p[0] = 60.0 + (p[0] - min_x) / range_x * width
            p[1] = 60.0 + (p[1] - min_y) / range_y * height

        # Explicit separation. Force-directed layout minimises energy, not
        # collisions; these rounds are what actually stop labels overlapping.
        min_gap = 130.0
        for _ in range(120):
            moved = False
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1:]:
                    dx = positions[n1.id][0] - positions[n2.id][0]
                    dy = positions[n1.id][1] - positions[n2.id][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_gap:
                        if dist < 0.01:
                            # Perfectly coincident: nudge deterministically so
                            # the pair has a direction to separate along.
                            angle = (i * 2.399) % (2 * math.pi)
                            dx, dy = math.cos(angle), math.sin(angle)
                            dist = 1.0
                        push = (min_gap - dist) / 2.0 + 0.5
                        ux, uy = dx / dist, dy / dist
                        positions[n1.id][0] += ux * push
                        positions[n1.id][1] += uy * push
                        positions[n2.id][0] -= ux * push
                        positions[n2.id][1] -= uy * push
                        moved = True
            if not moved:
                break

        # Shift everything positive; the view zooms to fit, so there is no
        # fixed canvas to clip against.
        min_x = min(p[0] for p in positions.values())
        min_y = min(p[1] for p in positions.values())
        for p in positions.values():
            p[0] += 60.0 - min_x
            p[1] += 60.0 - min_y

        return {nid: tuple(pos) for nid, pos in positions.items()}

    def clear(self) -> None:
        """Clear the entire graph."""
        self._nodes_store.clear()
        self._edges_store.clear()
        self._persist_nodes()
        self._persist_edges()
        self._signal_bus.graph.graph_cleared.emit()

    def get_stats(self) -> dict:
        """Get graph statistics."""
        node_types: dict[str, int] = {}
        for _, ndata in self._nodes_store.items():
            t = ndata.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        return {
            "total_nodes": self._nodes_store.count(),
            "total_edges": self._edges_store.count(),
            "node_types": node_types,
        }

    @staticmethod
    def _type_color(node_type: NodeType) -> str:
        """Return a color hex for each node type."""
        # Node hues are a HUD-tuned analogous ramp: violets and cyans that sit
        # inside the interface palette, rather than the saturated primaries
        # this used before (they read as clip-art against a dark HUD).
        colors = {
            NodeType.MEMORY: "#a855f7",
            NodeType.CHAT: "#8b5cf6",
            NodeType.FILE: "#22d3ee",
            NodeType.PERSON: "#f472b6",
            NodeType.PROJECT: "#c084fc",
            NodeType.IDEA: "#2dd4bf",
            NodeType.TAG: "#6b5a99",
            NodeType.TOPIC: "#818cf8",
        }
        return colors.get(node_type, "#a855f7")
