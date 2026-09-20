"""Derive the knowledge graph from memories.

WHY DERIVED RATHER THAN HAND-MAINTAINED
---------------------------------------
The graph was a second store the user had to fill in by hand, duplicating
facts already curated in the Memory tab. Two stores describing the same thing
drift apart immediately, and the hand-maintained one loses.

So the graph is now a *projection* of memories:

    memory  ->  one MEMORY node
    tag     ->  one TAG node, shared by every memory carrying it
    edge    ->  memory TAGGED_WITH tag

Tags are the join. Two memories tagged `discord` both connect to the same
`discord` node, and the shape of your knowledge falls out of labels you were
already writing. Nothing is invented.

IDEMPOTENT BY CONSTRUCTION
--------------------------
Sync can run any number of times and converges to the same graph. Derived
nodes are marked in `metadata` so a full rebuild can delete exactly what it
created and never touch a node the user added by hand.

PRIVACY
-------
Private memories are skipped entirely. A memory the owner chose not to write
to disk must not reappear as a graph node that *is* written to disk - that
would be a privacy hole disguised as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ai_companion.models.graph import EdgeType, NodeType

# Marks a node this module created, so a rebuild can clean up after itself
# without deleting anything the user made.
DERIVED_KEY = "derived_from"
DERIVED_MEMORY = "memory"
DERIVED_TAG = "tag"

MAX_LABEL = 48


@dataclass
class SyncResult:
    """What a sync pass changed."""

    memories_added: int = 0
    tags_added: int = 0
    edges_added: int = 0
    removed: int = 0
    skipped_private: int = 0

    @property
    def total_changes(self) -> int:
        return (
            self.memories_added
            + self.tags_added
            + self.edges_added
            + self.removed
        )

    def summary(self) -> str:
        if self.total_changes == 0 and self.skipped_private == 0:
            return "Graph is up to date."
        bits = []
        if self.memories_added:
            bits.append(f"{self.memories_added} memories")
        if self.tags_added:
            bits.append(f"{self.tags_added} tags")
        if self.edges_added:
            bits.append(f"{self.edges_added} links")
        if self.removed:
            bits.append(f"{self.removed} removed")
        text = "Graph updated: " + ", ".join(bits) if bits else "Graph updated."
        if self.skipped_private:
            text += (
                f" ({self.skipped_private} private "
                f"{'memory' if self.skipped_private == 1 else 'memories'} "
                "not added)"
            )
        return text


def _label_for(memory: Any) -> str:
    """Short, readable node label for a memory."""
    text = (getattr(memory, "summary", "") or "").strip()
    if not text:
        text = (getattr(memory, "content", "") or "").strip()
    text = " ".join(text.split())
    if len(text) > MAX_LABEL:
        text = text[: MAX_LABEL - 1].rstrip() + "\u2026"
    return text or "(empty memory)"


def _is_derived(node: Any, kind: Optional[str] = None) -> bool:
    meta = getattr(node, "metadata", None) or {}
    value = meta.get(DERIVED_KEY)
    if value is None:
        return False
    return True if kind is None else value == kind


def _source_id(node: Any) -> str:
    meta = getattr(node, "metadata", None) or {}
    return str(meta.get("source_id", ""))


def sync_memories_to_graph(
    memories: Iterable,
    graph_service: Any,
    include_private: bool = False,
) -> SyncResult:
    """Project memories (and their tags) into the graph.

    Only APPROVED memories are projected. Pending ones are unreviewed guesses
    and rejected ones were refused; graphing either would contradict the
    review workflow the Memory tab exists to enforce.
    """
    result = SyncResult()

    wanted: list = []
    for memory in memories:
        status = getattr(getattr(memory, "status", None), "value", "")
        if status != "approved":
            continue
        if getattr(memory, "private", False) and not include_private:
            result.skipped_private += 1
            continue
        wanted.append(memory)

    existing_nodes = _all_nodes(graph_service)

    # Index what we previously derived, so this pass updates instead of
    # duplicating.
    memory_nodes = {
        _source_id(n): n
        for n in existing_nodes
        if _is_derived(n, DERIVED_MEMORY)
    }
    tag_nodes = {
        _source_id(n): n for n in existing_nodes if _is_derived(n, DERIVED_TAG)
    }

    live_memory_ids = {getattr(m, "id", "") for m in wanted}
    live_tags: set[str] = set()

    # --- memory nodes ---
    for memory in wanted:
        mem_id = getattr(memory, "id", "")
        label = _label_for(memory)
        node = memory_nodes.get(mem_id)
        if node is None:
            node = graph_service.add_node(
                NodeType.MEMORY,
                label,
                description=(getattr(memory, "content", "") or "")[:500],
                metadata={DERIVED_KEY: DERIVED_MEMORY, "source_id": mem_id},
            )
            if node is None:
                continue
            memory_nodes[mem_id] = node
            result.memories_added += 1
        elif node.label != label:
            graph_service.update_node(node.id, label=label)

    # --- tag nodes + links ---
    for memory in wanted:
        mem_id = getattr(memory, "id", "")
        mem_node = memory_nodes.get(mem_id)
        if mem_node is None:
            continue
        for raw_tag in getattr(memory, "tags", []) or []:
            tag = str(raw_tag).strip().lower()
            if not tag:
                continue
            live_tags.add(tag)
            tag_node = tag_nodes.get(tag)
            if tag_node is None:
                tag_node = graph_service.add_node(
                    NodeType.TAG,
                    f"#{tag}",
                    metadata={DERIVED_KEY: DERIVED_TAG, "source_id": tag},
                )
                if tag_node is None:
                    continue
                tag_nodes[tag] = tag_node
                result.tags_added += 1

            if not _edge_exists(graph_service, mem_node.id, tag_node.id):
                edge = graph_service.add_edge(
                    mem_node.id,
                    tag_node.id,
                    EdgeType.TAGGED_WITH,
                    label=tag,
                )
                if edge is not None:
                    result.edges_added += 1

    # --- remove derived nodes whose source is gone ---
    for mem_id, node in list(memory_nodes.items()):
        if mem_id not in live_memory_ids:
            if graph_service.delete_node(node.id):
                result.removed += 1
    for tag, node in list(tag_nodes.items()):
        if tag not in live_tags:
            if graph_service.delete_node(node.id):
                result.removed += 1

    return result


def _all_nodes(graph_service: Any) -> list:
    try:
        found = graph_service.search()
    except Exception:  # noqa: BLE001 - never let sync break the caller
        return []
    return list(found.get("nodes", []))


def _edge_exists(graph_service: Any, source_id: str, target_id: str) -> bool:
    try:
        edges = graph_service.get_edges_for_node(source_id)
    except Exception:  # noqa: BLE001
        return False
    for edge in edges:
        if {edge.source_id, edge.target_id} == {source_id, target_id}:
            return True
    return False
