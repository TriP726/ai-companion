"""Memory service with owner-reviewed long-term memories."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.privacy import PrivacyMixin, save_public_only
from ai_companion.models.memory import Memory, MemoryFilter, MemorySource, MemoryStatus


class MemoryService(PrivacyMixin, BaseService):
    """Manages long-term memories with owner review.

    Memories can be:
    - Manually created by the user
    - Extracted from conversations (require approval)
    - Imported from external sources

    Each memory has:
    - Confidence score (0-1)
    - Provenance tracking
    - Pinning for important memories
    - Full CRUD with approval workflow
    """

    service_name = "Memory"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._store: Optional[JsonStore] = None

    def start(self) -> None:
        super().start()
        self._store = JsonStore(self._config.memory.store_path)
        self._store.load()
        self.emit_status(f"Memory store loaded: {self._store.count()} memories")

    def stop(self) -> None:
        self._persist()
        super().stop()

    def _persist(self) -> None:
        """Write to disk, excluding anything created in Private mode."""
        if not self._store:
            return
        save_public_only(self._store, dict(self._store.items()))

    def add_memory(
        self,
        content: str,
        summary: str = "",
        tags: Optional[list[str]] = None,
        confidence: float = 0.5,
        source: MemorySource = MemorySource.USER_EXPLICIT,
        provenance: str = "",
        auto_approve: bool = False,
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[str] = None,
    ) -> Memory:
        """Add a new memory. Returns the Memory object."""
        # Validate inputs
        if not content.strip():
            raise ValueError("Memory content cannot be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {confidence}")

        status = MemoryStatus.APPROVED if auto_approve else MemoryStatus.PENDING
        if self._config.memory.require_approval and source != MemorySource.USER_EXPLICIT:
            status = MemoryStatus.PENDING

        mem = Memory(
            content=content,
            summary=summary or content[:100],
            tags=tags or [],
            confidence=confidence,
            source=source,
            provenance=provenance,
            status=status,
            source_conversation_id=source_conversation_id,
            source_message_id=source_message_id,
            private=self.privacy_stamp(),
        )

        self._store.set(mem.id, mem.model_dump())
        self._persist()

        self._signal_bus.memory.memory_added.emit(mem.id)
        if status == MemoryStatus.PENDING:
            self._signal_bus.memory.candidate_extracted.emit(mem.id)

        return mem

    def update_memory(self, memory_id: str, **updates: Any) -> Optional[Memory]:
        """Update fields on an existing memory."""
        data = self._store.get(memory_id)
        if not data:
            self.emit_error(f"Memory {memory_id} not found")
            return None

        # Validate allowed fields
        allowed = {
            "content", "summary", "tags", "confidence", "pinned", "provenance"
        }
        for key in updates:
            if key not in allowed:
                raise ValueError(f"Cannot update field: {key}")

        data.update(updates)
        data["modified"] = datetime.now(timezone.utc).isoformat()

        mem = Memory(**data)
        self._store.set(memory_id, mem.model_dump())
        self._persist()

        self._signal_bus.memory.memory_updated.emit(memory_id)
        return mem

    def approve_memory(self, memory_id: str) -> Optional[Memory]:
        """Approve a pending memory."""
        data = self._store.get(memory_id)
        if not data:
            return None
        data["status"] = MemoryStatus.APPROVED.value
        data["modified"] = datetime.now(timezone.utc).isoformat()
        self._store.set(memory_id, data)
        self._persist()
        self._signal_bus.memory.memory_approved.emit(memory_id)
        return Memory(**data)

    def reject_memory(self, memory_id: str) -> Optional[Memory]:
        """Reject a pending memory."""
        data = self._store.get(memory_id)
        if not data:
            return None
        data["status"] = MemoryStatus.REJECTED.value
        data["modified"] = datetime.now(timezone.utc).isoformat()
        self._store.set(memory_id, data)
        self._persist()
        self._signal_bus.memory.memory_rejected.emit(memory_id)
        return Memory(**data)

    def delete_memory(self, memory_id: str) -> bool:
        """Permanently delete a memory."""
        if self._store.delete(memory_id):
            self._persist()
            self._signal_bus.memory.memory_deleted.emit(memory_id)
            return True
        return False

    def toggle_pin(self, memory_id: str) -> Optional[Memory]:
        """Toggle pin status of a memory."""
        data = self._store.get(memory_id)
        if not data:
            return None
        data["pinned"] = not data.get("pinned", False)
        data["modified"] = datetime.now(timezone.utc).isoformat()
        self._store.set(memory_id, data)
        self._persist()
        return Memory(**data)

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a single memory by ID."""
        data = self._store.get(memory_id)
        if data:
            return Memory(**data)
        return None

    def search(self, filters: Optional[MemoryFilter] = None) -> list[Memory]:
        """Search memories with optional filters."""
        if filters is None:
            filters = MemoryFilter()

        results = []
        for key, data in self._store.items():
            mem = Memory(**data)

            # Apply filters
            if filters.status and mem.status != filters.status:
                continue
            if filters.pinned is not None and mem.pinned != filters.pinned:
                continue
            if mem.confidence < filters.min_confidence:
                continue
            if mem.confidence > filters.max_confidence:
                continue
            if filters.tags and not any(t in mem.tags for t in filters.tags):
                continue
            if filters.query:
                query_lower = filters.query.lower()
                if (query_lower not in mem.content.lower() and
                        query_lower not in mem.summary.lower()):
                    continue

            results.append(mem)

        # Sort: pinned first, then by confidence
        results.sort(key=lambda m: (not m.pinned, -m.confidence))
        return results

    def get_pending(self) -> list[Memory]:
        """Get all memories pending approval."""
        return self.search(MemoryFilter(status=MemoryStatus.PENDING))

    def extract_candidates(
        self, text: str, conversation_id: str = "", message_id: str = ""
    ) -> list[Memory]:
        """Propose durable facts found in `text` as PENDING memories.

        Candidates are proposals, never facts. They are stored PENDING and are
        excluded from model injection until the owner approves them - an
        extractor able to write straight into the model's context would let a
        misread sentence become something the assistant believes.

        Private mode is respected: nothing is extracted, because the resulting
        memory would be a durable record of a conversation the user asked not
        to keep.
        """
        from ai_companion.services.memory_extraction import (
            extract_candidates as _extract,
        )

        if self.is_private_mode():
            return []

        try:
            proposals = _extract(text, self.search())
        except Exception as exc:  # noqa: BLE001 - never break a send
            self.emit_error(f"Memory extraction failed: {exc}")
            return []

        created: list[Memory] = []
        for proposal in proposals:
            try:
                memory = self.add_memory(
                    content=proposal.content,
                    summary=proposal.content[:100],
                    tags=proposal.tags,
                    source=MemorySource.EXTRACTED,
                    provenance=f"Auto-extracted: {proposal.reason}",
                    confidence=proposal.confidence,
                    auto_approve=False,
                    source_conversation_id=conversation_id,
                    source_message_id=message_id,
                )
            except Exception:  # noqa: BLE001
                continue
            created.append(memory)
        return created

    def export_all(self) -> list[dict]:
        """Export all memories as a list of dicts."""
        return [Memory(**data).model_dump() for _, data in self._store.items()]

    def import_memories(self, memories: list[dict]) -> int:
        """Import memories from a list of dicts. Returns count imported."""
        count = 0
        for data in memories:
            try:
                mem = Memory(**data)
                self._store.set(mem.id, mem.model_dump())
                count += 1
            except Exception:
                self.emit_error(f"Failed to import memory: {data.get('id', '?')}")
        self._persist()
        return count
