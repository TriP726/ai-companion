"""Mind Map service: derive concepts, cache overviews, feed the model.

Holds the structural map in memory (cheap to rebuild) and persists only the
expensive part - the LLM-written overviews - keyed by a fingerprint of the
memories that produced them.

PRIVACY
-------
Overviews are written to disk, so nothing derived from a private memory may
enter them. `build()` excludes private memories, and the cache is not written
at all while the vault is private.
"""
from __future__ import annotations

from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.privacy import PrivacyMixin
from ai_companion.services.mind_map import MindMap, build_mind_map, layout_mind_map
from ai_companion.services.mind_map_synthesis import (
    SynthesisResult,
    fingerprint,
    synthesise_overview,
)


class MindMapService(PrivacyMixin, BaseService):
    """Derives and caches the Mind Map."""

    service_name = "MindMap"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._store: Optional[JsonStore] = None
        self._map: MindMap = MindMap()

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        super().start()
        self._store = JsonStore(self._config.mind_map.store_path)
        self._store.load()
        self.emit_status(
            f"Mind Map: {self._store.count()} cached overviews"
        )

    def stop(self) -> None:
        self._persist()
        super().stop()

    def _persist(self) -> None:
        if not self._store:
            return
        if self.is_private_mode():
            # Overviews summarise memories; in private mode nothing derived
            # from this session may reach disk.
            return
        self._store.save()

    # -- structure ----------------------------------------------------

    def build(self) -> MindMap:
        """Rebuild the structural map from current memories.

        Cheap and deterministic - no model call. Cached overviews are
        reattached, and any whose underlying memories changed are marked
        stale rather than silently served.
        """
        memories: list = []
        manager = getattr(self, "_service_manager", None)
        if manager is not None:
            try:
                memory_service = manager.get("Memory")
                if memory_service is not None:
                    memories = list(memory_service.search())
            except Exception as exc:  # noqa: BLE001
                self.emit_error(f"Mind Map could not read memories: {exc}")
                memories = []

        self._map = build_mind_map(memories)

        if self._store is not None:
            for concept in self._map.concepts.values():
                cached = self._store.get(concept.key)
                if not isinstance(cached, dict):
                    continue
                concept.overview = str(cached.get("overview", ""))
                concept.overview_stale = (
                    cached.get("fingerprint") != fingerprint(concept.memory_ids)
                )

        self._signal_bus.graph.node_added.emit("mind_map")
        return self._map

    @property
    def map(self) -> MindMap:
        return self._map

    def layout(self) -> dict:
        return layout_mind_map(self._map)

    def memories_for(self, key: str) -> list:
        """The memories behind one concept, in store order."""
        concept = self._map.concepts.get(key)
        if concept is None:
            return []
        manager = getattr(self, "_service_manager", None)
        if manager is None:
            return []
        try:
            memory_service = manager.get("Memory")
            if memory_service is None:
                return []
            by_id = {str(m.id): m for m in memory_service.search()}
        except Exception:  # noqa: BLE001
            return []
        return [by_id[i] for i in concept.memory_ids if i in by_id]

    # -- synthesis ----------------------------------------------------

    def synthesise(self, key: str) -> SynthesisResult:
        """Write one concept's overview. Blocking, seconds on CPU."""
        concept = self._map.concepts.get(key)
        if concept is None:
            return SynthesisResult(False, error="Unknown concept.")

        manager = getattr(self, "_service_manager", None)
        llm = manager.get("LLM") if manager is not None else None

        result = synthesise_overview(
            concept.label, self.memories_for(key), llm
        )
        if not result.ok:
            return result

        concept.overview = result.text
        concept.overview_stale = False
        if self._store is not None:
            self._store.set(key, {
                "label": concept.label,
                "overview": result.text,
                "fingerprint": fingerprint(concept.memory_ids),
            })
            self._persist()
        return result

    # -- injection ----------------------------------------------------

    def overviews_for_query(self, query: str, limit: int = 3) -> list:
        """Concept overviews most relevant to `query`, as (label, text).

        Only fresh overviews are offered. A stale one summarises memories that
        have since changed, and feeding the model a quietly outdated summary
        is worse than feeding it none.
        """
        if not self._config.mind_map.inject_overviews:
            return []
        if not query.strip() or not self._map.concepts:
            return []

        from ai_companion.services.memory_relevance import tokenize

        terms = tokenize(query)
        if not terms:
            return []

        scored: list[tuple[float, Any]] = []
        for concept in self._map.concepts.values():
            if not concept.overview or concept.overview_stale:
                continue
            label_terms = tokenize(concept.label)
            score = 3.0 * len(terms & label_terms)
            score += len(terms & tokenize(concept.overview))
            if score > 0:
                # Busier concepts break ties: they describe more of the user.
                scored.append((score + concept.weight * 0.1, concept))

        scored.sort(key=lambda pair: (-pair[0], pair[1].label))
        capped = min(limit, self._config.mind_map.max_injected)
        return [(c.label, c.overview) for _, c in scored[:capped]]
