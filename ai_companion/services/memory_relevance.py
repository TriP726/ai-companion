"""Score memories against the user's question.

THE PROBLEM THIS SOLVES
-----------------------
Injection used to send the same top-N memories on every message, ranked by
pinned-then-confidence and nothing else. Measured against the 1200-char
budget:

     5 memories ->  5 reach the model (100%)
    25 memories -> 11 reach the model  (44%)
   150 memories -> 11 reach the model   (7%)

Past about a dozen memories the store silently stops scaling: you keep adding
facts, they look saved, and the model never sees them. Nothing reports it.

Relevance fixes the *selection*, not the budget. The budget is a hard limit of
the context window and is correct; what was wrong is spending it on arbitrary
memories rather than the ones that bear on the question.

APPROACH, AND WHY NOT EMBEDDINGS
--------------------------------
Lexical scoring: term overlap between the question and each memory, with tags
weighted heavily because tags are curated labels rather than incidental words.

Embeddings would be more accurate but need a model round-trip per memory. On a
CPU-only machine that is seconds of latency before generation even starts,
which is the wrong trade for a local assistant. This is a deliberate accuracy/
latency choice, not an oversight.

PINNED MEMORIES ALWAYS WIN
--------------------------
Pinning is the owner explicitly saying "this always matters". A relevance
score must never override that, or pinning stops meaning anything.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# Words too common to carry signal. Deliberately short: an aggressive stop list
# throws away terms that are meaningful in a technical conversation.
STOP_WORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its me my of on or our should so than that the their
them then there these they this to was we were what when where which who why
will with would you your
""".split())

# A tag is a curated label; prose is incidental. The gap has to be wide
# because summary and content overlap heavily - the same word usually scores
# in both, so their combined weight must still sit clearly below a tag match.
TAG_WEIGHT = 6.0
SUMMARY_WEIGHT = 1.5
CONTENT_WEIGHT = 1.0
PHRASE_BONUS = 2.5      # whole query phrase appearing verbatim
CONFIDENCE_WEIGHT = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> set[str]:
    """Lowercase word set with stop words and 1-char noise removed."""
    if not text:
        return set()
    words = _TOKEN_RE.findall(text.lower())
    return {w for w in words if len(w) > 1 and w not in STOP_WORDS}


def _overlap(query_terms: set[str], text: str) -> int:
    if not query_terms or not text:
        return 0
    return len(query_terms & tokenize(text))


def score_memory(memory: Any, query: str) -> float:
    """Relevance of one memory to one question. Higher is better.

    Returns 0.0 when nothing matches, which lets the caller distinguish
    "related" from "merely available".
    """
    terms = tokenize(query)
    if not terms:
        return 0.0

    tags = [str(t).lower() for t in (getattr(memory, "tags", None) or [])]
    summary = str(getattr(memory, "summary", "") or "")
    content = str(getattr(memory, "content", "") or "")

    score = 0.0

    # Tags: exact token match, plus substring so "discord" hits "discord-bot".
    tag_terms: set[str] = set()
    for tag in tags:
        tag_terms |= tokenize(tag)
    score += TAG_WEIGHT * len(terms & tag_terms)
    for term in terms:
        for tag in tags:
            if term != tag and term in tag:
                score += TAG_WEIGHT * 0.5
                break

    score += SUMMARY_WEIGHT * _overlap(terms, summary)
    score += CONTENT_WEIGHT * _overlap(terms, content)

    # A verbatim phrase is much stronger evidence than scattered words.
    cleaned = " ".join(query.lower().split())
    if len(cleaned) > 8:
        haystack = f"{summary} {content}".lower()
        if cleaned in haystack:
            score += PHRASE_BONUS

    if score > 0:
        score += CONFIDENCE_WEIGHT * float(getattr(memory, "confidence", 0.0))

    return score


def rank_memories(memories: Iterable, query: str) -> list[tuple[Any, float]]:
    """Rank approved memories by relevance, best first.

    Pinned memories are always ordered ahead of unpinned ones regardless of
    score. Ties fall back to confidence then recency, so ordering is stable
    and never depends on dict iteration order.
    """
    scored: list[tuple[Any, float]] = []
    for memory in memories:
        status = getattr(getattr(memory, "status", None), "value", "")
        if status != "approved":
            continue
        if not str(getattr(memory, "content", "") or "").strip():
            continue
        scored.append((memory, score_memory(memory, query)))

    scored.sort(
        key=lambda pair: (
            not bool(getattr(pair[0], "pinned", False)),
            -pair[1],
            -float(getattr(pair[0], "confidence", 0.0)),
            str(getattr(pair[0], "modified", "")),
        )
    )
    return scored


def expand_query_with_graph(
    query: str,
    graph_service: Any,
    max_tags: int = 4,
) -> str:
    """Widen a query using tags from the graph.

    The graph already models which tags co-occur. If the question mentions a
    tag, its neighbouring tags are appended so a memory tagged only with the
    sibling still surfaces - asking about "discord" can reach memories tagged
    "bots".

    Returns the query unchanged on any failure; this is an enhancement, and it
    must never be able to break a send.
    """
    if graph_service is None or not query.strip():
        return query

    try:
        nodes = graph_service.search().get("nodes", [])
    except Exception:  # noqa: BLE001 - enhancement only, never fatal
        return query

    terms = tokenize(query)
    if not terms:
        return query

    tag_nodes = {}
    for node in nodes:
        if getattr(getattr(node, "node_type", None), "value", "") != "tag":
            continue
        name = str(node.label).lstrip("#").lower()
        if name:
            tag_nodes[name] = node

    matched = [node for name, node in tag_nodes.items() if name in terms]
    if not matched:
        return query

    extra: list[str] = []
    for node in matched:
        try:
            edges = graph_service.get_edges_for_node(node.id)
        except Exception:  # noqa: BLE001
            continue
        for edge in edges:
            other_id = (
                edge.target_id if edge.source_id == node.id else edge.source_id
            )
            try:
                neighbour = graph_service.get_node(other_id)
            except Exception:  # noqa: BLE001
                continue
            if neighbour is None:
                continue
            # Memory nodes lead back to their other tags.
            try:
                mem_edges = graph_service.get_edges_for_node(neighbour.id)
            except Exception:  # noqa: BLE001
                continue
            for mem_edge in mem_edges:
                tag_id = (
                    mem_edge.target_id
                    if mem_edge.source_id == neighbour.id
                    else mem_edge.source_id
                )
                tag_node = tag_nodes.get("")
                for name, candidate in tag_nodes.items():
                    if candidate.id == tag_id and name not in terms:
                        if name not in extra:
                            extra.append(name)
                        break

    if not extra:
        return query
    return query + " " + " ".join(extra[:max_tags])
