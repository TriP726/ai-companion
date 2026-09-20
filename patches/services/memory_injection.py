"""Select approved memories and render them into the system prompt.

WHY THIS IS A SEPARATE, PURE MODULE
-----------------------------------
Selection is the part that can silently go wrong: inject too much and the
context window is eaten by memories, leaving no room for the actual
conversation; inject the wrong ones and the model is confidently misinformed.
Keeping it free of Qt and of the store makes every rule directly testable.

THE BUDGET PROBLEM
------------------
The user runs a 4096-token context. Memories compete with the system prompt,
the conversation history, attachments, and the reply itself. So injection is
capped by characters, not by memory count, and the cap defaults to a small
fraction of the window.

Ordering matters because the budget truncates:
  1. Pinned memories first - the owner explicitly marked these as important.
  2. Then by confidence, highest first.
  3. Then most recently modified.

Anything that does not fit is dropped silently rather than truncated
mid-sentence: half a fact is worse than no fact.

WHAT IS EXCLUDED
----------------
Only APPROVED memories are injected. Pending ones are unreviewed guesses and
rejected ones were explicitly refused; feeding either to the model would make
the review workflow meaningless.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

# ~4 characters per token is the usual rough English estimate. At 4096 tokens
# total, 1200 characters is roughly 300 tokens: enough for a dozen short facts
# while leaving the conversation the overwhelming majority of the window.
DEFAULT_CHAR_BUDGET = 1200

HEADER = (
    "Known facts about the user, provided by the user and approved by them. "
    "Treat these as true. Do not mention this list unless asked."
)


def select_memories(
    memories: Iterable,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    query: str = "",
) -> list:
    """Pick which approved memories fit the budget, best first.

    When `query` is given, memories are ranked by relevance to it rather than
    by confidence alone. Without this the same top-N was sent on every
    message, so a store of 150 memories delivered the same 11 regardless of
    the question - the rest were dead weight the user could not see.

    Pinned memories still sort first: pinning is the owner saying "this always
    matters", and a relevance score must not override it.
    """
    if query.strip():
        from ai_companion.services.memory_relevance import rank_memories

        ranked = rank_memories(memories, query)

        # Drop memories that score zero against the question. They are not
        # merely lower-priority, they are unrelated - and every one of them
        # spends context the conversation itself needs. Pinned memories are
        # exempt: the owner said those always matter.
        relevant = [
            m for m, score in ranked
            if score > 0 or bool(getattr(m, "pinned", False))
        ]

        # If nothing matched, fall back to the confidence ordering rather than
        # sending an empty block: a general question still benefits from
        # knowing who the user is.
        approved = relevant if relevant else [m for m, _ in ranked]
    else:
        approved = [
            m
            for m in memories
            if getattr(getattr(m, "status", None), "value", "") == "approved"
            and str(getattr(m, "content", "")).strip()
        ]
        approved.sort(
            key=lambda m: (
                not bool(getattr(m, "pinned", False)),
                -float(getattr(m, "confidence", 0.0)),
                str(getattr(m, "modified", "")),
            )
        )

    chosen: list = []
    used = 0
    for mem in approved:
        line = _render_line(mem)
        cost = len(line) + 1
        if used + cost > char_budget:
            # Keep scanning: a later, shorter memory may still fit.
            continue
        chosen.append(mem)
        used += cost
    return chosen


def _render_line(mem) -> str:
    content = str(getattr(mem, "content", "")).strip().replace("\n", " ")
    return f"- {content}"


def build_memory_block(
    memories: Iterable,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    query: str = "",
) -> str:
    """Render selected memories as a system-prompt section.

    Returns "" when nothing qualifies, so the caller can skip the section
    entirely rather than emitting a confusing empty header.
    """
    chosen = select_memories(memories, char_budget, query)
    if not chosen:
        return ""
    lines = "\n".join(_render_line(m) for m in chosen)
    return f"{HEADER}\n{lines}"


MINDMAP_HEADER = (
    "Big-picture context. These overviews connect several remembered facts; "
    "use them to understand how things relate."
)

# Mind Map overviews get their own slice of the budget so they cannot crowd
# out the individual memories. Nomi's own framing: the map assists memories,
# it does not replace them.
MINDMAP_BUDGET_SHARE = 0.4


def build_mindmap_block(overviews: list[tuple[str, str]],
                        char_budget: int) -> str:
    """Render concept overviews for the prompt.

    `overviews` is [(label, text)], already ordered by relevance.
    """
    if not overviews or char_budget <= 0:
        return ""
    lines: list[str] = []
    used = 0
    for label, text in overviews:
        entry = f"- {label}: {text}".strip()
        if used + len(entry) + 1 > char_budget:
            continue
        lines.append(entry)
        used += len(entry) + 1
    if not lines:
        return ""
    return MINDMAP_HEADER + "\n" + "\n".join(lines)


def compose_system_prompt(
    base_prompt: str,
    memories: Iterable,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    query: str = "",
    overviews: Optional[list] = None,
) -> str:
    """Combine the user's system prompt with the memory block.

    The base prompt comes FIRST so the user's behavioural instructions are not
    buried under a list of facts, and the memories read as context supplied to
    those instructions.
    """
    overviews = overviews or []
    map_budget = int(char_budget * MINDMAP_BUDGET_SHARE) if overviews else 0
    map_block = build_mindmap_block(overviews, map_budget)

    # Whatever the Mind Map did not use stays available to raw memories.
    memory_budget = char_budget - len(map_block)
    block = build_memory_block(memories, max(memory_budget, 0), query)

    parts = [p for p in ((base_prompt or "").strip(), map_block, block) if p]
    return "\n\n".join(parts)
