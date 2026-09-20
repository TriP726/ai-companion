"""Write a Mind Map entry's overview using the local model.

WHY ON DEMAND
-------------
Synthesising every concept in the background would cost roughly ten minutes of
generation for thirty concepts on a CPU-only machine, and would have to rerun
as memories accumulate. That competes directly with the chat the user is
waiting on, and an app that is mysteriously slow is worse than one that asks.

So: the structural Mind Map is instant, and the written overview is generated
for one concept at a time, when the user asks for it.

CACHING
-------
An overview is cached against a fingerprint of the memories that produced it.
Add a memory to the concept and the fingerprint changes, marking the overview
stale - so the UI can show that it is out of date instead of quietly serving
a summary of facts that have since changed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# Deliberately small. The point is a couple of connective sentences, not an
# essay, and every token costs real seconds on this hardware.
MAX_OVERVIEW_TOKENS = 130

PROMPT_TEMPLATE = """Write a 2-3 sentence overview of "{label}" based only on \
the facts below. Connect them into a single coherent picture rather than \
listing them. Do not invent anything that is not stated. Do not mention that \
you were given facts.

Facts:
{facts}

Overview of {label}:"""


@dataclass
class SynthesisResult:
    """Outcome of one synthesis attempt."""

    ok: bool
    text: str = ""
    error: str = ""


def fingerprint(memory_ids: Iterable[str]) -> str:
    """Stable id for the exact set of memories behind an overview."""
    joined = "|".join(sorted(str(m) for m in memory_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def build_prompt(label: str, memories: list, max_facts: int = 12) -> str:
    """Prompt for one concept.

    Capped at `max_facts` because prompt length is the main driver of
    generation time here, and a dozen facts is already more than enough for a
    three-sentence overview.
    """
    lines = []
    for memory in memories[:max_facts]:
        text = str(getattr(memory, "content", "") or "").strip()
        if text:
            lines.append(f"- {text}")
    return PROMPT_TEMPLATE.format(label=label, facts="\n".join(lines))


def synthesise_overview(
    label: str,
    memories: list,
    llm_service: Any,
    max_tokens: int = MAX_OVERVIEW_TOKENS,
) -> SynthesisResult:
    """Generate one concept overview synchronously.

    Returns a result rather than raising: a failed summary must never break
    the Mind Map, which is useful without any overviews at all.
    """
    if not memories:
        return SynthesisResult(False, error="No memories for this concept.")

    if llm_service is None or not getattr(llm_service, "model_loaded", False):
        return SynthesisResult(
            False,
            error="No model loaded. Load a model in Settings to write "
                  "overviews.",
        )

    model = getattr(llm_service, "_model", None)
    if model is None:
        return SynthesisResult(False, error="Model unavailable.")

    prompt = build_prompt(label, memories)

    try:
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return SynthesisResult(False, error=f"Generation failed: {exc}")

    try:
        text = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        return SynthesisResult(False, error=f"Unexpected model output: {exc}")

    if not text:
        return SynthesisResult(False, error="Model returned nothing.")

    return SynthesisResult(True, text=_tidy(text))


def _tidy(text: str) -> str:
    """Strip a leading echo of the label and collapse whitespace."""
    cleaned = " ".join(text.split())
    for prefix in ("Overview:", "Overview of"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip(" :")
    return cleaned
