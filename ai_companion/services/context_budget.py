"""Fit a conversation into the model's context window.

THE PROBLEM
-----------
`n_ctx` is a hard ceiling covering BOTH the prompt and the reply. Nothing in
this app enforced it. Three features I added all push against that ceiling at
once:

    * injected memories (up to 1200 chars)
    * attachment text (up to 12000 chars per file)
    * unbounded conversation history (last 50 messages)

Measured on a realistic setup - 8 memories, one attached source file, a dozen
turns, ctx 4096, max_tokens 2048 - the prompt alone was ~5100 tokens against a
4096 window. llama.cpp raises, and the user sees a raw C-level error after
waiting for a response that was never possible.

THE RULE
--------
Reserve room for the reply, then spend what is left on the prompt in priority
order:

    1. system prompt (behaviour + memories) - always kept
    2. the newest user message - always kept, it is the actual question
    3. older turns - kept newest-first until the budget runs out

Dropped history is reported so the UI can say so. Silently forgetting the
middle of a conversation is worse than saying "earlier turns omitted".

TOKEN ESTIMATION
----------------
Estimated at ~4 chars/token rather than tokenised exactly. A real tokeniser
would be precise but costs a model round-trip per message; the estimate is
deliberately conservative (it over-counts slightly) and the safety margin
absorbs the error. This is an approximation on purpose, not an oversight.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CHARS_PER_TOKEN = 4

# Held back so a slightly-wrong estimate cannot push us over the real limit.
SAFETY_TOKENS = 96

# If the reply budget would leave nothing for the prompt, shrink the reply
# rather than fail: a short answer beats an exception.
MIN_PROMPT_TOKENS = 256
MIN_REPLY_TOKENS = 128


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for a string."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN + 1)


def estimate_messages(messages: list[dict]) -> int:
    """Estimate a whole message list, including per-message overhead.

    Chat templates wrap every message in role markers, which cost real tokens.
    Four per message is a reasonable allowance across common templates.
    """
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


@dataclass
class BudgetResult:
    """Outcome of fitting a prompt to the window."""

    messages: list[dict]
    max_tokens: int
    dropped_messages: int = 0
    prompt_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.dropped_messages > 0


def fit_to_context(
    messages: list[dict],
    context_length: int,
    requested_max_tokens: int,
) -> BudgetResult:
    """Trim `messages` so prompt + reply fit inside `context_length`.

    `messages` is the full list in chronological order, optionally starting
    with a system message. The system message and the final message are never
    dropped.
    """
    if not messages:
        return BudgetResult([], max(MIN_REPLY_TOKENS, requested_max_tokens))

    system: list[dict] = []
    body = list(messages)
    if body and body[0].get("role") == "system":
        system = [body.pop(0)]

    reply = max(MIN_REPLY_TOKENS, int(requested_max_tokens))
    available = context_length - reply - SAFETY_TOKENS

    # If the reply reservation starves the prompt, claw budget back from it.
    if available < MIN_PROMPT_TOKENS:
        reply = max(
            MIN_REPLY_TOKENS,
            context_length - MIN_PROMPT_TOKENS - SAFETY_TOKENS,
        )
        available = context_length - reply - SAFETY_TOKENS

    notes: list[str] = []
    system_cost = estimate_messages(system)

    # A system prompt that alone exceeds the window is pathological; keep it
    # but warn, since dropping the user's instructions silently is worse.
    if system_cost >= available and system:
        notes.append(
            "System prompt and memories nearly fill the context window. "
            "Reduce memories or raise Context Length."
        )

    remaining = available - system_cost
    kept: list[dict] = []

    # Always keep the newest message - it is the question being asked. But it
    # can itself blow the window: an attachment contributes up to 12000 chars
    # (~3000 tokens) to a single message. Clip its CONTENT rather than drop
    # it, keeping the head (the file) and the tail (the actual question).
    if body:
        newest = dict(body[-1])
        cost = estimate_tokens(newest.get("content", "")) + 4
        if cost > remaining:
            allowed_chars = max(0, (remaining - 4)) * CHARS_PER_TOKEN
            content = newest.get("content", "")
            if allowed_chars < 200:
                # Nothing sensible fits; keep only the tail so the question
                # itself survives.
                newest["content"] = content[-200:]
            else:
                head = int(allowed_chars * 0.7)
                tail = allowed_chars - head - 40
                newest["content"] = (
                    content[:head]
                    + "\n...[attachment truncated to fit context]...\n"
                    + content[-tail:]
                )
            notes.append(
                "The latest message was too large for the context window and "
                "was truncated. Reduce the attachment or raise Context Length."
            )
            cost = estimate_tokens(newest["content"]) + 4
        kept.append(newest)
        remaining -= cost

    # Then walk backwards through history while budget allows.
    dropped = 0
    for message in reversed(body[:-1]):
        cost = estimate_tokens(message.get("content", "")) + 4
        if cost <= remaining:
            kept.append(message)
            remaining -= cost
        else:
            dropped += 1

    kept.reverse()
    final = system + kept

    if dropped:
        notes.append(
            f"{dropped} earlier message{'s' if dropped != 1 else ''} omitted "
            "to fit the context window."
        )

    return BudgetResult(
        messages=final,
        max_tokens=reply,
        dropped_messages=dropped,
        prompt_tokens=estimate_messages(final),
        notes=notes,
    )
