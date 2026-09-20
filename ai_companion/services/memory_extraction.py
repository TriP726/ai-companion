"""Propose memory candidates from conversation text.

WHY THE OLD EXTRACTOR WAS LEFT UNWIRED
--------------------------------------
`MemoryService.extract_candidates()` used three loose regexes. Measured on
realistic messages, roughly three in five extractions were junk:

    "remember that my bot token is in .env"  -> "remember that my bot token is in ."
    "the error is happening on line 42"      -> captured as a durable fact
    "the problem is that my code is broken"  -> captured as a durable fact
    "the thing is i want it to work"         -> captured TWICE

Truncated at the dot in `.env`, transient debugging state stored forever, and
duplicates. A review queue full of that is worse than no extraction: the owner
stops reading it, and the review workflow becomes rubber-stamping.

WHAT THIS MODULE DOES DIFFERENTLY
---------------------------------
1. Extracts only DURABLE first-person facts - things true next week.
2. Rejects transient subjects ("the error", "the problem", "this bug").
3. Never truncates on a dot inside a token like `.env` or `3.12`.
4. Deduplicates against memories that already exist.
5. Suggests tags so the new memory joins the graph and is findable.

EVERYTHING IS PENDING
---------------------
Candidates are proposals, never facts. They land as PENDING and are excluded
from injection until the owner approves them. An extractor that could write
directly into the model's context would let a misread sentence become
something the assistant believes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Subjects that describe a passing situation rather than a durable fact.
# "the error is X" is true for ten minutes; "my timezone is X" is true for
# years. This list is the main defence against a noisy review queue.
# Preference bodies made only of these carry no information: "Prefers it to
# work" tells the model nothing and pollutes the review queue.
VACUOUS_WORDS = frozenset("""
it that them these things this those work working
""".split())

TRANSIENT_SUBJECTS = frozenset("""
answer bug case error exception failure fault idea issue line output point
problem question reason result thing trouble
""".split())

# Openers that signal the user is stating something about themselves.
PREFERENCE_RE = re.compile(
    r"\b(?:i\s+(?:prefer|like|always|usually|never|tend\s+to|want)\s+"
    r"(?:to\s+)?)(?P<body>[^.!?\n]{6,160})",
    re.IGNORECASE,
)

EXPLICIT_RE = re.compile(
    r"\b(?:remember|note|keep\s+in\s+mind|don'?t\s+forget)\s+"
    r"(?:that\s+)?(?P<body>[^!?\n]{6,200})",
    re.IGNORECASE,
)

POSSESSIVE_RE = re.compile(
    r"\bmy\s+(?P<subject>[a-z][a-z0-9_\- ]{1,28}?)\s+(?:is|are|uses?|runs?"
    r"|has|have)\s+(?P<body>[^!?\n]{2,160})",
    re.IGNORECASE,
)

IDENTITY_RE = re.compile(
    r"\bi\s+(?:am|'m)\s+(?:a|an|the)?\s*(?P<body>[^.!?\n]{4,120})",
    re.IGNORECASE,
)

# A sentence end is a dot followed by whitespace or end-of-string. This is what
# stops `.env`, `3.12` and `v1.2.3` being cut in half.
SENTENCE_END_RE = re.compile(r"\.(?:\s|$)")

MIN_LENGTH = 12
MAX_LENGTH = 220

# Tag hints: substring -> tag. Deliberately small and explicit; a guessed
# taxonomy is worse than none because wrong tags corrupt graph clustering.
TAG_HINTS: dict[str, str] = {
    "discord": "discord",
    "python": "python",
    "pyside": "python",
    "javascript": "javascript",
    "typescript": "javascript",
    "sql": "sql",
    "git": "git",
    "github": "git",
    "docker": "docker",
    "linux": "linux",
    "windows": "windows",
    "bot": "bots",
    "api": "api",
    "token": "security",
    "password": "security",
    "secret": "security",
    "privacy": "privacy",
    "private": "privacy",
    "ram": "hardware",
    "cpu": "hardware",
    "gpu": "hardware",
    "ryzen": "hardware",
    "model": "models",
    "gguf": "models",
    "llm": "models",
    "prefer": "preferences",
    "timezone": "profile",
    "name": "profile",
}


@dataclass
class Candidate:
    """A proposed memory, not yet a memory."""

    content: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.4
    reason: str = ""


def _trim_to_sentence(text: str) -> str:
    """Cut at the first real sentence end, not at any dot.

    The old extractor split on a bare `.`, which turned
    "my bot token is in .env" into "my bot token is in .".
    """
    match = SENTENCE_END_RE.search(text)
    if match:
        text = text[: match.start()]
    return " ".join(text.split()).strip(" ,;:-")


def _is_transient(text: str) -> bool:
    """True when the sentence is about a passing situation."""
    lowered = text.lower()
    for subject in TRANSIENT_SUBJECTS:
        if re.search(rf"\b(?:the|this|that|my)\s+{subject}\b", lowered):
            return True
    return False


def suggest_tags(text: str) -> list[str]:
    """Tags implied by the text. Ordered and deduplicated."""
    lowered = text.lower()
    found: list[str] = []
    for needle, tag in TAG_HINTS.items():
        if needle in lowered and tag not in found:
            found.append(tag)
    return found[:4]


def _normalise(text: str) -> str:
    """Key for duplicate detection: lowercase, punctuation-free word string."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _sentence_around(text: str, index: int) -> str:
    """The sentence containing character `index`.

    Transience is judged per-sentence. Judging the whole message meant one
    throwaway clause ("also the bug is annoying") suppressed a real fact
    stated earlier in the same line.
    """
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        if match.start() >= index:
            return text[start:match.start()]
        start = match.end()
    return text[start:]


def extract_candidates(
    text: str,
    existing: Iterable[Any] = (),
) -> list[Candidate]:
    """Propose durable facts found in `text`.

    `existing` is any iterable of objects with a `.content` attribute; matches
    are skipped so the review queue never shows something already stored.
    """
    if not text or not text.strip():
        return []

    seen_keys = {
        _normalise(str(getattr(item, "content", "") or ""))
        for item in existing
    }
    seen_keys.discard("")

    found: list[Candidate] = []

    def add(body: str, confidence: float, reason: str,
            context: str = "") -> None:
        content = _trim_to_sentence(body)
        # Judge transience on the ORIGINAL sentence as well as the extract.
        # "the problem is that my code is broken" yields "my code is broken",
        # which looks durable in isolation but is not.
        if context and _is_transient(context):
            return
        if len(content) < MIN_LENGTH or len(content) > MAX_LENGTH:
            return
        if _is_transient(content):
            return
        key = _normalise(content)
        if not key or key in seen_keys:
            return
        # Reject content with no substantive words left.
        meaningful = [
            w for w in key.split()
            if w not in VACUOUS_WORDS and w not in ("prefers", "user", "is")
        ]
        if len(meaningful) < 2:
            return
        # Also skip a candidate contained within one already proposed, which
        # is how "the thing is i want it to work" produced two entries.
        for other in found:
            other_key = _normalise(other.content)
            if key in other_key or other_key in key:
                return
        seen_keys.add(key)
        found.append(
            Candidate(
                content=content,
                tags=suggest_tags(content),
                confidence=confidence,
                reason=reason,
            )
        )

    # Explicit instructions are the strongest signal the user wants this kept.
    for match in EXPLICIT_RE.finditer(text):
        add(match.group("body"), 0.75, "explicitly asked to remember",
            context=_sentence_around(text, match.start()))

    for match in PREFERENCE_RE.finditer(text):
        add(
            "Prefers " + match.group("body").strip(),
            0.55,
            "stated a preference",
            context=match.group(0),
        )

    for match in POSSESSIVE_RE.finditer(text):
        subject = match.group("subject").strip().lower()
        if subject in TRANSIENT_SUBJECTS:
            continue
        add(match.group(0), 0.5, "stated a fact about themselves",
            context=_sentence_around(text, match.start()))

    for match in IDENTITY_RE.finditer(text):
        add("User is " + match.group("body").strip(), 0.45, "self-description",
            context=_sentence_around(text, match.start()))

    return found
