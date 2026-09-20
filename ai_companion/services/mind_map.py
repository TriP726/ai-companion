"""Mind Map: concepts derived from memories, linked by shared evidence.

THE MODEL INVERSION
-------------------
The previous graph made each *memory* a node and tags secondary. This inverts
it, which is the right way round:

    concept  = a node (a person, place, topic, goal)
    memory   = evidence that a concept matters
    link     = two concepts appearing in the same memories

So "grandpa" is one bubble holding twelve memories, "space" is another, and
they link because six memories mention both. Bubble size is memories absorbed;
link weight is memories in common. That is what makes the picture mean
something rather than just being a scatter of facts.

WHAT IS DETERMINISTIC AND WHAT IS NOT
-------------------------------------
Everything in this module is deterministic and cheap: concept detection,
sizing, linking, clustering, colouring. No model call, no latency.

The *written overview* for a concept - the paragraph that connects satellites,
NASA and a childhood interest in engineering - needs an LLM and is generated
on demand elsewhere. Synthesising every concept in the background would cost
roughly ten minutes of CPU generation on this hardware and would compete with
the chat the user is waiting on.

PRIVACY
-------
Private memories never contribute. A concept derived from a private memory
would leak its content into a store that IS written to disk.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# --- categories -------------------------------------------------------

CATEGORY_PERSON = "person"
CATEGORY_PLACE = "place"
CATEGORY_TOPIC = "topic"
CATEGORY_GOAL = "goal"
CATEGORY_THING = "thing"

# Subtle, distinguishable hues that sit inside the HUD palette. Category is a
# hint, not a label - the colours are deliberately close together so the graph
# reads as one system rather than a pie chart.
CATEGORY_COLORS = {
    CATEGORY_PERSON: "#f472b6",
    CATEGORY_PLACE: "#2dd4bf",
    CATEGORY_TOPIC: "#a855f7",
    CATEGORY_GOAL: "#fbbf24",
    CATEGORY_THING: "#818cf8",
}

# Words that look like proper nouns but are not concepts.
NAME_STOPLIST = frozenset("""
I I'm I've Monday Tuesday Wednesday Thursday Friday Saturday Sunday
January February March April May June July August September October
November December Yes No OK Okay The A An This That These Those
""".split())

GOAL_HINTS = frozenset("""
goal goals plan plans want wants need needs learning build building ship
finish launch improve
""".split())

PLACE_HINTS = frozenset("""
city town country street office home room house lab studio
""".split())

# A capitalised token, allowing internal hyphens and apostrophes.
PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z'\-]{2,})\b")

MIN_MENTIONS_FOR_NAME = 2   # a name must recur before it becomes a concept
MIN_SHARED_FOR_LINK = 1     # memories in common before two concepts link


@dataclass
class Concept:
    """One Mind Map bubble."""

    key: str
    label: str
    category: str = CATEGORY_TOPIC
    memory_ids: list[str] = field(default_factory=list)
    overview: str = ""          # LLM-written, empty until synthesised
    overview_stale: bool = True

    @property
    def weight(self) -> int:
        """How many memories feed this concept. Drives bubble size."""
        return len(self.memory_ids)

    @property
    def color(self) -> str:
        return CATEGORY_COLORS.get(self.category, CATEGORY_COLORS[CATEGORY_TOPIC])

    def radius(self, smallest: float = 16.0, largest: float = 46.0,
               max_weight: int = 1) -> float:
        """Bubble radius scaled against the busiest concept.

        Uses sqrt so AREA grows with memory count. Scaling the radius linearly
        makes a 10-memory concept look 10x more important than a 1-memory one,
        which badly overstates the difference.
        """
        if max_weight <= 1:
            return smallest
        ratio = math.sqrt(self.weight) / math.sqrt(max_weight)
        return smallest + (largest - smallest) * ratio


@dataclass
class Link:
    """A connection between two concepts, weighted by shared memories."""

    source: str
    target: str
    shared: int = 0

    @property
    def strength(self) -> float:
        """0..1 pull. More shared memories pulls bubbles closer."""
        return min(1.0, self.shared / 5.0)


@dataclass
class MindMap:
    """The whole derived structure."""

    concepts: dict[str, Concept] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    skipped_private: int = 0

    @property
    def max_weight(self) -> int:
        if not self.concepts:
            return 1
        return max(c.weight for c in self.concepts.values())

    def ordered(self) -> list[Concept]:
        """Concepts, busiest first, then alphabetical for stability."""
        return sorted(
            self.concepts.values(),
            key=lambda c: (-c.weight, c.label.lower()),
        )

    def links_for(self, key: str) -> list[Link]:
        return [l for l in self.links if key in (l.source, l.target)]

    def summary(self) -> str:
        if not self.concepts:
            return "No concepts yet."
        text = (
            f"{len(self.concepts)} concepts, {len(self.links)} connections"
        )
        if self.skipped_private:
            text += f" ({self.skipped_private} private memories excluded)"
        return text


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


def _categorise(label: str, memories: list) -> str:
    """Best-effort category for a concept.

    Heuristic on purpose. Getting this exactly right needs an LLM, and a
    slightly wrong colour is a far smaller cost than minutes of generation.
    """
    lowered = label.lower()

    if lowered in PLACE_HINTS:
        return CATEGORY_PLACE
    if lowered in GOAL_HINTS:
        return CATEGORY_GOAL

    # A bare capitalised word that is not a known topic reads as a person.
    if label[:1].isupper() and " " not in label and label not in NAME_STOPLIST:
        return CATEGORY_PERSON

    blob = " ".join(
        str(getattr(m, "content", "") or "") for m in memories
    ).lower()
    if any(hint in blob for hint in ("want to", "plan to", "goal", "learning")):
        return CATEGORY_GOAL
    if any(hint in lowered for hint in ("city", "town", "street", "office")):
        return CATEGORY_PLACE

    return CATEGORY_TOPIC


def _eligible(memory: Any, include_private: bool) -> bool:
    status = getattr(getattr(memory, "status", None), "value", "")
    if status != "approved":
        return False
    if getattr(memory, "private", False) and not include_private:
        return False
    return bool(str(getattr(memory, "content", "") or "").strip())


def build_mind_map(
    memories: Iterable,
    include_private: bool = False,
    detect_names: bool = True,
) -> MindMap:
    """Derive concepts and links from memories.

    Concepts come from two sources:
      * tags - curated, high precision
      * repeated proper nouns - catches "Hector" without anyone tagging it

    A proper noun must appear in at least MIN_MENTIONS_FOR_NAME memories
    before it is promoted, so one passing mention does not create a bubble.
    """
    mind_map = MindMap()

    usable: list = []
    for memory in memories:
        if getattr(memory, "private", False) and not include_private:
            if getattr(getattr(memory, "status", None), "value", "") == "approved":
                mind_map.skipped_private += 1
            continue
        if _eligible(memory, include_private):
            usable.append(memory)

    if not usable:
        return mind_map

    # --- tag concepts ---
    for memory in usable:
        mem_id = str(getattr(memory, "id", ""))
        for raw in getattr(memory, "tags", None) or []:
            tag = str(raw).strip().lower()
            if not tag:
                continue
            key = f"tag:{tag}"
            concept = mind_map.concepts.get(key)
            if concept is None:
                concept = Concept(key=key, label=tag)
                mind_map.concepts[key] = concept
            if mem_id not in concept.memory_ids:
                concept.memory_ids.append(mem_id)

    # --- proper-noun concepts ---
    if detect_names:
        mentions: dict[str, list[str]] = {}
        display: dict[str, str] = {}
        for memory in usable:
            mem_id = str(getattr(memory, "id", ""))
            text = str(getattr(memory, "content", "") or "")
            seen_here: set[str] = set()
            for match in PROPER_NOUN_RE.finditer(text):
                word = match.group(1)
                if word in NAME_STOPLIST:
                    continue
                # Skip a word that only ever appears at the start of a
                # sentence - that is capitalisation, not a name.
                if match.start() == 0 and word.lower() in (
                    "the", "this", "that", "my", "our"
                ):
                    continue
                # "Grandpa" and "Grandpa's" are the same concept. Without
                # stripping the possessive they became two bubbles, each with
                # an undercounted weight.
                low = word.lower().rstrip("'").removesuffix("'s")
                if len(low) < 3:
                    continue
                word = word.rstrip("'")
                if word.lower().endswith("'s"):
                    word = word[:-2]
                if low in seen_here:
                    continue
                seen_here.add(low)
                mentions.setdefault(low, []).append(mem_id)
                display.setdefault(low, word)

        for low, ids in mentions.items():
            if len(ids) < MIN_MENTIONS_FOR_NAME:
                continue
            key = f"name:{low}"
            if f"tag:{low}" in mind_map.concepts:
                # A tag already covers it; do not create a duplicate bubble.
                continue
            mind_map.concepts[key] = Concept(
                key=key, label=display[low], memory_ids=list(dict.fromkeys(ids))
            )

    # --- categorise ---
    by_id = {str(getattr(m, "id", "")): m for m in usable}
    for concept in mind_map.concepts.values():
        members = [by_id[i] for i in concept.memory_ids if i in by_id]
        concept.category = _categorise(concept.label, members)

    # --- links from shared memories ---
    keys = list(mind_map.concepts)
    for i, a_key in enumerate(keys):
        a_ids = set(mind_map.concepts[a_key].memory_ids)
        for b_key in keys[i + 1:]:
            shared = len(a_ids & set(mind_map.concepts[b_key].memory_ids))
            if shared >= MIN_SHARED_FOR_LINK:
                mind_map.links.append(Link(a_key, b_key, shared))

    return mind_map


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def layout_mind_map(
    mind_map: MindMap,
    iterations: int = 220,
) -> dict[str, tuple[float, float]]:
    """Force-directed positions where link strength pulls concepts together.

    Differs from a plain spring layout in two ways that matter for reading:
      * attraction scales with shared-memory count, so strongly related
        concepts sit closer
      * separation accounts for BUBBLE RADIUS, so a large bubble does not
        swallow its neighbours
    """
    concepts = mind_map.ordered()
    if not concepts:
        return {}

    count = len(concepts)
    max_weight = mind_map.max_weight
    radii = {
        c.key: c.radius(max_weight=max_weight) for c in concepts
    }

    k = max(96.0, 34.0 * math.sqrt(count))
    positions: dict[str, list[float]] = {}
    for index, concept in enumerate(concepts):
        angle = 2 * math.pi * index / count
        # Busiest concepts start nearer the middle; they end up as the core.
        pull = 1.0 - (concept.weight / max(max_weight, 1)) * 0.55
        spread = max(140.0, 42.0 * math.sqrt(count)) * pull
        positions[concept.key] = [
            spread * math.cos(angle),
            spread * math.sin(angle),
        ]

    temperature = k * 0.5
    for step in range(iterations):
        cooling = temperature * (1.0 - step / iterations)
        disp: dict[str, list[float]] = {c.key: [0.0, 0.0] for c in concepts}

        for i, a in enumerate(concepts):
            for b in concepts[i + 1:]:
                dx = positions[a.key][0] - positions[b.key][0]
                dy = positions[a.key][1] - positions[b.key][1]
                dist = max(math.sqrt(dx * dx + dy * dy), 0.01)
                # Repulsion must decay. Unbounded 1/d blew the layout out to
                # a 20000px span for 16 bubbles, so zoom-to-fit shrank every
                # bubble to a dot. Cutting it off past ~2.5k keeps distant
                # pairs from pushing each other to infinity.
                if dist > k * 2.5:
                    continue
                force = (k * k) / dist
                fx, fy = (dx / dist) * force, (dy / dist) * force
                disp[a.key][0] += fx
                disp[a.key][1] += fy
                disp[b.key][0] -= fx
                disp[b.key][1] -= fy

        for link in mind_map.links:
            if link.source not in positions or link.target not in positions:
                continue
            dx = positions[link.source][0] - positions[link.target][0]
            dy = positions[link.source][1] - positions[link.target][1]
            dist = max(math.sqrt(dx * dx + dy * dy), 0.01)
            # Stronger links pull harder - this is what forms clusters.
            force = (dist * dist) / k * (0.5 + link.strength)
            fx, fy = (dx / dist) * force, (dy / dist) * force
            disp[link.source][0] -= fx
            disp[link.source][1] -= fy
            disp[link.target][0] += fx
            disp[link.target][1] += fy

        # Gravity toward the origin. Without it, concepts with no links are
        # only ever pushed outward and the canvas grows without bound.
        for concept in concepts:
            px, py = positions[concept.key]
            disp[concept.key][0] -= px * 0.30
            disp[concept.key][1] -= py * 0.30

        for concept in concepts:
            dx, dy = disp[concept.key]
            length = max(math.sqrt(dx * dx + dy * dy), 0.01)
            limit = min(length, cooling)
            positions[concept.key][0] += (dx / length) * limit
            positions[concept.key][1] += (dy / length) * limit

    # Separation that respects bubble size, plus room for the label beneath.
    label_gap = 16.0
    for _ in range(140):
        moved = False
        for i, a in enumerate(concepts):
            for b in concepts[i + 1:]:
                needed = radii[a.key] + radii[b.key] + label_gap
                dx = positions[a.key][0] - positions[b.key][0]
                dy = positions[a.key][1] - positions[b.key][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < needed:
                    if dist < 0.01:
                        angle = (i * 2.399) % (2 * math.pi)
                        dx, dy, dist = math.cos(angle), math.sin(angle), 1.0
                    push = (needed - dist) / 2.0 + 0.5
                    ux, uy = dx / dist, dy / dist
                    positions[a.key][0] += ux * push
                    positions[a.key][1] += uy * push
                    positions[b.key][0] -= ux * push
                    positions[b.key][1] -= uy * push
                    moved = True
        if not moved:
            break

    min_x = min(p[0] for p in positions.values())
    min_y = min(p[1] for p in positions.values())
    return {
        key: (p[0] - min_x + 80.0, p[1] - min_y + 80.0)
        for key, p in positions.items()
    }
