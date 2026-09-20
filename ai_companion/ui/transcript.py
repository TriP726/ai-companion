"""HTML rendering for the chat transcript.

Kept separate from ChatPanel so it is testable without a display server —
previously all of this logic was unreachable by the test suite.

Design notes:
- QTextEdit's rich-text engine collapses runs of spaces and ignores
  `white-space: pre-wrap`, so indentation must be emitted as &nbsp;.
- QTextEdit supports only a CSS subset: no flexbox, no border-radius on
  inline elements. Message "cards" are therefore built from tables, which
  is the only reliable way to get a filled block with padding.
- All model output is escaped. Unescaped output containing < or & is parsed
  as markup and silently swallowed.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

from ai_companion.ui.theme import Palette, Metrics

_FENCE = re.compile(r"^\s*```(\w*)\s*$")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")


@dataclass(frozen=True)
class Segment:
    """A run of the message: either prose or a fenced code block."""

    kind: str  # "text" | "code"
    text: str
    language: str = ""


def split_segments(text: str) -> list[Segment]:
    """Split raw model output into prose and fenced code segments.

    An unterminated fence (common while streaming) still yields a code
    segment, so a block being typed out renders as code immediately rather
    than flickering from prose to code when the closing fence arrives.
    """
    segments: list[Segment] = []
    buffer: list[str] = []
    in_code = False
    language = ""

    def flush(kind: str, lang: str = "") -> None:
        if buffer:
            segments.append(Segment(kind, "\n".join(buffer), lang))
            buffer.clear()

    for line in text.split("\n"):
        fence = _FENCE.match(line)
        if fence:
            if in_code:
                flush("code", language)
                in_code = False
                language = ""
            else:
                flush("text")
                in_code = True
                language = fence.group(1)
            continue
        buffer.append(line)

    flush("code" if in_code else "text", language)
    return segments


def _inline(text: str) -> str:
    """Escape, then re-apply a minimal inline markdown subset."""
    out = html.escape(text)
    out = _INLINE_CODE.sub(
        r'<span style="font-family:monospace;">\1</span>', out
    )
    out = _BOLD.sub(r"<b>\1</b>", out)
    return out


def render_code(segment: Segment, p: Palette, m: Metrics) -> str:
    """Render a fenced block as a bordered monospace card."""
    rows = []
    for line in segment.text.split("\n"):
        esc = html.escape(line)
        stripped = esc.lstrip(" ")
        indent = "&nbsp;" * (len(esc) - len(stripped))
        rows.append(f"{indent}{stripped or '&nbsp;'}")
    body = "<br>".join(rows)
    label = (
        f'<div style="color:{p.fg_muted};font-family:{m.font_mono};'
        f'font-size:{m.text_xs}px;">{html.escape(segment.language)}</div>'
        if segment.language
        else ""
    )
    return (
        f'<table cellpadding="10" cellspacing="0" width="100%" '
        f'style="background-color:{p.code_bg};border:1px solid {p.border};">'
        f"<tr><td>{label}"
        f'<div style="font-family:{m.font_mono};color:{p.code_fg};'
        f'font-size:{m.text_sm}px;">{body}</div>'
        f"</td></tr></table>"
    )


def render_body(text: str, p: Palette, m: Metrics) -> str:
    """Render a full message body: prose paragraphs plus code cards."""
    parts: list[str] = []
    for seg in split_segments(text):
        if seg.kind == "code":
            if seg.text.strip():
                parts.append(render_code(seg, p, m))
            continue
        lines = [
            _inline(line) if line.strip() else "&nbsp;"
            for line in seg.text.split("\n")
        ]
        # Trim blank lines at the edges of a prose run so code cards do not
        # end up with a stray empty line above and below them.
        while lines and lines[0] == "&nbsp;":
            lines.pop(0)
        while lines and lines[-1] == "&nbsp;":
            lines.pop()
        if lines:
            parts.append(
                f'<div style="color:{p.fg_primary};line-height:150%;">'
                + "<br>".join(lines)
                + "</div>"
            )
    return "".join(parts)


def render_message(role: str, text: str, p: Palette, m: Metrics) -> str:
    """Render one complete message as a labelled card."""
    colors = {
        "user": p.role_user,
        "assistant": p.role_assistant,
        "system": p.role_system,
    }
    accent = colors.get(role, p.fg_secondary)
    label = {"user": "YOU", "assistant": "ASSISTANT", "system": "SYSTEM"}.get(
        role, role.upper()
    )

    if role == "system":
        return (
            f'<div style="color:{p.fg_muted};font-family:{m.font_mono};'
            f'font-size:{m.text_xs}px;padding:6px 0;">— {html.escape(text)}</div>'
        )

    bg = p.bg_elevated if role == "assistant" else p.bg_base
    return (
        f'<table cellpadding="12" cellspacing="0" width="100%" '
        f'style="background-color:{bg};border-left:2px solid {accent};">'
        f"<tr><td>"
        f'<div style="color:{accent};font-family:{m.font_mono};'
        f'font-size:{m.text_xs}px;font-weight:700;letter-spacing:1px;">'
        f"{label}</div>"
        f'<div style="padding-top:6px;">{render_body(text, p, m)}</div>'
        f"</td></tr></table>"
    )


def render_spacer() -> str:
    """Vertical gap between message cards."""
    return '<div style="font-size:5px;">&nbsp;</div>'
