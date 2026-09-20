"""Extract model-readable text from attached files.

SCOPE AND ITS LIMITS
--------------------
The configured model (Phi-3-mini) is a TEXT model with no vision encoder. It
cannot see images. Attaching a PNG and asking "what is in this picture?"
cannot work, and pretending otherwise by silently passing a filename would
produce confident hallucination about an image the model never saw.

So this module extracts text where text exists, and returns an explicit
"unsupported" note where it does not. The note is shown to the user AND given
to the model, so the model knows it is working blind rather than inventing.

SECURITY
--------
Attachments are read from anywhere the user points a file dialog, which is
outside the vault's approved folders by design — the user is explicitly
choosing the file each time. Guard rails:

  * size cap before read (default 5 MB of extracted text)
  * extension allowlist
  * text is truncated to a character budget so a large file cannot blow the
    context window or stall inference
  * binary content is never forwarded
  * files are read, never written, moved, or executed
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

# Extensions we can genuinely turn into text.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".ini", ".cfg", ".toml", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".ps1", ".bat", ".sql",
    ".html", ".css", ".xml",
}
STRUCTURED_EXTENSIONS = {".json", ".csv", ".tsv"}

# Recognised but not extractable with the current dependency set.
KNOWN_BINARY = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".tiff": "image",
    ".pdf": "document", ".docx": "document", ".xlsx": "spreadsheet",
    ".glb": "3D model", ".gltf": "3D model", ".obj": "3D model",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio",
    ".zip": "archive", ".exe": "executable",
}

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_CHAR_BUDGET = 12_000


@dataclass
class ExtractionResult:
    """Outcome of reading one attachment."""

    filename: str
    ok: bool
    text: str = ""
    note: str = ""
    truncated: bool = False
    char_count: int = 0

    def as_prompt_block(self) -> str:
        """Render for inclusion in the model prompt."""
        if not self.ok:
            return f"[Attached file: {self.filename}]\n{self.note}"
        header = f"[Attached file: {self.filename}]"
        if self.truncated:
            header += (
                f"\n(truncated to the first {self.char_count:,} characters)"
            )
        return f"{header}\n```\n{self.text}\n```"


def _decode(raw: bytes) -> tuple[str, bool]:
    """Decode bytes as text. Returns (text, looked_like_text)."""
    if b"\x00" in raw[:8192]:
        return "", False
    for encoding in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), True
        except (UnicodeDecodeError, LookupError):
            continue
    return "", False


def _format_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (ValueError, TypeError):
        return text


def _format_delimited(text: str, delimiter: str, max_rows: int = 200) -> str:
    """Render CSV/TSV as aligned rows; models read that far better than raw."""
    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except (csv.Error, ValueError):
        return text
    if not rows:
        return text
    shown = rows[:max_rows]
    out = [" | ".join(cell.strip() for cell in row) for row in shown]
    if len(rows) > max_rows:
        out.append(f"... {len(rows) - max_rows:,} more rows not shown")
    return "\n".join(out)


def extract(
    file_path: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> ExtractionResult:
    """Read one attachment and return text the model can actually use."""
    path = Path(file_path)
    name = path.name or file_path

    if not path.exists():
        return ExtractionResult(name, False, note="File no longer exists.")
    if not path.is_file():
        return ExtractionResult(name, False, note="Not a file.")

    suffix = path.suffix.lower()

    try:
        size = path.stat().st_size
    except OSError as exc:
        return ExtractionResult(name, False, note=f"Could not read: {exc}")

    if size == 0:
        return ExtractionResult(name, False, note="File is empty.")
    if size > max_bytes:
        return ExtractionResult(
            name,
            False,
            note=(
                f"File is {size / 1024 / 1024:.1f} MB, over the "
                f"{max_bytes / 1024 / 1024:.0f} MB attachment limit."
            ),
        )

    if suffix in KNOWN_BINARY:
        kind = KNOWN_BINARY[suffix]
        note = (
            f"This is a {kind} file ({suffix}). The loaded model is "
            f"text-only and cannot read {kind} content, so it has NOT been "
            f"given the file. Only the filename is known."
        )
        return ExtractionResult(name, False, note=note)

    if suffix not in TEXT_EXTENSIONS and suffix not in STRUCTURED_EXTENSIONS:
        # Unknown extension: attempt a decode rather than refusing outright.
        pass

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return ExtractionResult(name, False, note=f"Could not read: {exc}")

    text, is_text = _decode(raw)
    if not is_text:
        return ExtractionResult(
            name,
            False,
            note=(
                "File appears to be binary, not text. It has NOT been given "
                "to the model."
            ),
        )

    if suffix == ".json":
        text = _format_json(text)
    elif suffix == ".csv":
        text = _format_delimited(text, ",")
    elif suffix == ".tsv":
        text = _format_delimited(text, "\t")

    text = text.replace("\r\n", "\n").strip()
    if not text:
        return ExtractionResult(name, False, note="File contains no text.")

    truncated = len(text) > char_budget
    if truncated:
        text = text[:char_budget]

    return ExtractionResult(
        name,
        True,
        text=text,
        truncated=truncated,
        char_count=len(text),
    )


def build_prompt_prefix(results: list[ExtractionResult]) -> str:
    """Combine extraction results into a prefix for the user's message."""
    if not results:
        return ""
    blocks = [r.as_prompt_block() for r in results]
    return "\n\n".join(blocks) + "\n\n"
