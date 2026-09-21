"""Offline spell checking with right-click corrections.

Wires a red squiggle under unknown words in a QLineEdit/QTextEdit and adds
suggestions to the right-click menu, the same as a browser or word processor.

FULLY OFFLINE
-------------
`pyspellchecker` ships its dictionary as a bundled data file and performs no
network access, which is the only reason it is acceptable here. If the package
is missing the feature degrades to a no-op rather than breaking the composer.

WHY THE CUSTOM WORD LIST
------------------------
A general English dictionary flags `async`, `json`, `gguf`, `def`, `pyside`
and most of the vocabulary this app is used with. Unfiltered, nearly every
technical sentence would be underlined, the squiggles would become noise, and
the user would switch the feature off. TECH_WORDS suppresses that.

Users can also add their own words permanently via the context menu; those
persist in the config directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

try:
    from spellchecker import SpellChecker

    _AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SpellChecker = None  # type: ignore[assignment]
    _AVAILABLE = False

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QSyntaxHighlighter,
    QTextCharFormat,
)

# Vocabulary a general dictionary does not know but this app sees constantly.
TECH_WORDS = {
    # languages / runtimes
    "python", "py", "javascript", "js", "typescript", "ts", "html", "css",
    "sql", "bash", "powershell", "cmd", "json", "yaml", "yml", "toml", "xml",
    "csv", "regex", "http", "https", "url", "uri", "api", "cli", "gui", "ui",
    "ide", "os", "cpu", "gpu", "ram", "vram", "ssd", "usb", "io",
    # python
    "def", "async", "await", "init", "str", "int", "bool", "dict", "len",
    "args", "kwargs", "stdout", "stderr", "stdin", "repr", "enum", "dataclass",
    "pydantic", "pytest", "pip", "venv", "numpy", "pathlib", "asyncio",
    "traceback", "runtime", "runtimes", "iterable", "tuple", "boolean",
    "param", "params", "arg", "config", "configs", "env", "util", "utils",
    "func", "impl", "src", "lib", "libs", "pkg", "exe", "dll",
    # this project
    "pyside", "qt", "qwidget", "gguf", "llm", "llms", "llama", "gpt", "phi",
    "qwen", "mistral", "ollama", "comfyui", "sdxl", "whisper", "tts", "quant",
    "quantization", "quantized", "tokenizer", "tokenizers", "inference",
    "embeddings", "multimodal", "backend", "backends", "frontend", "offline",
    "workspace", "workspaces", "sandbox", "sandboxed", "vault", "changelog",
    # tooling / services
    "discord", "github", "gitlab", "git", "repo", "repos", "commit", "commits",
    "docker", "linux", "windows", "macos", "ubuntu", "debian", "nvidia", "amd",
    "intel", "ryzen", "huggingface", "openai", "anthropic",
    # common informal usage
    "app", "apps", "auth", "admin", "async", "config", "dev", "info", "spec",
    "specs", "sync", "temp", "webhook", "webhooks", "plugin", "plugins",
    "bot", "bots", "chatbot", "chatbots", "login", "logout", "username",
    "filename", "filenames", "filepath",
    "timestamp", "timestamps", "metadata", "changelog", "screenshot",
    "screenshots", "dropdown", "checkbox", "tooltip", "scrollbar", "toolbar",
    "hotkey", "hotkeys", "keybind", "keybinds", "autocomplete",
}

_WORD_RE = QRegularExpression(r"\b[A-Za-z][A-Za-z']*\b")
# Skip things that are not prose: code-ish tokens, paths, URLs, numbers.
_SKIP_RE = re.compile(
    r"^(?:[A-Z]{2,}|.*\d.*|.*_.*|.*[/\\].*|.*\..*)$"
)


class SpellChecker_:
    """Thin wrapper so the rest of the app never imports the library."""

    def __init__(self, custom_path: Optional[Path] = None) -> None:
        self._custom_path = custom_path
        self._custom: set[str] = set()
        self._checker = None
        if _AVAILABLE:
            self._checker = SpellChecker()
        self._load_custom()

    @property
    def available(self) -> bool:
        return self._checker is not None

    def _load_custom(self) -> None:
        if self._custom_path and self._custom_path.exists():
            try:
                data = json.loads(
                    self._custom_path.read_text(encoding="utf-8")
                )
                self._custom = {str(w).lower() for w in data}
            except (ValueError, OSError):
                self._custom = set()

    def _save_custom(self) -> None:
        if not self._custom_path:
            return
        try:
            self._custom_path.parent.mkdir(parents=True, exist_ok=True)
            self._custom_path.write_text(
                json.dumps(sorted(self._custom), indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def add_word(self, word: str) -> None:
        self._custom.add(word.lower())
        self._save_custom()

    def is_misspelled(self, word: str) -> bool:
        if not self._checker or len(word) < 3:
            return False
        lowered = word.lower()
        if lowered in TECH_WORDS or lowered in self._custom:
            return False
        if _SKIP_RE.match(word):
            return False
        # Treat CamelCase as code, not prose.
        if word[1:] != word[1:].lower():
            return False
        return lowered not in self._checker

    def suggestions(self, word: str, limit: int = 5) -> list[str]:
        if not self._checker:
            return []
        try:
            candidates = self._checker.candidates(word.lower()) or set()
        except Exception:  # noqa: BLE001 - never break the context menu
            return []
        ranked = sorted(
            candidates,
            key=lambda w: -self._checker.word_usage_frequency(w),
        )
        out = [w for w in ranked if w != word.lower()][:limit]
        # Mirror the original capitalisation.
        if word[:1].isupper():
            out = [w.capitalize() for w in out]
        return out


class SpellHighlighter(QSyntaxHighlighter):
    """Draws the red wavy underline under unknown words."""

    def __init__(self, document, checker: SpellChecker_) -> None:
        super().__init__(document)
        self._checker = checker
        from ai_companion.ui.theme import theme

        self._format = QTextCharFormat()
        self._format.setUnderlineColor(QColor(theme.p.danger))
        self._format.setUnderlineStyle(
            QTextCharFormat.UnderlineStyle.SpellCheckUnderline
        )

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt naming
        if not self._checker.available:
            return
        it = _WORD_RE.globalMatch(text)
        while it.hasNext():
            match = it.next()
            word = match.captured()
            if self._checker.is_misspelled(word):
                self.setFormat(
                    match.capturedStart(),
                    match.capturedLength(),
                    self._format,
                )


def word_at_cursor(text: str, position: int) -> tuple[str, int, int]:
    """Return (word, start, end) for the word under `position`."""
    if not text:
        return "", 0, 0
    start = position
    end = position
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "'"):
        start -= 1
    while end < len(text) and (text[end].isalpha() or text[end] == "'"):
        end += 1
    return text[start:end], start, end


def build_menu_entries(
    menu,
    checker: SpellChecker_,
    word: str,
    on_replace,
    on_add,
) -> bool:
    """Prepend spelling suggestions to an existing context menu.

    Returns True when something was added.
    """
    if not word or not checker.is_misspelled(word):
        return False

    suggestions = checker.suggestions(word)
    first = menu.actions()[0] if menu.actions() else None

    if suggestions:
        for suggestion in suggestions:
            action = QAction(suggestion, menu)
            action.setData(suggestion)
            # Bold the top suggestion so the likely pick is obvious.
            font = action.font()
            if suggestion == suggestions[0]:
                font.setBold(True)
                action.setFont(font)
            action.triggered.connect(
                lambda _=False, s=suggestion: on_replace(s)
            )
            menu.insertAction(first, action)
    else:
        none_action = QAction("(no suggestions)", menu)
        none_action.setEnabled(False)
        menu.insertAction(first, none_action)

    add_action = QAction(f'Add "{word}" to dictionary', menu)
    add_action.triggered.connect(lambda: on_add(word))
    menu.insertAction(first, add_action)
    menu.insertSeparator(first)
    return True
