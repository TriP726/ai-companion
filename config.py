"""Application configuration management."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject

from ai_companion.models.config_models import AppConfig

logger = logging.getLogger(__name__)

def _default_config_path() -> Path:
    """Absolute path to config.json, anchored to the project root.

    This was previously the bare relative string "config.json", which Python
    resolves against the CURRENT WORKING DIRECTORY. Launch the app from
    anywhere other than the project folder - a desktop shortcut, a pinned
    taskbar item, Explorer's "Open with", an IDE run button - and the app
    silently read and wrote a DIFFERENT config.json, so every setting looked
    like it had reset: no model loaded, vault back to its default.

    Anchoring to this file's location makes the config independent of how the
    app was started.
    """
    return app_root() / "config.json"


def _is_absolute(value: str) -> bool:
    """Absolute-path test that recognises Windows drive letters everywhere.

    `Path("D:/MyData").is_absolute()` is False on Linux, so a legitimate
    Windows path would be treated as relative and mangled into
    "/home/user/D:/MyData". Checking the drive-letter form explicitly keeps
    the behaviour identical on both platforms, which matters because the
    tests run on Linux and the app runs on Windows.
    """
    import re

    if not value:
        return False
    if Path(value).is_absolute():
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(value)))


def app_root() -> Path:
    """The folder holding config.json, data/, models/ and vault/.

    Differs between running from source and running as a frozen exe:

      source : <root>/ai_companion/config.py  ->  <root>
      frozen : <dist>/AICompanion/_internal/  ->  <dist>/AICompanion

    PyInstaller unpacks the package into an `_internal` subfolder, so
    anchoring to this file's location would have put config.json and the
    models directory inside `_internal` - which is why the packaged build
    reported "no voices installed" while models/piper sat beside the exe.

    Everything user-owned therefore lives next to AICompanion.exe, where it
    is visible and backup-able, rather than buried in the bundle.
    """
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # config.py lives at <root>/ai_companion/config.py
    return Path(__file__).resolve().parent.parent


DEFAULT_CONFIG_PATH = str(_default_config_path())


class ConfigManager(QObject):
    """Manages loading, saving, and accessing application configuration."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        # Read the module attribute at call time, not as a default argument
        # value bound at import, so tests (and any future override) can
        # redirect it. Resolve so a relative override is still anchored
        # predictably rather than following the process around.
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        self._path = Path(config_path).expanduser().resolve()
        self._config: AppConfig = AppConfig()
        self._migration_notes: list[str] = []
        self._migration_applied: int = 0
        self._migration_dirty: bool = False

    @property
    def config(self) -> AppConfig:
        return self._config

    # Bumped when a stored config needs a one-time correction on load.
    MIGRATION_VERSION = 2

    def load(self) -> AppConfig:
        """Load config from disk, falling back to defaults."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data = self._reject_temp_paths(data)
                data = self._migrate(data)
                self._config = AppConfig(**data)
                self._anchor_relative_paths(self._config)
                logger.info("Config loaded from %s", self._path)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to load config, using defaults: %s", e)
                self._config = AppConfig()
        else:
            self._config = AppConfig()
            self._anchor_relative_paths(self._config)
            self.save()
            logger.info("Default config created at %s", self._path)
        return self._config

    @staticmethod
    def _looks_like_temp(value: str) -> bool:
        """True for a path inside a temp or pytest tree."""
        lowered = str(value).lower().replace("\\", "/")
        markers = (
            "pytest-of-", "/pytest-", "appdata/local/temp", "/tmp/",
            "/temp/", "windows/temp",
        )
        return any(marker in lowered for marker in markers)

    def _reject_temp_paths(self, data: dict) -> dict:
        """Drop any store path that points into a temp directory.

        A build shipped earlier let the test suite persist its relocated
        config over the real one, rewriting every store path to a pytest temp
        folder. Windows deletes those, so the app silently recreated them and
        wrote new memories somewhere that vanishes - the user saw an empty
        MEM tab and lost anything added afterwards.

        Removing the field makes Pydantic fall back to the correct default,
        so a damaged config self-heals on load instead of needing a repair
        script. Belt and braces with the set_mode fix: even if something else
        writes a temp path in future, it cannot survive a restart.
        """
        repaired: list[str] = []

        if self._looks_like_temp(str(data.get("data_dir", ""))):
            data.pop("data_dir", None)
            repaired.append("data_dir")

        path_fields = (
            ("llm", "conversations_path"),
            ("memory", "store_path"),
            ("graph", "store_path"),
            ("mind_map", "store_path"),
            ("vault", "vault_root"),
            ("code_workspace", "sandbox_dir"),
            ("code_workspace", "snapshot_dir"),
            ("image_gen", "output_dir"),
            ("speech", "voices_dir"),
        )
        for section, field in path_fields:
            block = data.get(section)
            if not isinstance(block, dict):
                continue
            value = block.get(field)
            if value and self._looks_like_temp(str(value)):
                block.pop(field, None)
                repaired.append(f"{section}.{field}")

        if repaired:
            logger.warning(
                "Config pointed at temp directories; reset to defaults: %s",
                ", ".join(repaired),
            )
            self._migration_notes.append(
                "Storage paths pointed into a temp folder and have been reset "
                "to the defaults under data\\. If memories look missing, run "
                "recover_memories.py."
            )
            self._migration_dirty = True
        return data

    def _anchor_relative_paths(self, config) -> None:
        """Make relative store paths absolute against the app root.

        Config keeps paths like "data/memories.json" so the project stays
        portable, but a relative path resolves against the CURRENT WORKING
        DIRECTORY. A frozen exe is launched from wherever the shortcut points,
        so without this the app would read and write its stores somewhere
        unpredictable - the same class of bug that lost memories once already.
        """
        root = app_root()
        sections = (
            ("llm", "conversations_path"),
            ("memory", "store_path"),
            ("graph", "store_path"),
            ("mind_map", "store_path"),
            ("vault", "vault_root"),
            ("code_workspace", "sandbox_dir"),
            ("code_workspace", "snapshot_dir"),
            ("image_gen", "output_dir"),
            ("speech", "voices_dir"),
            ("speech", "kokoro_dir"),
        )
        if not _is_absolute(config.data_dir):
            config.data_dir = str(root / config.data_dir)
        for section, field in sections:
            block = getattr(config, section, None)
            if block is None:
                continue
            value = getattr(block, field, "")
            if value and not _is_absolute(value):
                setattr(block, field, str(root / value))

    def _migrate(self, data: dict) -> dict:
        """Apply one-time corrections to a config written by an older build.

        Changing a Pydantic default only affects NEW configs; an existing
        config.json keeps whatever it already stored. That is normally right,
        but the vault default flipped PRIVATE -> NORMAL precisely because
        private-by-default silently discarded the owner's conversations and
        memories. Leaving an existing install on the old value reproduces the
        exact problem the change was meant to fix, so it is corrected once,
        and recorded so a deliberate later choice of private is never
        overridden again.
        """
        applied = int(data.get("migration_version", 0) or 0)
        if applied >= self.MIGRATION_VERSION:
            data.pop("migration_version", None)
            self._migration_applied = applied
            return data

        vault = data.get("vault")
        if isinstance(vault, dict) and vault.get("mode") == "private":
            vault["mode"] = "normal"
            logger.info(
                "Migration: vault mode private -> normal so conversations "
                "and memories are saved. Use the chat header badge to go "
                "private again."
            )
            self._migration_notes.append(
                "Vault switched to SAVING. Private mode was the old default "
                "and silently discarded chats and memories. Click the header "
                "badge to go private whenever you want."
            )

        data.pop("migration_version", None)
        self._migration_applied = self.MIGRATION_VERSION
        self._migration_dirty = True
        return data

    @property
    def migration_notes(self) -> list[str]:
        """Human-readable notes about anything changed on load."""
        return list(self._migration_notes)

    def save(self) -> None:
        """Persist current config to disk atomically.

        Writes to a temporary file and replaces, so an interrupted write
        cannot leave a truncated config.json that fails to parse on the next
        launch (which would look like every setting resetting again).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._config.model_dump(mode="json")
        payload["migration_version"] = max(
            self._migration_applied, self.MIGRATION_VERSION
        )
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self._path)
        self._migration_dirty = False
        logger.info("Config saved to %s", self._path)

    def update(self, **kwargs: object) -> None:
        """Update config fields and save."""
        data = self._config.model_dump()
        data.update(kwargs)
        self._config = AppConfig(**data)
        self.save()

    def get_data_dir(self) -> Path:
        """Get the resolved data directory, creating it if needed."""
        p = Path(self._config.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
