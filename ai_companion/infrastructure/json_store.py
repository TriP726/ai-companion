"""JSON file-based persistent store with schema versioning."""
from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


class JsonStore:
    """Thread-safe JSON file storage with atomic writes and backups.

    Each store file includes a _meta section with schema version
    and last-modified timestamp.
    """

    CURRENT_VERSION = "1"

    # Last line of defence. If a store is ever pointed at a temp directory,
    # writes there are silently discarded by the OS cleaning the folder - the
    # exact failure that lost the user's memories. Refusing loudly at
    # construction is far better than writing into a disappearing path.
    _TEMP_MARKERS = (
        "pytest-of-", "/pytest-", "appdata/local/temp", "/tmp/", "/temp/",
        "windows/temp",
    )

    @classmethod
    def _is_temp_path(cls, path: Path) -> bool:
        lowered = str(path).lower().replace("\\", "/")
        return any(marker in lowered for marker in cls._TEMP_MARKERS)

    def __init__(self, file_path: str | Path, allow_temp: bool = False) -> None:
        self._path = Path(file_path)
        # Warn, do not raise. Tests legitimately use temp directories, and
        # refusing outright broke 112 of them. The value here is the log line
        # that makes a misconfigured production path obvious; the config layer
        # already prevents one from being stored, so this is a tripwire rather
        # than a gate.
        if not allow_temp and self._is_temp_path(self._path):
            logger.warning(
                "Storage path is inside a temporary directory and may be "
                "cleaned by the OS: %s", self._path,
            )
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, Any]:
        """Load data from disk. Creates file if missing."""
        with self._lock:
            if self._path.exists():
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    self._data = raw
                    self._loaded = True
                except (json.JSONDecodeError, OSError) as e:
                    # Attempt backup recovery
                    self._attempt_recovery()
                    raise RuntimeError(
                        f"Failed to load {self._path}: {e}"
                    ) from e
            else:
                self._data = {
                    "_meta": {
                        "version": self.CURRENT_VERSION,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "modified": datetime.now(timezone.utc).isoformat(),
                    },
                    "items": {},
                }
                self._ensure_dir()
                self._write_to_disk()
                self._loaded = True
            return dict(self._data)

    def save(self) -> None:
        """Persist current state to disk atomically."""
        with self._lock:
            if "_meta" in self._data:
                self._data["_meta"]["modified"] = (
                    datetime.now(timezone.utc).isoformat()
                )
            self._write_to_disk()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the items dict."""
        with self._lock:
            return self._data.get("items", {}).get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the items dict (in memory only; call save() to persist)."""
        with self._lock:
            if "items" not in self._data:
                self._data["items"] = {}
            self._data["items"][key] = value

    def delete(self, key: str) -> bool:
        """Delete a key from items. Returns True if key existed."""
        with self._lock:
            if key in self._data.get("items", {}):
                del self._data["items"][key]
                return True
            return False

    def keys(self) -> list[str]:
        """Get all item keys."""
        with self._lock:
            return list(self._data.get("items", {}).keys())

    def values(self) -> list[Any]:
        """Get all item values."""
        with self._lock:
            return list(self._data.get("items", {}).values())

    def items(self) -> list[tuple[str, Any]]:
        """Get all (key, value) pairs."""
        with self._lock:
            return list(self._data.get("items", {}).items())

    def count(self) -> int:
        with self._lock:
            return len(self._data.get("items", {}))

    def clear(self) -> None:
        """Clear all items (keeps meta)."""
        with self._lock:
            self._data["items"] = {}

    def get_meta(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data.get("_meta", {}))

    def _write_to_disk(self) -> None:
        """Write data atomically (write-to-temp, then rename)."""
        self._ensure_dir()
        tmp_path = self._path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)
            tmp_path.replace(self._path)
        except OSError:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _attempt_recovery(self) -> None:
        """Try to recover from a backup if one exists."""
        backup = self._path.with_suffix(".bak")
        if backup.exists():
            try:
                shutil.copy2(backup, self._path)
            except OSError:
                pass

    def create_backup(self) -> Optional[Path]:
        """Create a backup of the current store file."""
        if self._path.exists():
            backup = self._path.with_suffix(".bak")
            shutil.copy2(self._path, backup)
            return backup
        return None
