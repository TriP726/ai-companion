"""Private Vault service — secure local file memory."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.path_validator import PathValidator, PathValidationError
from ai_companion.models.config_models import VaultMode


class VaultService(BaseService):
    """Manages the Private Vault — a controlled local file store.

    In PRIVATE mode:
    - No persistent memories are written
    - No file copies created outside the vault
    - No generated files written outside approved locations
    - No audit trail written (logs only in memory)
    - All file access is validated and audited
    """

    service_name = "Vault"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mode = self._config.vault.mode
        self._audit_log: list[dict] = []
        self._vault_store: Optional[JsonStore] = None
        self._path_validator: Optional[PathValidator] = None
        self._vault_root: Optional[Path] = None

    @property
    def mode(self) -> VaultMode:
        return self._mode

    @property
    def is_private(self) -> bool:
        return self._mode == VaultMode.PRIVATE

    def start(self) -> None:
        super().start()
        self._vault_root = Path(self._config.vault.vault_root).resolve()
        self._vault_root.mkdir(parents=True, exist_ok=True)

        # Include both vault root and any approved folders
        approved = [str(self._vault_root)]
        approved.extend(self._config.vault.approved_folders)

        self._path_validator = PathValidator(
            approved_roots=approved,
            allowed_extensions=self._config.vault.allowed_extensions,
            max_file_size_bytes=self._config.vault.max_file_size_mb * 1024 * 1024,
        )

        self._vault_store = JsonStore(self._vault_root / "vault_index.json")
        self._vault_store.load()

        self.emit_status(f"Vault started in {self._mode.value} mode")

    def stop(self) -> None:
        if self._vault_store:
            self._vault_store.save()
        super().stop()

    def set_mode(self, mode: VaultMode, persist: bool = True) -> None:
        """Switch vault mode.

        `persist` writes the choice to config.json so it survives a restart.
        A mode toggle that silently reverts on next launch is worse than no
        toggle: the user believes they are private when they are not.
        """
        old = self._mode
        if old == mode:
            return
        self._mode = mode
        self._config.vault.mode = mode
        self._audit("mode_change", f"{old.value} -> {mode.value}")

        if persist:
            try:
                self._persist_mode(mode)
            except Exception as exc:  # noqa: BLE001 - never block the toggle
                self.emit_error(f"Could not save vault mode: {exc}")

        self._signal_bus.vault.mode_changed.emit(mode.value)

    def _persist_mode(self, mode: VaultMode) -> None:
        """Write ONLY the vault mode back to config.json.

        This used to save the service's whole in-memory AppConfig. Under test
        that object has every store path relocated to a pytest temp folder, so
        running the suite rewrote the user's real config to point at
        directories Windows later deletes - the app then found no memories and
        looked like it had wiped them.

        Reading the file, changing one key, and writing it back means a caller
        holding a relocated config can no longer corrupt the real one.
        """
        import json

        from ai_companion.config import DEFAULT_CONFIG_PATH

        path = Path(DEFAULT_CONFIG_PATH)
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}

        if not isinstance(data, dict):
            data = {}
        data.setdefault("vault", {})
        if not isinstance(data["vault"], dict):
            data["vault"] = {}
        data["vault"]["mode"] = mode.value

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def import_file(self, source_path: str, category: str = "general") -> Optional[str]:
        """Import a file into the vault.

        Returns the vault-relative path, or None on failure.
        """
        if self.is_private:
            self._audit("import_blocked", f"Private mode: {source_path}")
            self.emit_error("Cannot import files in Private mode")
            return None

        try:
            src = self._path_validator.validate_path(
                source_path, must_exist=True, check_extension=True
            )
            self._path_validator.validate_file_size(src)
        except PathValidationError as e:
            self.emit_error(f"Path validation failed: {e}")
            return None

        # Create category directory in vault
        cat_dir = self._vault_root / "files" / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize and copy
        safe_name = self._path_validator.sanitize_filename(src.name)
        dest = cat_dir / safe_name

        # Avoid overwrite
        counter = 1
        while dest.exists():
            stem = src.stem
            dest = cat_dir / f"{stem}_{counter}{src.suffix}"
            counter += 1

        try:
            shutil.copy2(src, dest)
            # Update index
            rel_path = str(dest.relative_to(self._vault_root))
            entry = {
                "original_name": src.name,
                "vault_path": rel_path,
                "category": category,
                "size": dest.stat().st_size,
                "imported": datetime.now(timezone.utc).isoformat(),
            }
            self._vault_store.set(rel_path, entry)
            self._vault_store.save()

            self._audit("import", f"{src} -> {rel_path}")
            self._signal_bus.vault.file_imported.emit(rel_path)
            return rel_path
        except (OSError, shutil.Error) as e:
            self.emit_error(f"File import failed: {e}")
            return None

    def remove_file(self, vault_path: str) -> bool:
        """Remove a file from the vault."""
        try:
            full_path = self._path_validator.validate_path(
                str(self._vault_root / vault_path), must_exist=True
            )
        except PathValidationError as e:
            self.emit_error(f"Cannot remove: {e}")
            return False

        try:
            full_path.unlink()
            self._vault_store.delete(vault_path)
            self._vault_store.save()
            self._audit("remove", vault_path)
            self._signal_bus.vault.file_removed.emit(vault_path)
            return True
        except OSError as e:
            self.emit_error(f"Remove failed: {e}")
            return False

    def list_files(self, category: Optional[str] = None) -> list[dict]:
        """List files in the vault."""
        files = []
        for key, entry in self._vault_store.items():
            if category and entry.get("category") != category:
                continue
            files.append(entry)
        return files

    def get_file_path(self, vault_path: str) -> Optional[Path]:
        """Get the full filesystem path for a vault file."""
        full = self._vault_root / vault_path
        if full.exists():
            return full
        return None

    def can_write_outside(self) -> bool:
        """In private mode, writing outside vault is forbidden."""
        return not self.is_private

    def _audit(self, event_type: str, detail: str) -> None:
        """Log an audit event (in-memory only in private mode)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "detail": detail,
        }
        self._audit_log.append(entry)
        self._signal_bus.vault.audit_event.emit(event_type, detail)

    def get_audit_log(self) -> list[dict]:
        """Return a copy of the in-memory audit log."""
        return list(self._audit_log)
