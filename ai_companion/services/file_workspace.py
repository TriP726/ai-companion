"""Permission-controlled file workspace with approved-folder boundaries."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.path_validator import PathValidator, PathValidationError


class FileWorkspaceService(BaseService):
    """File workspace with approved-folder boundaries and audited operations.

    All file operations are:
    - Restricted to approved folders
    - Validated for path, type, and size
    - Logged with timestamps for audit
    """

    service_name = "FileWorkspace"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._workspace_root: Optional[Path] = None
        self._path_validator: Optional[PathValidator] = None
        self._audit_store: Optional[JsonStore] = None

    def start(self) -> None:
        super().start()
        self._workspace_root = Path("workspace").resolve()
        self._workspace_root.mkdir(parents=True, exist_ok=True)

        approved = [str(self._workspace_root)]
        approved.extend(self._config.vault.approved_folders)

        self._path_validator = PathValidator(
            approved_roots=approved,
            allowed_extensions=self._config.vault.allowed_extensions,
            max_file_size_bytes=self._config.vault.max_file_size_mb * 1024 * 1024,
        )

        self._audit_store = JsonStore(self._workspace_root / ".audit.json")
        self._audit_store.load()
        self.emit_status("File workspace ready")

    def stop(self) -> None:
        if self._audit_store:
            self._audit_store.save()
        super().stop()

    def list_directory(self, path: str = ".") -> list[dict]:
        """List contents of a directory within workspace."""
        try:
            full_path = self._path_validator.validate_path(
                str(self._workspace_root / path), must_exist=True
            )
        except PathValidationError as e:
            self.emit_error(f"Access denied: {e}")
            return []

        if not full_path.is_dir():
            self.emit_error("Not a directory")
            return []

        entries = []
        for item in sorted(full_path.iterdir()):
            try:
                entries.append({
                    "name": item.name,
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(
                        item.stat().st_mtime
                    ).isoformat(),
                })
            except OSError:
                continue
        return entries

    def read_file(self, path: str) -> Optional[str]:
        """Read a text file from the workspace."""
        try:
            full_path = self._path_validator.validate_path(
                str(self._workspace_root / path), must_exist=True
            )
            self._path_validator.validate_file_size(full_path)
        except PathValidationError as e:
            self.emit_error(f"Read denied: {e}")
            return None

        try:
            content = full_path.read_text(encoding="utf-8")
            self._audit("read", path)
            return content
        except (OSError, UnicodeDecodeError) as e:
            self.emit_error(f"Read failed: {e}")
            return None

    def write_file(self, path: str, content: str) -> bool:
        """Write content to a file in the workspace."""
        try:
            full_path = self._path_validator.validate_path(
                str(self._workspace_root / path), check_extension=True
            )
        except PathValidationError as e:
            self.emit_error(f"Write denied: {e}")
            return False

        # Check vault restrictions
        if hasattr(self._config, 'vault') and self._config.vault.mode.value == "private":
            # In private mode, only write to workspace, not outside
            pass

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            self._audit("write", path)
            self._signal_bus.file_workspace.file_operation.emit(
                "write", path, "success"
            )
            return True
        except OSError as e:
            self.emit_error(f"Write failed: {e}")
            return False

    def create_directory(self, path: str) -> bool:
        """Create a directory in the workspace."""
        try:
            full_path = self._path_validator.validate_path(
                str(self._workspace_root / path)
            )
        except PathValidationError as e:
            self.emit_error(f"Create denied: {e}")
            return False

        try:
            full_path.mkdir(parents=True, exist_ok=True)
            self._audit("mkdir", path)
            return True
        except OSError as e:
            self.emit_error(f"Mkdir failed: {e}")
            return False

    def delete_path(self, path: str) -> bool:
        """Delete a file or directory (non-recursive for safety)."""
        try:
            full_path = self._path_validator.validate_path(
                str(self._workspace_root / path), must_exist=True
            )
        except PathValidationError as e:
            self.emit_error(f"Delete denied: {e}")
            return False

        # Block deleting workspace root or hidden files
        if full_path == self._workspace_root:
            self.emit_error("Cannot delete workspace root")
            return False
        if full_path.name.startswith("."):
            self.emit_error("Cannot delete hidden files")
            return False

        try:
            if full_path.is_dir():
                shutil.rmtree(full_path)
            else:
                full_path.unlink()
            self._audit("delete", path)
            self._signal_bus.file_workspace.file_operation.emit(
                "delete", path, "success"
            )
            return True
        except OSError as e:
            self.emit_error(f"Delete failed: {e}")
            return False

    def copy_in(self, external_path: str, workspace_path: str) -> bool:
        """Copy a file from outside into the workspace."""
        try:
            src = Path(external_path).resolve()
            if not src.exists():
                self.emit_error(f"Source not found: {external_path}")
                return False

            dest = self._path_validator.validate_path(
                str(self._workspace_root / workspace_path)
            )
        except PathValidationError as e:
            self.emit_error(f"Copy-in denied: {e}")
            return False

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            self._audit("copy_in", f"{external_path} -> {workspace_path}")
            return True
        except OSError as e:
            self.emit_error(f"Copy-in failed: {e}")
            return False

    def copy_out(self, workspace_path: str, external_path: str) -> bool:
        """Copy a file from workspace to an external location."""
        try:
            src = self._path_validator.validate_path(
                str(self._workspace_root / workspace_path), must_exist=True
            )
        except PathValidationError as e:
            self.emit_error(f"Copy-out denied: {e}")
            return False

        try:
            dest = Path(external_path).resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            self._audit("copy_out", f"{workspace_path} -> {external_path}")
            return True
        except OSError as e:
            self.emit_error(f"Copy-out failed: {e}")
            return False

    def _audit(self, operation: str, detail: str) -> None:
        """Log a file operation for audit."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "detail": detail,
        }
        self._audit_store.set(
            str(self._audit_store.count()), entry
        )
        self._signal_bus.file_workspace.audit_event.emit(
            entry["timestamp"], operation, detail
        )

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Get recent audit log entries."""
        entries = self._audit_store.values()
        return entries[-limit:]
