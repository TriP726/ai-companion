"""Centralized path validation to enforce filesystem boundaries."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Optional


class PathValidationError(Exception):
    """Raised when a path fails validation."""
    pass


class PathValidator:
    """Validates and constrains file paths to approved boundaries.

    Security model:
    - All paths are resolved to absolute, canonical form
    - Traversal attacks (../) are blocked
    - Only paths under approved root directories are allowed
    - File extensions are validated against an allowlist
    - File sizes are checked before operations
    """

    def __init__(
        self,
        approved_roots: list[str],
        allowed_extensions: Optional[list[str]] = None,
        max_file_size_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        self._approved_roots: list[Path] = []
        for root in approved_roots:
            resolved = Path(root).resolve()
            self._approved_roots.append(resolved)
        self._allowed_extensions: set[str] | None = (
            {ext.lower() for ext in allowed_extensions}
            if allowed_extensions else None
        )
        self._max_file_size = max_file_size_bytes

    def validate_path(
        self,
        path_str: str,
        must_exist: bool = False,
        check_extension: bool = True,
    ) -> Path:
        """Validate a path string and return its resolved Path.

        Raises PathValidationError on any violation.
        """
        if not path_str or not path_str.strip():
            raise PathValidationError("Empty path")

        # Block null bytes
        if "\x00" in path_str:
            raise PathValidationError("Path contains null bytes")

        path = Path(path_str).resolve()

        # Check it's under at least one approved root
        under_root = False
        for root in self._approved_roots:
            try:
                path.relative_to(root)
                under_root = True
                break
            except ValueError:
                continue

        if not under_root:
            raise PathValidationError(
                f"Path '{path_str}' is outside approved directories"
            )

        if must_exist and not path.exists():
            raise PathValidationError(f"Path does not exist: {path}")

        if check_extension and self._allowed_extensions is not None:
            ext = path.suffix.lower()
            if ext and ext not in self._allowed_extensions:
                raise PathValidationError(
                    f"Extension '{ext}' not in allowed list"
                )

        return path

    def validate_file_size(self, path: Path) -> None:
        """Check that a file is within size limits."""
        if path.is_file():
            size = path.stat().st_size
            if size > self._max_file_size:
                raise PathValidationError(
                    f"File size {size} exceeds limit {self._max_file_size}"
                )

    def is_within_roots(self, path_str: str) -> bool:
        """Check if a path is within approved roots without raising."""
        try:
            self.validate_path(path_str)
            return True
        except PathValidationError:
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to prevent path traversal and invalid chars."""
        # Strip path separators and traversal
        name = PurePosixPath(filename).name
        if not name or name in (".", ".."):
            raise PathValidationError(f"Invalid filename: {filename}")

        # Remove characters invalid on Windows
        invalid_chars = '<>:"|?*'
        for ch in invalid_chars:
            name = name.replace(ch, "_")

        # Block hidden files
        if name.startswith("."):
            raise PathValidationError("Hidden files not allowed")

        return name
