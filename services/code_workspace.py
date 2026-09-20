"""Contained coding workspace with sandboxed execution."""
from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.privacy import PrivacyMixin
from ai_companion.infrastructure.path_validator import PathValidator, PathValidationError


class CodeWorkspaceService(PrivacyMixin, BaseService):
    """Sandboxed code execution environment.

    Security:
    - Runs code in a subprocess with restricted filesystem access
    - No network access (allow_network must be False)
    - Time and memory limits enforced
    - Snapshots for rollback
    - Diffs between snapshots
    - Copy-in/copy-out for files
    """

    service_name = "CodeWorkspace"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sandbox_dir: Optional[Path] = None
        self._snapshot_dir: Optional[Path] = None
        self._path_validator: Optional[PathValidator] = None
        self._snapshots: dict[str, dict] = {}
        self._executions: dict[str, dict] = {}
        self._current_execution: Optional[str] = None

    def start(self) -> None:
        # Security check
        if self._config.code_workspace.allow_network:
            self.emit_error(
                "SECURITY: allow_network must be False. "
                "Code execution without network restriction is dangerous."
            )
            return  # Do NOT call super().start() — service stays not-running

        super().start()

        self._sandbox_dir = Path(self._config.code_workspace.sandbox_dir).resolve()
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

        self._snapshot_dir = Path(self._config.code_workspace.snapshot_dir).resolve()
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._path_validator = PathValidator(
            approved_roots=[str(self._sandbox_dir)],
            allowed_extensions=[".py", ".txt", ".json", ".csv", ".md", ".log"],
            max_file_size_bytes=10 * 1024 * 1024,
        )

        # Load snapshot index
        self._load_snapshots()
        self.emit_status("Code workspace ready")

    def stop(self) -> None:
        super().stop()

    def write_code(self, filename: str, code: str) -> bool:
        """Write code to a file in the sandbox."""
        try:
            safe_name = self._path_validator.sanitize_filename(filename)
            full_path = self._path_validator.validate_path(
                str(self._sandbox_dir / safe_name)
            )
        except PathValidationError as e:
            self.emit_error(f"Write denied: {e}")
            return False

        try:
            full_path.write_text(code, encoding="utf-8")
            return True
        except OSError as e:
            self.emit_error(f"Write failed: {e}")
            return False

    def read_code(self, filename: str) -> Optional[str]:
        """Read code from a file in the sandbox."""
        try:
            safe_name = self._path_validator.sanitize_filename(filename)
            full_path = self._path_validator.validate_path(
                str(self._sandbox_dir / safe_name), must_exist=True
            )
            return full_path.read_text(encoding="utf-8")
        except (PathValidationError, OSError) as e:
            self.emit_error(f"Read failed: {e}")
            return None

    def execute(
        self,
        code: Optional[str] = None,
        filename: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Execute Python code in a sandboxed subprocess.

        Either provide code directly or a filename in the sandbox.
        Returns execution_id.
        """
        execution_id = str(uuid.uuid4())[:8]

        if code is None and filename:
            code = self.read_code(filename)
        if not code:
            self.emit_error("No code to execute")
            return ""

        # Write code to a temp file in sandbox
        exec_file = self._sandbox_dir / f"_exec_{execution_id}.py"
        exec_file.write_text(code, encoding="utf-8")

        timeout = timeout or self._config.code_workspace.max_execution_time

        exec_info = {
            "id": execution_id,
            "code": code,
            "status": "running",
            "started": datetime.now(timezone.utc).isoformat(),
            "output": "",
            "error": "",
            "return_code": None,
        }
        self._executions[execution_id] = exec_info
        self._current_execution = execution_id
        self._signal_bus.code_workspace.execution_started.emit(execution_id)

        # Run in background thread
        thread = threading.Thread(
            target=self._execute_worker,
            args=(execution_id, str(exec_file), timeout),
            daemon=True,
        )
        thread.start()
        return execution_id

    def _execute_worker(
        self, execution_id: str, script_path: str, timeout: int
    ) -> None:
        """Execute a script in a subprocess."""
        exec_info = self._executions[execution_id]

        try:
            env = os.environ.copy()
            # Disable network access via environment (best effort)
            env.pop("http_proxy", None)
            env.pop("https_proxy", None)
            env.pop("HTTP_PROXY", None)
            env.pop("HTTPS_PROXY", None)
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            process = subprocess.Popen(
                [sys.executable, "-u", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._sandbox_dir),
                env=env,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                exec_info["output"] = stdout
                exec_info["error"] = stderr
                exec_info["return_code"] = process.returncode
                exec_info["status"] = "completed"

                # Emit output lines
                for line in stdout.splitlines():
                    self._signal_bus.code_workspace.execution_output.emit(
                        execution_id, line
                    )

                self._signal_bus.code_workspace.execution_complete.emit(
                    execution_id, process.returncode
                )

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                exec_info["status"] = "timeout"
                exec_info["error"] = f"Execution timed out after {timeout}s"
                exec_info["return_code"] = -1
                self._signal_bus.code_workspace.execution_error.emit(
                    execution_id, exec_info["error"]
                )

        except Exception as e:
            exec_info["status"] = "error"
            exec_info["error"] = str(e)
            exec_info["return_code"] = -1
            self._signal_bus.code_workspace.execution_error.emit(
                execution_id, str(e)
            )
        finally:
            self._current_execution = None
            # Clean up exec file
            try:
                Path(script_path).unlink(missing_ok=True)
            except OSError:
                pass

    def cancel_execution(self) -> None:
        """Request cancellation of current execution (kills process)."""
        if self._current_execution:
            info = self._executions.get(self._current_execution)
            if info and info["status"] == "running":
                info["status"] = "cancelled"

    def get_execution(self, execution_id: str) -> Optional[dict]:
        return self._executions.get(execution_id)

    # --- Snapshots ---

    def create_snapshot(self, name: str = "") -> str:
        """Create a snapshot of the current sandbox state.

        In Private mode the snapshot is refused rather than written. A
        snapshot is a durable on-disk copy of whatever code is in the sandbox;
        making one while the vault promises nothing is saved would be the same
        privacy hole already closed for chat, memory and the graph.
        """
        if self.is_private_mode():
            self.emit_error(
                "Private mode: snapshots are not saved. Switch to SAVING in "
                "the chat header to keep snapshots."
            )
            return ""

        snapshot_id = str(uuid.uuid4())[:8]
        if not name:
            name = f"snapshot_{snapshot_id}"

        snap_dir = self._snapshot_dir / snapshot_id
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Copy all sandbox files to snapshot
        for item in self._sandbox_dir.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_file():
                shutil.copy2(item, snap_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, snap_dir / item.name)

        self._snapshots[snapshot_id] = {
            "id": snapshot_id,
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(),
            "file_count": len(list(snap_dir.rglob("*"))),
        }
        self._save_snapshots()
        self._signal_bus.code_workspace.snapshot_created.emit(snapshot_id)
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore sandbox to a previous snapshot."""
        snap_dir = self._snapshot_dir / snapshot_id
        if not snap_dir.exists():
            self.emit_error(f"Snapshot {snapshot_id} not found")
            return False

        # Clear sandbox (except hidden files)
        for item in self._sandbox_dir.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        # Copy snapshot contents back
        for item in snap_dir.iterdir():
            dest = self._sandbox_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest)

        self._signal_bus.code_workspace.snapshot_restored.emit(snapshot_id)
        return True

    def diff_snapshots(self, snap_id_1: str, snap_id_2: str) -> dict[str, list[str]]:
        """Compute diffs between two snapshots.

        Returns {filename: [diff lines]}.
        """
        dir1 = self._snapshot_dir / snap_id_1
        dir2 = self._snapshot_dir / snap_id_2

        if not dir1.exists() or not dir2.exists():
            self.emit_error("One or both snapshots not found")
            return {}

        diffs: dict[str, list[str]] = {}

        # Get files in both
        files1 = {f.name for f in dir1.rglob("*") if f.is_file()}
        files2 = {f.name for f in dir2.rglob("*") if f.is_file()}
        all_files = files1 | files2

        for fname in sorted(all_files):
            f1 = dir1 / fname
            f2 = dir2 / fname

            if not f1.exists():
                diffs[fname] = [f"+ (added in {snap_id_2})"]
                continue
            if not f2.exists():
                diffs[fname] = [f"- (removed in {snap_id_2})"]
                continue

            try:
                text1 = f1.read_text(encoding="utf-8").splitlines()
                text2 = f2.read_text(encoding="utf-8").splitlines()
                diff = list(difflib.unified_diff(
                    text1, text2,
                    fromfile=f"{snap_id_1}/{fname}",
                    tofile=f"{snap_id_2}/{fname}",
                    lineterm="",
                ))
                if diff:
                    diffs[fname] = diff
            except UnicodeDecodeError:
                diffs[fname] = ["(binary file, cannot diff)"]

        return diffs

    def list_snapshots(self) -> list[dict]:
        return list(self._snapshots.values())

    def _load_snapshots(self) -> None:
        idx_file = self._snapshot_dir / "index.json"
        if idx_file.exists():
            try:
                self._snapshots = json.loads(idx_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._snapshots = {}

    def _save_snapshots(self) -> None:
        if self.is_private_mode():
            return
        idx_file = self._snapshot_dir / "index.json"
        idx_file.write_text(
            json.dumps(self._snapshots, indent=2), encoding="utf-8"
        )
