"""Workspace panel — file and code workspace UI."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QInputDialog,
)

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class WorkspacePanel(QWidget):
    """Combined file and code workspace UI."""

    def __init__(self, service_manager: ServiceManager) -> None:
        super().__init__()
        self._sm = service_manager
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()

        # File workspace tab
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)

        # Toolbar
        toolbar = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_files)
        toolbar.addWidget(self._refresh_btn)

        self._new_file_btn = QPushButton("New File")
        self._new_file_btn.clicked.connect(self._new_file)
        toolbar.addWidget(self._new_file_btn)

        self._new_dir_btn = QPushButton("New Folder")
        self._new_dir_btn.clicked.connect(self._new_directory)
        toolbar.addWidget(self._new_dir_btn)

        self._copy_in_btn = QPushButton("Copy In")
        self._copy_in_btn.clicked.connect(self._copy_in)
        toolbar.addWidget(self._copy_in_btn)

        toolbar.addStretch()
        file_layout.addLayout(toolbar)

        # File list + preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._file_list = QListWidget()
        self._file_list.currentItemChanged.connect(self._on_file_selected)
        splitter.addWidget(self._file_list)

        self._file_preview = QTextEdit()
        self._file_preview.setReadOnly(True)
        self._file_preview.setFont(QFont("Consolas", 10))
        self._file_preview.setObjectName("CodeSurface")
        splitter.addWidget(self._file_preview)

        splitter.setSizes([250, 550])
        file_layout.addWidget(splitter)

        # Delete button
        del_row = QHBoxLayout()
        self._delete_btn = QPushButton("Delete Selected")
        self._delete_btn.clicked.connect(self._delete_selected)
        del_row.addWidget(self._delete_btn)
        self._copy_out_btn = QPushButton("Copy Out")
        self._copy_out_btn.clicked.connect(self._copy_out)
        del_row.addWidget(self._copy_out_btn)
        file_layout.addLayout(del_row)

        tabs.addTab(file_tab, "FILES")

        # Code workspace tab
        code_tab = QWidget()
        code_layout = QVBoxLayout(code_tab)

        # Code toolbar
        code_toolbar = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.clicked.connect(self._run_code)
        code_toolbar.addWidget(self._run_btn)

        self._save_code_btn = QPushButton("Save")
        self._save_code_btn.clicked.connect(self._save_code)
        code_toolbar.addWidget(self._save_code_btn)

        self._snapshot_btn = QPushButton("Snapshot")
        self._snapshot_btn.clicked.connect(self._create_snapshot)
        code_toolbar.addWidget(self._snapshot_btn)

        code_toolbar.addStretch()
        code_layout.addLayout(code_toolbar)

        # Code editor + output
        code_splitter = QSplitter(Qt.Orientation.Vertical)

        self._code_editor = QTextEdit()
        self._code_editor.setFont(QFont("Consolas", 11))
        self._code_editor.setObjectName("CodeSurface")
        self._code_editor.setPlainText("# Write Python code here\nprint('Hello from sandbox!')\n")
        code_splitter.addWidget(self._code_editor)

        self._output_display = QTextEdit()
        self._output_display.setReadOnly(True)
        self._output_display.setFont(QFont("Consolas", 10))
        self._output_display.setObjectName("OutputSurface")
        code_splitter.addWidget(self._output_display)

        code_splitter.setSizes([400, 200])
        code_layout.addWidget(code_splitter)

        # Snapshot list
        snap_row = QHBoxLayout()
        snap_row.addWidget(QLabel("Snapshots:"))
        self._snapshot_list = QListWidget()
        self._snapshot_list.setMaximumHeight(80)
        snap_row.addWidget(self._snapshot_list)
        self._restore_snap_btn = QPushButton("Restore")
        self._restore_snap_btn.clicked.connect(self._restore_snapshot)
        snap_row.addWidget(self._restore_snap_btn)
        code_layout.addLayout(snap_row)

        tabs.addTab(code_tab, "CODE")

        # Execution history
        exec_tab = QWidget()
        exec_layout = QVBoxLayout(exec_tab)
        self._exec_list = QListWidget()
        exec_layout.addWidget(self._exec_list)
        tabs.addTab(exec_tab, "HISTORY")

        layout.addWidget(tabs)

        # Initial load
        self._refresh_files()

    def _refresh_files(self) -> None:
        self._file_list.clear()
        ws = self._sm.get("FileWorkspace")
        if ws:
            for entry in ws.list_directory():
                prefix = "[D]" if entry["is_dir"] else "   "
                item = QListWidgetItem(f"{prefix} {entry['name']}")
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self._file_list.addItem(item)

    def _on_file_selected(self, current: QListWidgetItem, previous) -> None:
        if not current:
            return
        entry = current.data(Qt.ItemDataRole.UserRole)
        if entry.get("is_dir"):
            return

        ws = self._sm.get("FileWorkspace")
        if ws:
            content = ws.read_file(entry["name"])
            if content is not None:
                self._file_preview.setPlainText(content)

    def _new_file(self) -> None:
        name, ok = QInputDialog.getText(self, "New File", "Filename:")
        if ok and name:
            ws = self._sm.get("FileWorkspace")
            if ws:
                ws.write_file(name, "")
                self._refresh_files()

    def _new_directory(self) -> None:
        name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if ok and name:
            ws = self._sm.get("FileWorkspace")
            if ws:
                ws.create_directory(name)
                self._refresh_files()

    def _copy_in(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Copy File Into Workspace"
        )
        if file_path:
            import os
            name = os.path.basename(file_path)
            ws = self._sm.get("FileWorkspace")
            if ws:
                ws.copy_in(file_path, name)
                self._refresh_files()

    def _copy_out(self) -> None:
        current = self._file_list.currentItem()
        if not current:
            return
        entry = current.data(Qt.ItemDataRole.UserRole)
        dest, _ = QFileDialog.getSaveFileName(
            self, "Copy Out", entry["name"]
        )
        if dest:
            ws = self._sm.get("FileWorkspace")
            if ws:
                ws.copy_out(entry["name"], dest)

    def _delete_selected(self) -> None:
        current = self._file_list.currentItem()
        if not current:
            return
        entry = current.data(Qt.ItemDataRole.UserRole)
        ws = self._sm.get("FileWorkspace")
        if ws:
            ws.delete_path(entry["name"])
            self._refresh_files()

    def _run_code(self) -> None:
        code = self._code_editor.toPlainText()
        if not code.strip():
            return

        self._output_display.clear()
        self._output_display.append("Running...")

        code_ws = self._sm.get("CodeWorkspace")
        if code_ws:
            exec_id = code_ws.execute(code=code)
            if exec_id:
                # Connect to execution signals
                sb = self._sm.signal_bus
                sb.code_workspace.execution_output.connect(self._on_exec_output)
                sb.code_workspace.execution_complete.connect(self._on_exec_complete)
                sb.code_workspace.execution_error.connect(self._on_exec_error)

    def _on_exec_output(self, exec_id: str, line: str) -> None:
        self._output_display.append(line)

    def _on_exec_complete(self, exec_id: str, return_code: int) -> None:
        self._output_display.append(f"\n[Process exited with code {return_code}]")

    def _on_exec_error(self, exec_id: str, error: str) -> None:
        self._output_display.append(f"\n[Error: {error}]")

    def _save_code(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save Code", "Filename:", text="script.py"
        )
        if ok and name:
            code_ws = self._sm.get("CodeWorkspace")
            if code_ws:
                code_ws.write_code(name, self._code_editor.toPlainText())

    def _create_snapshot(self) -> None:
        name, ok = QInputDialog.getText(self, "Snapshot", "Name:")
        if ok:
            code_ws = self._sm.get("CodeWorkspace")
            if code_ws:
                snap_id = code_ws.create_snapshot(name or "")
                self._refresh_snapshots()

    def _restore_snapshot(self) -> None:
        current = self._snapshot_list.currentItem()
        if not current:
            return
        snap_id = current.data(Qt.ItemDataRole.UserRole)
        code_ws = self._sm.get("CodeWorkspace")
        if code_ws:
            code_ws.restore_snapshot(snap_id)

    def _refresh_snapshots(self) -> None:
        self._snapshot_list.clear()
        code_ws = self._sm.get("CodeWorkspace")
        if code_ws:
            for snap in code_ws.list_snapshots():
                item = QListWidgetItem(
                    f"{snap['name']} ({snap['created'][:10]})"
                )
                item.setData(Qt.ItemDataRole.UserRole, snap["id"])
                self._snapshot_list.addItem(item)
