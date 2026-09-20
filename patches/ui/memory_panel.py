"""Memory panel — view, search, edit, and approve memories."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QInputDialog,
)

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class MemoryPanel(QWidget):
    """UI for managing long-term memories."""

    def __init__(self, service_manager: ServiceManager) -> None:
        super().__init__()
        self._sm = service_manager
        self._current_memory_id: str = ""
        self._setup_ui()
        self._connect_signals()
        self._refresh_list()

    def _setup_ui(self) -> None:
        from ai_companion.ui.theme import theme

        m = theme.m
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: memory list + search
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(m.space_4, m.space_4, m.space_3, m.space_4)
        left_layout.setSpacing(m.space_3)

        # Search bar
        search_row = QHBoxLayout()
        search_row.setSpacing(m.space_2)
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search memories...")
        self._search_input.textChanged.connect(self._on_search)
        search_row.addWidget(self._search_input)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "Approved", "Pending", "Pinned"])
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        search_row.addWidget(self._filter_combo)
        left_layout.addLayout(search_row)

        # Without this the tab looks like a notepad. State plainly that
        # approved memories are fed to the model and pending ones are not.
        self._injection_label = QLabel("")
        self._injection_label.setObjectName("MetaText")
        self._injection_label.setWordWrap(True)
        left_layout.addWidget(self._injection_label)

        self._memory_list = QListWidget()
        self._memory_list.currentItemChanged.connect(self._on_memory_selected)
        left_layout.addWidget(self._memory_list)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(m.space_2)
        self._add_btn = QPushButton("+ Add")
        self._add_btn.setObjectName("PrimaryButton")
        self._add_btn.clicked.connect(self._add_memory)
        btn_row.addWidget(self._add_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("DangerButton")
        self._delete_btn.clicked.connect(self._delete_memory)
        btn_row.addWidget(self._delete_btn)

        self._pin_btn = QPushButton("Pin")
        self._pin_btn.clicked.connect(self._toggle_pin)
        btn_row.addWidget(self._pin_btn)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # Right: memory detail / editor
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(m.space_3, m.space_4, m.space_4, m.space_4)
        right_layout.setSpacing(m.space_3)

        self._detail_label = QLabel("Select a memory to view details")
        self._detail_label.setObjectName("TitleText")
        right_layout.addWidget(self._detail_label)

        # Content editor
        right_layout.addWidget(QLabel("Content:"))
        self._content_edit = QTextEdit()
        self._content_edit.setMinimumHeight(150)
        right_layout.addWidget(self._content_edit)

        # Tags
        tags_row = QHBoxLayout()
        tags_row.addWidget(QLabel("Tags (comma-separated):"))
        self._tags_input = QLineEdit()
        tags_row.addWidget(self._tags_input)
        right_layout.addLayout(tags_row)

        # Confidence
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.1)
        conf_row.addWidget(self._confidence_spin)
        right_layout.addLayout(conf_row)

        # Status & provenance
        self._status_label = QLabel("Status: —")
        self._status_label.setObjectName("MetaText")
        right_layout.addWidget(self._status_label)

        self._provenance_label = QLabel("Provenance: —")
        self._provenance_label.setObjectName("MonoText")
        right_layout.addWidget(self._provenance_label)

        # Action buttons
        action_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Changes")
        self._save_btn.clicked.connect(self._save_changes)
        action_row.addWidget(self._save_btn)

        self._approve_btn = QPushButton("Approve")
        self._approve_btn.clicked.connect(self._approve_memory)
        action_row.addWidget(self._approve_btn)

        self._reject_btn = QPushButton("Reject")
        self._reject_btn.clicked.connect(self._reject_memory)
        action_row.addWidget(self._reject_btn)
        right_layout.addLayout(action_row)

        splitter.addWidget(right)
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        sb = self._sm.signal_bus
        sb.vault.mode_changed.connect(lambda _: self._refresh_list())
        sb.memory.memory_added.connect(lambda _: self._refresh_list())
        sb.memory.memory_updated.connect(lambda _: self._refresh_list())
        sb.memory.memory_deleted.connect(self._on_memory_deleted)
        sb.memory.memory_approved.connect(lambda _: self._refresh_list())
        sb.memory.memory_rejected.connect(lambda _: self._refresh_list())

    @staticmethod
    def _restyle(widget) -> None:
        """Re-evaluate the stylesheet after an objectName change."""
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _update_injection_label(self) -> None:
        """Show what reaches the model, and warn when nothing will persist.

        Private mode stops memories being written to disk. Without saying so
        here, a memory added in private mode looks saved, works all session,
        and is silently gone on restart - which is exactly the confusion this
        banner exists to prevent.
        """
        from ai_companion.services.memory_injection import (
            DEFAULT_CHAR_BUDGET,
            select_memories,
        )

        mem_svc = self._sm.get("Memory")
        if mem_svc is None:
            return

        vault = self._sm.get("Vault")
        is_private = bool(vault is not None and vault.is_private)
        try:
            all_memories = mem_svc.search()
        except Exception:  # noqa: BLE001
            return

        approved = [
            m for m in all_memories if m.status.value == "approved"
        ]
        pending = [m for m in all_memories if m.status.value == "pending"]
        chosen = select_memories(all_memories)
        used = sum(len(m.content) + 3 for m in chosen)

        if is_private:
            self._injection_label.setObjectName("PrivateNotice")
            self._restyle(self._injection_label)
            self._injection_label.setText(
                "PRIVATE MODE - memories added now are NOT written to disk "
                "and will be lost when you close the app. Click the badge in "
                "the Chat header to switch to SAVING."
            )
            return

        if self._injection_label.objectName() != "MetaText":
            self._injection_label.setObjectName("MetaText")
            self._restyle(self._injection_label)

        if not all_memories:
            self._injection_label.setText(
                "Approved memories are added to the system prompt of every "
                "message, so the model remembers them across conversations. "
                "Nothing here yet - press + Add."
            )
            return

        parts = [
            f"{len(chosen)} of {len(approved)} approved "
            f"{'memory' if len(approved) == 1 else 'memories'} sent to the "
            f"model ({used}/{DEFAULT_CHAR_BUDGET} chars)"
        ]
        if len(chosen) < len(approved):
            parts.append("- the rest do not fit the context budget")
        if pending:
            parts.append(
                f"- {len(pending)} pending, NOT sent until approved"
            )
        self._injection_label.setText(" ".join(parts))

    def _refresh_list(self) -> None:
        self._memory_list.clear()
        mem_svc = self._sm.get("Memory")
        if not mem_svc:
            return

        from ai_companion.models.memory import MemoryFilter, MemoryStatus
        filter_text = self._filter_combo.currentText()
        filters = MemoryFilter(query=self._search_input.text())

        if filter_text == "Approved":
            filters.status = MemoryStatus.APPROVED
        elif filter_text == "Pending":
            filters.status = MemoryStatus.PENDING
        elif filter_text == "Pinned":
            filters.pinned = True

        for mem in mem_svc.search(filters):
            prefix = "* " if mem.pinned else ""
            status_icon = {
                "approved": "[ok]",
                "pending": "[..]",
                "rejected": "[no]",
            }.get(mem.status.value, "")
            display = f"{prefix}{status_icon} {mem.summary[:60]}"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, mem.id)
            self._memory_list.addItem(item)

        self._update_injection_label()

    def _on_memory_selected(self, current: QListWidgetItem, previous) -> None:
        if not current:
            return
        memory_id = current.data(Qt.ItemDataRole.UserRole)
        self._load_memory_detail(memory_id)

    def _load_memory_detail(self, memory_id: str) -> None:
        mem_svc = self._sm.get("Memory")
        if not mem_svc:
            return
        mem = mem_svc.get_memory(memory_id)
        if not mem:
            return

        self._current_memory_id = memory_id
        self._detail_label.setText(f"Memory: {mem.id[:8]}...")
        self._content_edit.setPlainText(mem.content)
        self._tags_input.setText(", ".join(mem.tags))
        self._confidence_spin.setValue(mem.confidence)
        self._status_label.setText(f"Status: {mem.status.value}")
        self._provenance_label.setText(
            f"Provenance: {mem.provenance or 'none'} | "
            f"Source: {mem.source.value} | "
            f"Created: {mem.created[:10]}"
        )

    def _on_search(self, text: str) -> None:
        self._refresh_list()

    def _on_filter_changed(self, text: str) -> None:
        self._refresh_list()

    def _add_memory(self) -> None:
        text, ok = QInputDialog.getMultiLineText(
            self, "Add Memory", "Memory content:", ""
        )
        if ok and text.strip():
            mem_svc = self._sm.get("Memory")
            if mem_svc:
                mem_svc.add_memory(
                    content=text.strip(),
                    auto_approve=True,
                    provenance="Manual entry",
                )
                self._refresh_list()

    def _save_changes(self) -> None:
        if not self._current_memory_id:
            return
        mem_svc = self._sm.get("Memory")
        if mem_svc:
            tags = [t.strip() for t in self._tags_input.text().split(",") if t.strip()]
            mem_svc.update_memory(
                self._current_memory_id,
                content=self._content_edit.toPlainText(),
                tags=tags,
                confidence=self._confidence_spin.value(),
            )

    def _approve_memory(self) -> None:
        if not self._current_memory_id:
            return
        mem_svc = self._sm.get("Memory")
        if mem_svc:
            mem_svc.approve_memory(self._current_memory_id)
            self._refresh_list()

    def _reject_memory(self) -> None:
        if not self._current_memory_id:
            return
        mem_svc = self._sm.get("Memory")
        if mem_svc:
            mem_svc.reject_memory(self._current_memory_id)
            self._refresh_list()

    def _delete_memory(self) -> None:
        if not self._current_memory_id:
            return
        mem_svc = self._sm.get("Memory")
        if mem_svc:
            mem_svc.delete_memory(self._current_memory_id)
            self._current_memory_id = ""
            self._refresh_list()

    def _toggle_pin(self) -> None:
        if not self._current_memory_id:
            return
        mem_svc = self._sm.get("Memory")
        if mem_svc:
            mem_svc.toggle_pin(self._current_memory_id)
            self._refresh_list()

    def _on_memory_deleted(self, memory_id: str) -> None:
        if memory_id == self._current_memory_id:
            self._current_memory_id = ""
            self._detail_label.setText("Select a memory")
        self._refresh_list()
