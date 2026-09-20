"""Chat panel — streaming conversation UI."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
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
)

from ai_companion.ui.icons import svg_icon
from ai_companion.ui.spellcheck import (
    SpellChecker_,
    SpellHighlighter,
    build_menu_entries,
    word_at_cursor,
)
from ai_companion.ui.theme import theme
from ai_companion.ui.transcript import render_message, render_spacer

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class ComposerInput(QTextEdit):
    """Chat composer: Enter sends, Shift+Enter makes a new line.

    A QTextEdit is required (rather than QLineEdit) because the spell-check
    squiggle is drawn by a QSyntaxHighlighter, which needs a QTextDocument.
    """

    submitted = Signal()
    escaped = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setTabChangesFocus(True)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._min_height = 38
        self._max_height = 160
        self.setFixedHeight(self._min_height)
        self.textChanged.connect(self._autosize)

    def _autosize(self) -> None:
        """Grow with the text, up to a cap, like a modern chat box."""
        doc_height = int(self.document().size().height()) + 12
        self.setFixedHeight(
            max(self._min_height, min(doc_height, self._max_height))
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submitted.emit()
            return
        if key == Qt.Key.Key_Escape:
            self.escaped.emit()
            return
        super().keyPressEvent(event)


class ChatPanel(QWidget):
    """Main chat interface with streaming, history, and attachments."""

    def __init__(self, service_manager: ServiceManager) -> None:
        super().__init__()
        self._sm = service_manager
        self._current_conv_id: str = ""
        self._streaming_text: str = ""
        self._stream_anchor: int | None = None
        self._attachment_paths: list[str] = []
        self._voice_turn = False
        self._speller = SpellChecker_(
            custom_path=Path("data") / "custom_dictionary.json"
        )
        self._setup_ui()
        self._connect_signals()
        self._sync_service_state()
        self._restore_session()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        m = theme.m
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- sidebar ----------------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(200)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, m.space_2, 0, m.space_3)
        side_layout.setSpacing(m.space_2)

        heading = QLabel("CONVERSATIONS")
        heading.setObjectName("SectionLabel")
        side_layout.addWidget(heading)

        self._conv_list = QListWidget()
        self._conv_list.setFrameShape(QFrame.Shape.NoFrame)
        self._conv_list.currentItemChanged.connect(self._on_conv_selected)
        side_layout.addWidget(self._conv_list, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(m.space_3, 0, m.space_3, 0)
        btn_row.setSpacing(m.space_2)

        self._new_btn = QPushButton("  New")
        self._new_btn.setObjectName("PrimaryButton")
        self._new_btn.clicked.connect(self.new_conversation)
        btn_row.addWidget(self._new_btn)

        self._del_btn = QPushButton("")
        self._del_btn.setObjectName("DangerButton")
        self._del_btn.setToolTip("Delete conversation")
        self._del_btn.setFixedWidth(40)
        self._del_btn.clicked.connect(self._delete_conversation)
        btn_row.addWidget(self._del_btn)
        side_layout.addLayout(btn_row)

        splitter.addWidget(sidebar)

        # ---------------- main column ----------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("HeaderBar")
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(
            m.space_4, m.space_3, m.space_4, m.space_3
        )
        header_row.setSpacing(m.space_3)

        self._model_label = QLabel("NO MODEL LOADED")
        self._model_label.setObjectName("StatusWarn")
        header_row.addWidget(self._model_label)
        header_row.addStretch()

        self._voice_btn = QPushButton("VOICE OFF")
        self._voice_btn.setObjectName("VoiceBadge")
        self._voice_btn.setCheckable(True)
        self._voice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._voice_btn.setToolTip(
            "Hands-free voice. Say the wake word, then speak.\n"
            "Run tools/get_speech.py first to download the models."
        )
        self._voice_btn.clicked.connect(self._toggle_voice)
        header_row.addWidget(self._voice_btn)

        # The indicator IS the control. A badge that reports privacy while the
        # switch lives in a settings dialog is backwards - the user reads the
        # badge, not the dialog, so that is where the toggle belongs.
        self._privacy_btn = QPushButton("SAVING")
        self._privacy_btn.setObjectName("SavingBadge")
        self._privacy_btn.setCheckable(True)
        self._privacy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._privacy_btn.clicked.connect(self._toggle_private)
        header_row.addWidget(self._privacy_btn)
        right_layout.addWidget(header)

        self._chat_display = QTextEdit()
        self._chat_display.setObjectName("ChatDisplay")
        self._chat_display.setReadOnly(True)
        self._chat_display.setFrameShape(QFrame.Shape.NoFrame)
        right_layout.addWidget(self._chat_display, 1)

        # ---------------- composer ----------------
        composer = QFrame()
        composer.setObjectName("HeaderBar")
        input_row = QHBoxLayout(composer)
        input_row.setContentsMargins(
            m.space_4, m.space_3, m.space_4, m.space_3
        )
        input_row.setSpacing(m.space_2)

        self._attach_btn = QPushButton("")
        self._attach_btn.setObjectName("GhostButton")
        self._attach_btn.setToolTip("Attach a file")
        self._attach_btn.setFixedWidth(38)
        self._attach_btn.clicked.connect(self._attach_file)
        input_row.addWidget(self._attach_btn)

        # A QTextEdit rather than a QLineEdit: only a QTextDocument can carry
        # the red squiggle, and it gives multi-line input for free.
        self._input = ComposerInput()
        self._input.setObjectName("ChatInput")
        self._input.setPlaceholderText("Message your local model…")
        self._input.submitted.connect(self._send_message)
        self._input.escaped.connect(self._clear_attachments)
        self._input.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._input.customContextMenuRequested.connect(
            self._show_input_context_menu
        )
        self._input_highlighter = SpellHighlighter(
            self._input.document(), self._speller
        )
        input_row.addWidget(self._input, 1)

        self._send_btn = QPushButton("  Send")
        self._send_btn.setObjectName("PrimaryButton")
        self._send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self._send_btn)

        self._cancel_btn = QPushButton("  Stop")
        self._cancel_btn.setObjectName("DangerButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_generation)
        input_row.addWidget(self._cancel_btn)

        right_layout.addWidget(composer)

        self._attachments_label = QLabel("")
        self._attachments_label.setObjectName("MonoText")
        self._attachments_label.setContentsMargins(m.space_4, 0, m.space_4, 0)
        self._attachments_label.hide()
        right_layout.addWidget(self._attachments_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([theme.m.sidebar_width, 900])
        layout.addWidget(splitter)

        self.apply_theme()

    def _connect_signals(self) -> None:
        sb = self._sm.signal_bus
        sb.llm.chunk_received.connect(self._on_chunk)
        sb.llm.generation_started.connect(self._on_gen_started)
        sb.llm.generation_complete.connect(self._on_gen_complete)
        sb.llm.generation_cancelled.connect(self._on_gen_cancelled)
        sb.llm.generation_error.connect(self._on_gen_error)
        sb.llm.model_loaded.connect(self._on_model_loaded)
        sb.llm.model_unloaded.connect(self._on_model_unloaded)
        sb.vault.mode_changed.connect(self._on_vault_mode_changed)
        sb.llm.model_status_changed.connect(self._on_model_status)
        sb.llm.context_truncated.connect(self._on_context_truncated)
        sb.speech.state_changed.connect(self._on_voice_state)
        sb.speech.transcribed.connect(self._on_transcribed)
        sb.speech.prompt_ready.connect(self._on_voice_prompt)
        sb.speech.utterance_ignored.connect(self._on_utterance_ignored)

    def _sync_service_state(self) -> None:
        """Read current service state directly.

        Services start before this panel exists, so signals such as
        model_loaded have already fired with nothing connected.
        """
        llm = self._sm.get("LLM")
        if llm is not None and getattr(llm, "model_loaded", False):
            self._on_model_loaded(llm.model_name)

        vault = self._sm.get("Vault")
        if vault is not None:
            self._on_vault_mode_changed(
                "private" if vault.is_private else "normal"
            )

    def apply_theme(self) -> None:
        """Recolour icons and re-render the transcript for the active theme."""
        p = theme.p
        self._attach_btn.setIcon(svg_icon("attach", p.fg_secondary))
        self._send_btn.setIcon(svg_icon("send", p.accent_fg))
        self._cancel_btn.setIcon(svg_icon("stop", p.danger))
        self._new_btn.setIcon(svg_icon("plus", p.accent_fg))
        self._del_btn.setIcon(svg_icon("trash", p.danger))
        for button in (
            self._attach_btn,
            self._send_btn,
            self._cancel_btn,
            self._new_btn,
            self._del_btn,
        ):
            button.setIconSize(QSize(16, 16))
        if self._current_conv_id:
            self._load_conversation(self._current_conv_id)

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def _restore_session(self) -> None:
        """Show saved conversations; only create one if none exist."""
        llm = self._sm.get("LLM")
        existing = llm.list_conversations() if llm else []
        if not existing:
            self.new_conversation()
            return
        newest = max(existing, key=lambda c: c.modified)
        self._current_conv_id = newest.id
        self._update_conv_list()
        self._load_conversation(newest.id)

    def new_conversation(self) -> None:
        llm = self._sm.get("LLM")
        if llm:
            self._current_conv_id = llm.create_conversation()
            self._update_conv_list()
            self._chat_display.clear()
            self._append_system("New conversation started.")

    def _update_conv_list(self) -> None:
        """Rebuild the sidebar, newest first, preserving the selection."""
        llm = self._sm.get("LLM")
        if not llm:
            return
        selected = self._current_conv_id
        self._conv_list.blockSignals(True)
        self._conv_list.clear()
        convs = sorted(
            llm.list_conversations(), key=lambda c: c.modified, reverse=True
        )
        for conv in convs:
            label = f"[P] {conv.title}" if conv.private else conv.title
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, conv.id)
            if conv.private:
                item.setToolTip("Private — not saved to disk")
            self._conv_list.addItem(item)
            if conv.id == selected:
                self._conv_list.setCurrentItem(item)
        self._conv_list.blockSignals(False)

    def _on_conv_selected(self, current: QListWidgetItem, previous) -> None:
        if current:
            self._load_conversation(current.data(Qt.ItemDataRole.UserRole))

    def _load_conversation(self, conv_id: str) -> None:
        self._current_conv_id = conv_id
        self._chat_display.clear()
        llm = self._sm.get("LLM")
        if not llm:
            return
        conv = llm.get_conversation(conv_id)
        if not conv:
            return
        if not conv.messages:
            self._append_system("New conversation started.")
            return
        for msg in conv.messages:
            self._append(msg.role.value, msg.content)

    def _delete_conversation(self) -> None:
        current = self._conv_list.currentItem()
        if not current:
            return
        conv_id = current.data(Qt.ItemDataRole.UserRole)
        llm = self._sm.get("LLM")
        if llm:
            llm.delete_conversation(conv_id)
            self._current_conv_id = ""
            self._update_conv_list()
            self._chat_display.clear()
            remaining = llm.list_conversations()
            if remaining:
                newest = max(remaining, key=lambda c: c.modified)
                self._load_conversation(newest.id)
                self._update_conv_list()
            else:
                self.new_conversation()

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def _send_message(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        if not self._current_conv_id:
            self.new_conversation()

        attachments = list(self._attachment_paths)
        self._input.clear()

        shown = text
        if attachments:
            names = ", ".join(Path(p).name for p in attachments)
            shown = f"{text}\n\n[attached: {names}]"
        self._append("user", shown)

        llm = self._sm.get("LLM")
        if llm:
            llm.send_message(self._current_conv_id, text, attachments)

        self._clear_attachments()

    def _cancel_generation(self) -> None:
        llm = self._sm.get("LLM")
        if llm:
            llm.cancel_generation()

    def _show_input_context_menu(self, pos) -> None:
        """Right-click menu with spelling suggestions, like a browser."""
        menu = self._input.createStandardContextMenu()
        text = self._input.toPlainText()
        cursor_pos = self._input.cursorForPosition(pos).position()
        word, start, end = word_at_cursor(text, cursor_pos)

        def replace(suggestion: str) -> None:
            cursor = self._input.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            cursor.insertText(suggestion)

        def add(w: str) -> None:
            self._speller.add_word(w)
            if self._input_highlighter is not None:
                self._input_highlighter.rehighlight()

        build_menu_entries(menu, self._speller, word, replace, add)
        menu.exec(self._input.mapToGlobal(pos))

    def _attach_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach Files", "", "All Files (*)"
        )
        for path in paths:
            if path and path not in self._attachment_paths:
                self._attachment_paths.append(path)
        self._refresh_attachments_label()

    def _clear_attachments(self) -> None:
        self._attachment_paths.clear()
        self._refresh_attachments_label()

    def _refresh_attachments_label(self) -> None:
        """Show what will be sent, and warn when a file cannot be read.

        The model is text-only, so flagging an unreadable file here - before
        sending - is the difference between an informed user and one who
        trusts an answer about an image the model never saw.
        """
        if not self._attachment_paths:
            self._attachments_label.hide()
            return

        from ai_companion.services.attachment_reader import extract

        parts: list[str] = []
        for path in self._attachment_paths:
            result = extract(path)
            name = Path(path).name
            if result.ok:
                size = f"{result.char_count:,} chars"
                if result.truncated:
                    size += ", truncated"
                parts.append(f"{name} ({size})")
            else:
                parts.append(f"{name} - NOT READABLE")
        self._attachments_label.setText(
            "attached: " + "  |  ".join(parts) + "    (Esc to clear)"
        )
        self._attachments_label.show()

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _on_chunk(self, conv_id: str, chunk: str) -> None:
        if conv_id == self._current_conv_id:
            self._streaming_text += chunk
            self._update_streaming_display()

    def _on_gen_started(self, conv_id: str) -> None:
        if conv_id != self._current_conv_id:
            return
        self._streaming_text = ""
        self._send_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._input.setEnabled(False)
        self._chat_display.append(render_spacer())
        self._chat_display.moveCursor(QTextCursor.MoveOperation.End)
        self._stream_anchor = self._chat_display.textCursor().position()

    STICK_TO_BOTTOM_PX = 40

    def _is_at_bottom(self) -> bool:
        """True when the view is scrolled to (or very near) the bottom.

        A small tolerance matters: after inserting a chunk the maximum grows,
        so an exact comparison would read as 'not at bottom' and strand the
        user one line up.
        """
        bar = self._chat_display.verticalScrollBar()
        return bar.value() >= bar.maximum() - self.STICK_TO_BOTTOM_PX

    def _update_streaming_display(self) -> None:
        """Replace only the text generated since the stream anchor.

        Scrolling is preserved deliberately. Calling setTextCursor() moves the
        visible cursor, and ensureCursorVisible() then forces the view back to
        the bottom - so scrolling up mid-generation used to be impossible, the
        next chunk yanked you straight back down. Now the view only follows
        the output when the user was already at the bottom.
        """
        if self._stream_anchor is None:
            return

        bar = self._chat_display.verticalScrollBar()
        was_at_bottom = self._is_at_bottom()
        prev_scroll = bar.value()

        # Edit through a standalone cursor so the widget's own cursor - and
        # therefore any active selection - is left untouched.
        cursor = QTextCursor(self._chat_display.document())
        cursor.setPosition(self._stream_anchor)
        cursor.movePosition(
            QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor
        )
        cursor.removeSelectedText()
        cursor.insertHtml(
            render_message(
                "assistant", self._streaming_text, theme.p, theme.m
            )
        )

        if was_at_bottom:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(prev_scroll)

    def _maybe_speak_reply(self, text: str) -> None:
        """Speak the answer only if the question arrived by voice."""
        if not getattr(self, "_voice_turn", False):
            return
        self._voice_turn = False
        voice = self._sm.get("Voice")
        if voice is not None and voice.enabled:
            voice.on_reply_complete(text)

    def _on_gen_complete(self, conv_id: str) -> None:
        self._update_conv_list()
        if conv_id != self._current_conv_id:
            return
        self._send_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._input.setEnabled(True)
        self._input.setFocus()
        reply = self._streaming_text
        self._streaming_text = ""
        self._stream_anchor = None
        self._maybe_speak_reply(reply)

    def _on_gen_cancelled(self, conv_id: str) -> None:
        if conv_id != self._current_conv_id:
            return
        self._send_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._input.setEnabled(True)
        self._streaming_text = ""
        self._stream_anchor = None
        self._append_system("Generation cancelled.")

    def _on_gen_error(self, conv_id: str, error: str) -> None:
        if conv_id != self._current_conv_id:
            return
        self._send_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._input.setEnabled(True)
        self._stream_anchor = None
        self._append_system(f"Error: {error}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _on_model_loaded(self, model_name: str) -> None:
        self._model_label.setText(f"MODEL: {model_name}")
        self._model_label.setObjectName("StatusOk")
        self._restyle(self._model_label)

    def _toggle_voice(self) -> None:
        """Turn hands-free listening on or off."""
        voice = self._sm.get("Voice")
        if voice is None:
            self._append_system("Voice service unavailable.")
            self._voice_btn.setChecked(False)
            return
        ok, error = voice.toggle()
        if not ok:
            self._voice_btn.setChecked(False)
            self._append_system(f"Could not start voice: {error}")

    def _on_voice_state(self, state: str) -> None:
        """Reflect the voice loop's real state.

        Driven entirely by the service. A voice UI that guesses its own state
        is worse than none - the user cannot tell "not listening" from
        "listening but did not hear you".
        """
        labels = {
            "off": ("VOICE OFF", "VoiceBadge"),
            "idle": ("LISTENING", "VoiceBadgeOn"),
            "capturing": ("HEARING YOU", "VoiceBadgeActive"),
            "thinking": ("THINKING", "VoiceBadgeActive"),
            "speaking": ("SPEAKING", "VoiceBadgeActive"),
        }
        text, style = labels.get(state, ("VOICE OFF", "VoiceBadge"))
        self._voice_btn.blockSignals(True)
        self._voice_btn.setChecked(state != "off")
        self._voice_btn.blockSignals(False)
        self._voice_btn.setText(text)
        self._voice_btn.setObjectName(style)
        self._restyle(self._voice_btn)

    def _on_transcribed(self, text: str) -> None:
        """Show what was heard while the utterance is being judged.

        This is transient. If the utterance is accepted the placeholder is
        restored when the turn is sent; if it is ignored the reason replaces
        it. Leaving "heard: ..." sitting in the box looked like the app had
        typed the message and then refused to send it.
        """
        self._input.setPlaceholderText(f"heard: {text[:60]}")

    def _on_utterance_ignored(self, heard: str, reason: str) -> None:
        """Say why a clearly-heard utterance was not acted on."""
        self._append_system(f'Heard "{heard}" but ignored it: {reason}.')
        self._input.setPlaceholderText(
            f'Ignored - {reason}'
        )
        # Restore the normal prompt shortly after, so the hint does not
        # linger and read as stuck state.
        QTimer.singleShot(
            6000,
            lambda: self._input.setPlaceholderText(
                "Message your local model\u2026"
            ),
        )

    def _on_voice_prompt(self, conv_id: str, prompt: str) -> None:
        """A spoken prompt passed the wake word - send it as a normal turn."""
        if not self._current_conv_id:
            self.new_conversation()
        self._voice_turn = True
        self._input.setPlaceholderText("Message your local model\u2026")
        self._append("user", prompt)
        llm = self._sm.get("LLM")
        if llm is not None:
            llm.send_message(self._current_conv_id, prompt)

    def _on_context_truncated(self, conv_id: str, dropped: int) -> None:
        """Say when history was dropped to fit the context window.

        Silently forgetting the middle of a conversation makes the model look
        like it has amnesia; stating it turns a mystery into a known limit.
        """
        if conv_id != self._current_conv_id:
            return
        plural = "s" if dropped != 1 else ""
        self._append_system(
            f"{dropped} earlier message{plural} omitted to fit the context "
            f"window. Raise Context Length in Settings, or start a new chat."
        )

    def _on_model_status(self, status: str, detail: str) -> None:
        """Surface load failures where the user is actually looking.

        The status bar truncates and is easy to miss; a model that silently
        fails to load looks like a broken app.
        """
        if status == "error":
            self._model_label.setText("MODEL LOAD FAILED")
            self._model_label.setObjectName("StatusError")
            self._restyle(self._model_label)
            self._append_system(f"Model load failed: {detail}")
        elif status == "loading":
            self._model_label.setText("LOADING MODEL...")
            self._model_label.setObjectName("StatusWarn")
            self._restyle(self._model_label)

    def _on_model_unloaded(self) -> None:
        self._model_label.setText("NO MODEL LOADED")
        self._model_label.setObjectName("StatusWarn")
        self._restyle(self._model_label)

    def _toggle_private(self) -> None:
        """Flip the vault between normal and private from the header."""
        from ai_companion.models.config_models import VaultMode

        vault = self._sm.get("Vault")
        if vault is None:
            return
        target = VaultMode.NORMAL if vault.is_private else VaultMode.PRIVATE
        vault.set_mode(target)
        # The signal drives the repaint, so no local state to update here.

    def _on_vault_mode_changed(self, mode: str) -> None:
        """Repaint the badge. Driven by the service, never set optimistically.

        If the toggle failed, the badge must keep showing the real state - a
        badge that lies about privacy is worse than no badge.
        """
        private = mode == "private"
        self._privacy_btn.blockSignals(True)
        self._privacy_btn.setChecked(private)
        self._privacy_btn.blockSignals(False)

        if private:
            self._privacy_btn.setText("PRIVATE · NOT SAVED")
            self._privacy_btn.setObjectName("PrivateBadge")
            self._privacy_btn.setToolTip(
                "Private mode: this chat is not written to disk.\n"
                "Click to resume saving."
            )
        else:
            self._privacy_btn.setText("SAVING")
            self._privacy_btn.setObjectName("SavingBadge")
            self._privacy_btn.setToolTip(
                "Conversations are being saved.\n"
                "Click to switch to private mode."
            )
        self._restyle(self._privacy_btn)
        self._update_conv_list()

    @staticmethod
    def _restyle(widget: QWidget) -> None:
        """Force a re-evaluation of the stylesheet after an objectName change."""
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def _append(self, role: str, text: str) -> None:
        bar = self._chat_display.verticalScrollBar()
        was_at_bottom = self._is_at_bottom()
        prev_scroll = bar.value()

        self._chat_display.append(render_spacer())
        self._chat_display.append(
            render_message(role, text, theme.p, theme.m)
        )

        if was_at_bottom:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(prev_scroll)

    def _append_system(self, text: str) -> None:
        self._append("system", text)
