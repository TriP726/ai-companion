"""Main application window — icon rail navigation with a stacked content area."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from ai_companion.ui.nav_rail import NavRail
from ai_companion.ui.theme import ThemeMode, theme

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class _GridCanvas(QWidget):
    """Central widget that paints the HUD grid as its own background.

    Painting in the container rather than adding a child widget avoids any
    stacking-order or geometry-tracking problems: children simply draw on top.
    """

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
        from PySide6.QtCore import QPointF

        from ai_companion.ui.theme import theme as _t

        p = _t.p
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(p.bg_base))
        painter.setPen(QPen(QColor(p.grid), 1))
        step = 34
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)
        glow = QRadialGradient(
            QPointF(self.width() / 2, self.height() / 2),
            max(self.width(), self.height()) * 0.8,
        )
        glow.setColorAt(0.0, QColor(0, 0, 0, 0))
        glow.setColorAt(1.0, QColor(0, 0, 0, 150))
        painter.fillRect(self.rect(), QBrush(glow))
        painter.end()


class MainWindow(QMainWindow):
    """The main application window."""

    def __init__(self, service_manager: ServiceManager) -> None:
        super().__init__()
        self._sm = service_manager
        self.setWindowTitle("AI COMPANION // LOCAL")
        # Lowered from 1200x800: the old value exceeded some 1080p work areas
        # once Windows display scaling was applied, clipping the right edge.
        self.setMinimumSize(1000, 680)
        self.resize(1320, 860)

        self._setup_menu()
        self._setup_body()
        self._setup_status_bar()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        new_chat = QAction("New &Chat", self)
        new_chat.setShortcut("Ctrl+N")
        new_chat.triggered.connect(self._new_chat)
        file_menu.addAction(new_chat)
        file_menu.addSeparator()

        settings_action = QAction("&Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)
        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu_bar.addMenu("&View")
        self._theme_action = QAction("Toggle &Theme", self)
        self._theme_action.setShortcut("Ctrl+T")
        self._theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self._theme_action)
        view_menu.addSeparator()

        # Only destinations that actually exist are listed. 3D Sandbox is
        # deliberately absent: it has no UI yet, and a menu entry that
        # silently does nothing is worse than no entry.
        for index, name in enumerate(
            ["Chat", "Memory", "Graph", "Workspace", "Camera", "Mind Map"]
        ):
            action = QAction(name, self)
            action.setShortcut(f"Ctrl+{index + 1}")
            action.triggered.connect(
                lambda _=False, i=index: self._switch_to(i)
            )
            view_menu.addAction(action)

    def _setup_body(self) -> None:
        from ai_companion.ui.camera_panel import CameraPanel
        from ai_companion.ui.chat_panel import ChatPanel
        from ai_companion.ui.memory_panel import MemoryPanel
        from ai_companion.ui.graph_panel import GraphPanel
        from ai_companion.ui.mind_map_panel import MindMapPanel
        from ai_companion.ui.workspace_panel import WorkspacePanel

        from ai_companion.ui.hud import (
            GridBackground,
            PulseBar,
            ReactorCore,
            TelemetryStrip,
        )
        from ai_companion.ui.theme import theme as _theme

        m = _theme.m

        # The grid is a sibling behind everything, not a parent, so no panel
        # has to be transparent for it to show through at the edges.
        container = _GridCanvas()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- telemetry header -------------------------------------
        header = QWidget()
        header.setObjectName("HudHeader")
        header.setFixedHeight(42)
        head_row = QHBoxLayout(header)
        head_row.setContentsMargins(m.space_4, 0, m.space_4, 0)
        head_row.setSpacing(m.space_4)

        title = QLabel("A . I .   C O M P A N I O N")
        title.setObjectName("HudTitle")
        head_row.addWidget(title)

        self._telemetry = TelemetryStrip()
        self._telemetry.add_field("model", "MODEL", "OFFLINE")
        self._telemetry.add_field("mode", "VAULT", "--")
        self._telemetry.add_field("panel", "VIEW", "CHAT")
        self._telemetry.add_stretch()
        head_row.addWidget(self._telemetry, 1)
        outer.addWidget(header)

        self._pulse = PulseBar()
        outer.addWidget(self._pulse)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        outer.addWidget(body, 1)

        self._rail = NavRail()
        self._reactor = ReactorCore()
        self._reactor.clicked.connect(self._open_settings)
        self._rail.add_widget(self._reactor)
        self._rail.navigated.connect(self._switch_to)
        layout.addWidget(self._rail)

        self._stack = QStackedWidget()
        self._chat_panel = ChatPanel(self._sm)
        self._memory_panel = MemoryPanel(self._sm)
        self._graph_panel = GraphPanel(self._sm)
        self._workspace_panel = WorkspacePanel(self._sm)
        self._camera_panel = CameraPanel(self._sm)
        self._mind_map_panel = MindMapPanel(self._sm)

        for panel, icon, label in [
            (self._chat_panel, "chat", "CHAT"),
            (self._memory_panel, "memory", "MEM"),
            (self._graph_panel, "graph", "GRAPH"),
            (self._workspace_panel, "workspace", "WORK"),
            (self._camera_panel, "camera", "CAM"),
            (self._mind_map_panel, "graph", "MAP"),
        ]:
            self._stack.addWidget(panel)
            self._rail.add_destination(icon, label)

        self._rail.add_stretch()
        self._rail.add_action("settings", "CONFIG", self._open_settings)

        layout.addWidget(self._stack, 1)
        self.setCentralWidget(container)
        self._connect_hud_signals()

    def _connect_hud_signals(self) -> None:
        """Drive the reactor and pulse bar from real generation state.

        Motion is event-driven on purpose: on an integrated-GPU machine doing
        CPU inference, an always-on animation steals cycles from the tokens
        the user is waiting for.
        """
        sb = self._sm.signal_bus
        sb.llm.generation_started.connect(lambda _: self._set_busy(True))
        sb.llm.generation_complete.connect(lambda *_: self._set_busy(False))
        sb.llm.generation_cancelled.connect(lambda *_: self._set_busy(False))
        sb.llm.model_loaded.connect(
            lambda name: self._telemetry.set_value("model", name.upper()[:22])
        )
        sb.llm.model_unloaded.connect(
            lambda: self._telemetry.set_value("model", "OFFLINE")
        )
        sb.vault.mode_changed.connect(
            lambda mode: self._telemetry.set_value("mode", mode.upper())
        )

        vault = self._sm.get("Vault")
        if vault is not None:
            self._telemetry.set_value(
                "mode", "PRIVATE" if vault.is_private else "NORMAL"
            )
        llm = self._sm.get("LLM")
        if llm is not None and getattr(llm, "model_loaded", False):
            self._telemetry.set_value("model", llm.model_name.upper()[:22])

    def show_notice(self, text: str) -> None:
        """Surface a one-off message in the chat transcript and status bar."""
        try:
            self._chat_panel._append_system(text)
        except Exception:  # noqa: BLE001 - a notice must never break startup
            pass
        self._status_text.setText(text[:90])

    def _set_busy(self, busy: bool) -> None:
        self._reactor.set_active(busy)
        self._pulse.set_active(busy)



    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_text = QLabel("READY")
        self._status_text.setObjectName("MonoText")
        self._status_bar.addWidget(self._status_text)

        self._mode_text = QLabel("")
        self._mode_text.setObjectName("MonoText")
        self._status_bar.addPermanentWidget(self._mode_text)
        self._refresh_mode_label()

    def _connect_signals(self) -> None:
        sb = self._sm.signal_bus
        sb.service.status_changed.connect(self._on_service_status)
        sb.service.error_occurred.connect(self._on_service_error)
        sb.llm.model_status_changed.connect(self._on_model_status)
        sb.vault.mode_changed.connect(lambda _m: self._refresh_mode_label())

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def toggle_theme(self) -> None:
        """Kept for the Ctrl+T shortcut and the View menu.

        Light mode was dropped in the HUD redesign - a pale holographic
        interface is incoherent - so this now only repaints. The shortcut is
        retained rather than removed so existing muscle memory does not hit a
        dead key, and so nothing that calls it breaks.
        """
        self.apply_theme()
        self._status_text.setText("HUD THEME ACTIVE")

    def apply_theme(self) -> None:
        app = self.window().windowHandle()  # noqa: F841 - kept for clarity
        from PySide6.QtWidgets import QApplication

        instance = QApplication.instance()
        if instance is not None:
            instance.setStyleSheet(theme.stylesheet())
        self._rail.refresh_icons()
        for panel in (
            self._chat_panel,
            self._memory_panel,
            self._graph_panel,
            self._workspace_panel,
            self._camera_panel,
            self._mind_map_panel,
        ):
            if hasattr(panel, "apply_theme"):
                panel.apply_theme()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _refresh_mode_label(self) -> None:
        vault = self._sm.get("Vault")
        if vault is None:
            self._mode_text.setText("")
            return
        if vault.is_private:
            self._mode_text.setText("PRIVATE MODE")
            self._mode_text.setStyleSheet(f"color: {theme.p.private};")
        else:
            self._mode_text.setText("NORMAL MODE")
            self._mode_text.setStyleSheet(f"color: {theme.p.fg_muted};")

    def _on_service_status(self, service_name: str, status: str) -> None:
        self._status_text.setText(f"{service_name.upper()}: {status.upper()}")

    def _on_service_error(self, service_name: str, error: str) -> None:
        self._status_text.setText(f"! {service_name.upper()}: {error}")
        self._status_text.setStyleSheet(f"color: {theme.p.danger};")

    def _on_model_status(self, status: str, detail: str) -> None:
        if status == "loaded":
            self._status_text.setText(f"MODEL: {detail}")
            self._status_text.setStyleSheet(f"color: {theme.p.success};")

    def _new_chat(self) -> None:
        self._switch_to(0)
        self._chat_panel.new_conversation()

    def _open_settings(self) -> None:
        from ai_companion.ui.settings_panel import SettingsPanel

        panel = SettingsPanel(self._sm, self)
        panel.exec()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Release the camera before the window goes away.

        Without this the OS keeps the device marked in-use after exit, and
        the camera light can stay on.
        """
        try:
            self._camera_panel.shutdown()
        except Exception:  # noqa: BLE001 - never block shutdown
            pass
        super().closeEvent(event)

    PANEL_NAMES = ("CHAT", "MEMORY", "GRAPH", "WORKSPACE", "CAMERA",
                   "MINDMAP")

    def _switch_to(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self._rail.set_current(index)
        self._rail.refresh_icons()
        if 0 <= index < len(self.PANEL_NAMES):
            self._telemetry.set_value("panel", self.PANEL_NAMES[index])
        # Event-driven fade: costs one short animation on an explicit user
        # action, nothing while idle.
        from ai_companion.ui.hud import fade_in

        current = self._stack.currentWidget()
        if current is not None:
            fade_in(current)
