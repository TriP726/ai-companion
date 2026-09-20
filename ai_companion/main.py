"""AI Companion — main entry point.

Local-first desktop AI companion with:
- Local LLM chat (llama.cpp)
- Private vault with file memory
- Owner-reviewed long-term memories
- Visual knowledge graph
- Manual camera access (no recording)
- Local image generation (ComfyUI/SDXL)
- Local speech I/O
- Permission-controlled file workspace
- Sandboxed code execution
- 3D model sandbox
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AICompanion")


def create_app() -> QApplication:
    """Create and configure the QApplication."""
    app = QApplication(sys.argv)
    app.setApplicationName("AI Companion")
    app.setOrganizationName("AICompanion")
    app.setApplicationVersion("0.1.0")

    # Global stylesheet comes from the token-based theme engine so that a
    # light/dark toggle never requires touching individual widgets.
    from ai_companion.ui.theme import theme

    app.setStyleSheet(theme.stylesheet())

    return app


MAX_DATA_SNAPSHOTS = 10


def _snapshot_data(config) -> None:
    """Copy the data directory to a timestamped folder, pruning old ones.

    Cheap: these are small JSON files. The point is that a corrupted or
    misdirected store is always recoverable from the previous launch.
    """
    import shutil
    from datetime import datetime
    from pathlib import Path

    data_dir = Path(config.data_dir)
    if not data_dir.is_dir() or not any(data_dir.iterdir()):
        return

    root = Path(__file__).resolve().parent.parent
    backups = root / "data_backups"
    backups.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backups / stamp
    if target.exists():
        return
    shutil.copytree(data_dir, target)
    logger.info("Data snapshot written to %s", target)

    # Keep only the most recent few so this cannot grow without bound.
    existing = sorted(
        (p for p in backups.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    for stale in existing[:-MAX_DATA_SNAPSHOTS]:
        shutil.rmtree(stale, ignore_errors=True)


def main() -> int:
    """Application entry point."""
    logger.info("Starting AI Companion...")

    app = create_app()

    # Import after QApplication is created (Qt requirement)
    from ai_companion.config import ConfigManager
    from ai_companion.infrastructure.service_manager import ServiceManager
    from ai_companion.services.llm_service import LlmService
    from ai_companion.services.vault_service import VaultService
    from ai_companion.services.memory_service import MemoryService
    from ai_companion.services.graph_service import GraphService
    from ai_companion.services.mind_map_service import MindMapService
    from ai_companion.services.camera_service import CameraService
    from ai_companion.services.image_gen_service import ImageGenService
    from ai_companion.services.speech_service import SpeechService
    from ai_companion.services.voice_service import VoiceService
    from ai_companion.services.file_workspace import FileWorkspaceService
    from ai_companion.services.code_workspace import CodeWorkspaceService
    from ai_companion.services.sandbox3d_service import Sandbox3DService
    from ai_companion.ui.main_window import MainWindow

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load()

    # Automatic data snapshot on every launch, kept for the last few runs.
    # The user lost memories once to a storage path that silently pointed at a
    # deleted temp folder; a rolling on-disk copy means that class of failure
    # can never be unrecoverable again.
    try:
        _snapshot_data(config)
    except Exception:
        logger.exception("Could not snapshot data directory")

    # Create service manager and register services
    sm = ServiceManager(config_manager)

    sm.register(LlmService(config, sm.signal_bus))
    sm.register(VaultService(config, sm.signal_bus))
    sm.register(MemoryService(config, sm.signal_bus))
    sm.register(GraphService(config, sm.signal_bus))
    sm.register(MindMapService(config, sm.signal_bus))
    sm.register(CameraService(config, sm.signal_bus))
    sm.register(ImageGenService(config, sm.signal_bus))
    sm.register(SpeechService(config, sm.signal_bus))
    sm.register(VoiceService(config, sm.signal_bus))
    sm.register(FileWorkspaceService(config, sm.signal_bus))
    sm.register(CodeWorkspaceService(config, sm.signal_bus))
    sm.register(Sandbox3DService(config, sm.signal_bus))

    # Boot sequence. Services are started BY the boot screen, one line at a
    # time, so the log reports real progress instead of animating a fake bar.
    from ai_companion.ui.boot import BootScreen, build_steps

    boot = BootScreen(build_steps(sm, config))
    boot.resize(1100, 720)
    boot.setWindowTitle("AI Companion")
    boot.show()

    window: dict = {}

    def _on_boot_finished() -> None:
        # Any service the boot screen could not reach still gets a chance;
        # start_all() skips anything already running.
        try:
            sm.start_all()
        except Exception:
            logger.exception("Failed to start services")

        main_window = MainWindow(sm)
        window["w"] = main_window
        main_window.show()

        # Tell the user about anything the config migration changed under
        # them. A silent mode flip is exactly the kind of surprise this
        # whole change exists to stop.
        for note in config_manager.migration_notes:
            main_window.show_notice(note)
        boot.close()
        logger.info("AI Companion ready")

    boot.finished.connect(_on_boot_finished)
    boot.start()

    # Run event loop
    result = app.exec()

    # Clean shutdown
    logger.info("Shutting down...")
    sm.stop_all()
    config_manager.save()

    return result


if __name__ == "__main__":
    sys.exit(main())
