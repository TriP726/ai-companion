"""Base class for all services."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

if TYPE_CHECKING:
    from ai_companion.infrastructure.signal_bus import SignalBus
    from ai_companion.models.config_models import AppConfig


class BaseService(QObject):
    """Abstract base for all application services.

    Each service receives config, signal_bus at init, and implements
    start()/stop() lifecycle methods.
    """

    service_name: str = "BaseService"

    def __init__(
        self,
        config: AppConfig,
        signal_bus: SignalBus,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._signal_bus = signal_bus
        self._logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        # Set by ServiceManager.register(); lets a service query its peers
        # without importing them (avoids circular imports).
        self._service_manager: object | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Initialize and start the service."""
        self._running = True
        self._logger.info("%s started", self.service_name)
        self._signal_bus.service.status_changed.emit(
            self.service_name, "running"
        )

    def stop(self) -> None:
        """Cleanly shut down the service."""
        self._running = False
        self._logger.info("%s stopped", self.service_name)
        self._signal_bus.service.status_changed.emit(
            self.service_name, "stopped"
        )

    def emit_error(self, message: str) -> None:
        """Emit an error through the signal bus."""
        self._logger.error("%s: %s", self.service_name, message)
        self._signal_bus.service.error_occurred.emit(self.service_name, message)

    def emit_status(self, message: str) -> None:
        """Emit a status update."""
        self._signal_bus.service.status_changed.emit(self.service_name, message)
