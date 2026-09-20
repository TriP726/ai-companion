"""Central service manager — owns all services and the signal bus."""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject

from ai_companion.config import ConfigManager
from ai_companion.infrastructure.signal_bus import SignalBus
from ai_companion.infrastructure.base_service import BaseService

logger = logging.getLogger(__name__)


class ServiceManager(QObject):
    """Registers, initializes, and manages the lifecycle of all services."""

    def __init__(self, config_manager: ConfigManager) -> None:
        super().__init__()
        self._config_manager = config_manager
        self._signal_bus = SignalBus(self)
        self._services: dict[str, BaseService] = {}

    @property
    def signal_bus(self) -> SignalBus:
        return self._signal_bus

    @property
    def config(self):
        return self._config_manager.config

    def register(self, service: BaseService) -> None:
        """Register a service. Does not start it."""
        name = service.service_name
        if name in self._services:
            raise ValueError(f"Service '{name}' already registered")
        service._service_manager = self
        self._services[name] = service
        logger.info("Registered service: %s", name)

    @property
    def services(self) -> dict:
        """Read-only view of registered services, in registration order.

        Exposed so callers (the boot screen) can iterate services without
        reaching into the private dict.
        """
        return dict(self._services)

    def get(self, name: str) -> BaseService | None:
        return self._services.get(name)

    def start_all(self) -> None:
        """Start all registered services in registration order.

        Idempotent: a service that is already running is skipped. The boot
        screen starts services itself as each line appears, then calls this
        as a safety net, so double-starting must be harmless.
        """
        for name, service in self._services.items():
            if service.is_running:
                continue
            try:
                logger.info("Starting %s...", name)
                service.start()
            except Exception:
                logger.exception("Failed to start %s", name)
                raise

    def stop_all(self) -> None:
        """Stop all services in reverse registration order."""
        for name in reversed(list(self._services.keys())):
            try:
                logger.info("Stopping %s...", name)
                self._services[name].stop()
            except Exception:
                logger.exception("Error stopping %s", name)

    def service_status(self) -> dict[str, bool]:
        """Return running status of all services."""
        return {name: svc.is_running for name, svc in self._services.items()}
