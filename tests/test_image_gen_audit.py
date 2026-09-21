"""Regression tests for the ComfyUI loopback-binding fix — 2026-09-20.

image_gen_service.py used to pass config.image_gen.host straight through
to both the worker's `--listen` flag and the outbound request URL. The
class docstring claimed "binds to 127.0.0.1 only", but nothing in the
code enforced that — it was only true because the config default
happened to be 127.0.0.1. A bad config value (typo, a bad patch, a
future settings field) could have put the worker on the network with
zero warning.

Fixed by pinning both the bind and connect address to a hardcoded
_LOOPBACK_HOST constant, ignoring config.image_gen.host entirely for
anything network-relevant. These tests pin that behaviour: even a
config deliberately set to "0.0.0.0" must not affect the worker.

Standalone file, not added to the existing test_services.py — there was
no prior ImageGenService test class to conflict with.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ai_companion.infrastructure.signal_bus import SignalBus
from ai_companion.models.config_models import AppConfig
from ai_companion.services.image_gen_service import ImageGenService, _LOOPBACK_HOST


@pytest.fixture
def tmp_env(tmp_path):
    config = AppConfig().relocate(tmp_path)
    os.makedirs(config.data_dir, exist_ok=True)
    return config, tmp_path


@pytest.fixture
def signal_bus():
    return SignalBus()


class TestWorkerAlwaysBindsToLoopback:
    def test_listen_flag_ignores_hostile_config_host(self, tmp_env, signal_bus, tmp_path):
        config, _ = tmp_env
        # A deliberately hostile value — the kind a bad patch or a typo
        # in a future settings field could produce.
        config.image_gen.host = "0.0.0.0"
        config.image_gen.comfyui_path = str(tmp_path)  # just needs to exist

        svc = ImageGenService(config, signal_bus)
        svc.start()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            assert svc.start_worker() is True

        cmd = mock_popen.call_args[0][0]
        assert "--listen" in cmd
        listen_value = cmd[cmd.index("--listen") + 1]
        assert listen_value == _LOOPBACK_HOST
        assert listen_value != "0.0.0.0"

        svc.stop()

    def test_emits_a_warning_when_config_host_is_ignored(self, tmp_env, signal_bus, tmp_path):
        config, _ = tmp_env
        config.image_gen.host = "0.0.0.0"
        config.image_gen.comfyui_path = str(tmp_path)

        svc = ImageGenService(config, signal_bus)
        svc.start()

        statuses = []
        signal_bus.service.status_changed.connect(
            lambda name, msg: statuses.append((name, msg))
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            svc.start_worker()

        svc.stop()
        assert any(
            "0.0.0.0" in msg and "ignored" in msg for _name, msg in statuses
        ), "expected a status message explaining the ignored config value"

    def test_default_config_still_works_silently(self, tmp_env, signal_bus, tmp_path):
        """The already-correct, common case must not regress or start warning."""
        config, _ = tmp_env
        assert config.image_gen.host == _LOOPBACK_HOST  # sanity on the default
        config.image_gen.comfyui_path = str(tmp_path)

        svc = ImageGenService(config, signal_bus)
        svc.start()

        statuses = []
        signal_bus.service.status_changed.connect(
            lambda name, msg: statuses.append((name, msg))
        )

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            assert svc.start_worker() is True

        cmd = mock_popen.call_args[0][0]
        listen_value = cmd[cmd.index("--listen") + 1]
        assert listen_value == _LOOPBACK_HOST
        assert not any("ignored" in msg for _name, msg in statuses)

        svc.stop()


class TestSubmitRequestAlwaysConnectsToLoopback:
    def test_request_url_ignores_hostile_config_host(self, tmp_env, signal_bus):
        config, _ = tmp_env
        config.image_gen.host = "0.0.0.0"

        svc = ImageGenService(config, signal_bus)
        svc.start()
        svc._worker_running = True  # simulate a worker already running

        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["url"] = req.full_url
            resp = MagicMock()
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            resp.read.return_value = b'{"prompt_id": "abc"}'
            return resp

        request = {"id": "test-id", "seed": -1, "steps": 20, "status": "queued"}
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            svc._submit_to_worker(request)

        assert captured["url"].startswith(f"http://{_LOOPBACK_HOST}:")
        assert "0.0.0.0" not in captured["url"]
        svc.stop()
