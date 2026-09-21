"""Regression tests for the vault privacy/security audit — 2026-09-20.

Two real gaps, found by re-checking VaultService against the guarantees in
its own class docstring and against how PathValidator is used everywhere
else in that file:

1. remove_file() wrote the on-disk index unconditionally, even while the
   Vault was in PRIVATE mode. Every other persistent service in this app
   suppresses disk writes while Private; this one didn't.

2. get_file_path() built its path with plain string concatenation and an
   `.exists()` check — no PathValidator, no containment check, no
   traversal protection at all, unlike import_file() and remove_file() in
   the same class. Nothing in the shipped UI calls it yet, but it's a
   public method, so it gets the same guarantee as the rest of the file.

Both are fixed in vault_service.py. These tests pin the fixed behaviour
so neither regresses silently.

This is a standalone file (not added to the existing test_services.py)
so it can be dropped straight into tests/ without hand-editing a
4,000+ line file.
"""
from __future__ import annotations

import json
import os

import pytest

from ai_companion.infrastructure.signal_bus import SignalBus
from ai_companion.models.config_models import AppConfig, VaultMode
from ai_companion.services.vault_service import VaultService


@pytest.fixture
def tmp_env(tmp_path):
    """Isolated config: every store path relocated under tmp_path."""
    config = AppConfig().relocate(tmp_path)
    os.makedirs(config.data_dir, exist_ok=True)
    return config, tmp_path


@pytest.fixture
def signal_bus():
    return SignalBus()


def _read_raw_index(vault_root) -> dict:
    """Read vault_index.json straight off disk, bypassing any in-memory cache."""
    path = vault_root / "vault_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestGetFilePathTraversal:
    """get_file_path() must reject any path that would escape the vault root."""

    def test_rejects_parent_traversal_to_a_real_file(self, tmp_env, signal_bus):
        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.NORMAL
        # Deliberately NOT adding tmp_path to approved_folders: only the
        # vault root itself (tmp_path/vault) is approved here, so `secret`
        # below is a real file that sits genuinely outside every approved
        # root. Adding tmp_path as approved would make the traversal land
        # somewhere legitimately accessible and defeat the point of the test.
        svc = VaultService(config, signal_bus)
        svc.start()

        # A file that genuinely exists outside the vault, so a broken
        # implementation would actually succeed in reaching it rather than
        # just failing to find something.
        secret = tmp_path / "secret.txt"
        secret.write_text("outside the vault")

        # Generous "../" count: resolving past the filesystem root just
        # clamps there on both POSIX and Windows, so this reliably lands
        # on `secret` regardless of how deeply nested tmp_path is.
        traversal = "../" * 20 + str(secret).lstrip("/\\")

        assert svc.get_file_path(traversal) is None
        svc.stop()

    def test_rejects_absolute_path_outside_vault(self, tmp_env, signal_bus, tmp_path_factory):
        config, _ = tmp_env
        config.vault.mode = VaultMode.NORMAL
        svc = VaultService(config, signal_bus)
        svc.start()

        elsewhere = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
        elsewhere.write_text("outside the vault")

        assert svc.get_file_path(str(elsewhere)) is None
        svc.stop()

    def test_still_resolves_legitimate_vault_files(self, tmp_env, signal_bus):
        """The fix must not break the normal, non-malicious path."""
        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.NORMAL
        config.vault.approved_folders.append(str(tmp_path))
        svc = VaultService(config, signal_bus)
        svc.start()

        src = tmp_path / "ok.txt"
        src.write_text("fine")
        vault_path = svc.import_file(str(src))
        assert vault_path is not None

        resolved = svc.get_file_path(vault_path)
        assert resolved is not None
        assert resolved.exists()
        svc.stop()

    def test_returns_none_for_nonexistent_file(self, tmp_env, signal_bus):
        config, _ = tmp_env
        svc = VaultService(config, signal_bus)
        svc.start()

        assert svc.get_file_path("files/general/nope.txt") is None
        svc.stop()


class TestRemoveFileRespectsPrivateMode:
    """remove_file() must delete immediately but defer the index write
    while the vault is Private, instead of writing it unconditionally."""

    def _import_in_normal_mode(self, tmp_env, signal_bus):
        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.NORMAL
        config.vault.approved_folders.append(str(tmp_path))
        svc = VaultService(config, signal_bus)
        svc.start()

        src = tmp_path / "doc.txt"
        src.write_text("content")
        vault_path = svc.import_file(str(src))
        assert vault_path is not None
        return svc, vault_path

    def test_file_is_actually_deleted_while_private(self, tmp_env, signal_bus):
        svc, vault_path = self._import_in_normal_mode(tmp_env, signal_bus)
        on_disk = svc.get_file_path(vault_path)
        assert on_disk is not None and on_disk.exists()

        svc.set_mode(VaultMode.PRIVATE, persist=False)
        assert svc.remove_file(vault_path) is True

        # Deletion is a destructive op the owner asked for — it must
        # happen immediately, regardless of mode.
        assert not on_disk.exists()
        svc.stop()

    def test_in_memory_view_is_consistent_immediately(self, tmp_env, signal_bus):
        svc, vault_path = self._import_in_normal_mode(tmp_env, signal_bus)
        svc.set_mode(VaultMode.PRIVATE, persist=False)
        svc.remove_file(vault_path)

        assert svc.get_file_path(vault_path) is None
        assert all(f["vault_path"] != vault_path for f in svc.list_files())
        svc.stop()

    def test_index_on_disk_is_not_touched_while_private(self, tmp_env, signal_bus):
        svc, vault_path = self._import_in_normal_mode(tmp_env, signal_bus)
        vault_root = svc._vault_root  # test-only reach-in, to inspect raw disk state
        before = _read_raw_index(vault_root)
        assert vault_path in before.get("items", {})

        svc.set_mode(VaultMode.PRIVATE, persist=False)
        svc.remove_file(vault_path)

        after = _read_raw_index(vault_root)
        assert vault_path in after.get("items", {}), (
            "vault_index.json was written to while the vault was Private"
        )
        svc.stop()

    def test_index_flushes_on_return_to_normal(self, tmp_env, signal_bus):
        svc, vault_path = self._import_in_normal_mode(tmp_env, signal_bus)
        vault_root = svc._vault_root

        svc.set_mode(VaultMode.PRIVATE, persist=False)
        svc.remove_file(vault_path)
        svc.set_mode(VaultMode.NORMAL, persist=False)

        after = _read_raw_index(vault_root)
        assert vault_path not in after.get("items", {}), (
            "removal made while Private was never flushed after leaving Private mode"
        )
        svc.stop()

    def test_stop_while_private_does_not_flush_index(self, tmp_env, signal_bus):
        svc, vault_path = self._import_in_normal_mode(tmp_env, signal_bus)
        vault_root = svc._vault_root

        svc.set_mode(VaultMode.PRIVATE, persist=False)
        svc.remove_file(vault_path)
        svc.stop()  # process "exits" while still Private

        after = _read_raw_index(vault_root)
        assert vault_path in after.get("items", {}), (
            "stop() flushed the index even though the vault was Private"
        )

    def test_remove_still_works_normally_outside_private(self, tmp_env, signal_bus):
        """Guard against the fix breaking the common, non-Private path."""
        svc, vault_path = self._import_in_normal_mode(tmp_env, signal_bus)
        vault_root = svc._vault_root

        assert svc.remove_file(vault_path) is True

        after = _read_raw_index(vault_root)
        assert vault_path not in after.get("items", {})
        svc.stop()
