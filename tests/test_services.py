"""Tests for application services."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_companion.config import ConfigManager
from ai_companion.infrastructure.service_manager import ServiceManager
from ai_companion.infrastructure.signal_bus import SignalBus
from ai_companion.models.config_models import AppConfig, VaultMode


@pytest.fixture
def tmp_env(tmp_path):
    """Isolated config: every store path relocated under tmp_path.

    Uses AppConfig.relocate() rather than listing paths by hand — adding a new
    store to PATH_FIELDS isolates it here automatically. The previous
    hand-written version missed llm.conversations_path and let tests read and
    overwrite the developer's real chat history.
    """
    config = AppConfig().relocate(tmp_path)
    os.makedirs(config.data_dir, exist_ok=True)
    return config, tmp_path


@pytest.fixture
def signal_bus():
    return SignalBus()


class TestLlmService:
    """Test LLM service (without loading a real model)."""

    def test_create_conversation(self, tmp_env, signal_bus):
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        svc = LlmService(config, signal_bus)
        svc.start()

        conv_id = svc.create_conversation("Test system prompt")
        assert conv_id

        conv = svc.get_conversation(conv_id)
        assert conv is not None
        assert conv.system_prompt == "Test system prompt"

        svc.stop()

    def test_list_conversations(self, tmp_env, signal_bus):
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        svc = LlmService(config, signal_bus)
        svc.start()

        svc.create_conversation()
        svc.create_conversation()
        assert len(svc.list_conversations()) == 2

        svc.stop()

    def test_delete_conversation(self, tmp_env, signal_bus):
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        svc = LlmService(config, signal_bus)
        svc.start()

        conv_id = svc.create_conversation()
        assert svc.delete_conversation(conv_id) is True
        assert svc.get_conversation(conv_id) is None
        assert svc.delete_conversation("nonexistent") is False

        svc.stop()

    def test_status(self, tmp_env, signal_bus):
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        svc = LlmService(config, signal_bus)

        status = svc.get_status()
        assert status["model_loaded"] is False
        assert status["generating"] is False

    def test_no_model_send_fails(self, tmp_env, signal_bus):
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        svc = LlmService(config, signal_bus)
        svc.start()

        conv_id = svc.create_conversation()
        # Should emit error, not crash
        svc.send_message(conv_id, "Hello")
        assert not svc.is_generating

        svc.stop()


class TestVaultService:
    """Test the vault service."""

    def test_start_in_private_mode(self, tmp_env, signal_bus):
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        config.vault.mode = VaultMode.PRIVATE
        svc = VaultService(config, signal_bus)
        svc.start()

        assert svc.is_private is True
        svc.stop()

    def test_import_blocked_in_private(self, tmp_env, signal_bus):
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        config.vault.mode = VaultMode.PRIVATE
        svc = VaultService(config, signal_bus)
        svc.start()

        result = svc.import_file("/tmp/test.txt")
        assert result is None

        svc.stop()

    def test_import_in_normal_mode(self, tmp_env, signal_bus):
        from ai_companion.services.vault_service import VaultService

        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.NORMAL
        # Add tmp_path as an approved folder so we can read from it
        config.vault.approved_folders.append(str(tmp_path))
        svc = VaultService(config, signal_bus)
        svc.start()

        # Create a source file
        src = tmp_path / "source.txt"
        src.write_text("test content")

        result = svc.import_file(str(src), category="test")
        assert result is not None
        assert "source.txt" in result

        # Verify file exists in vault
        vault_path = svc.get_file_path(result)
        assert vault_path is not None
        assert vault_path.exists()

        svc.stop()

    def test_list_files(self, tmp_env, signal_bus):
        from ai_companion.services.vault_service import VaultService

        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.NORMAL
        config.vault.approved_folders.append(str(tmp_path))
        svc = VaultService(config, signal_bus)
        svc.start()

        src = tmp_path / "test.txt"
        src.write_text("content")
        svc.import_file(str(src), category="docs")

        files = svc.list_files(category="docs")
        assert len(files) == 1

        svc.stop()

    def test_remove_file(self, tmp_env, signal_bus):
        from ai_companion.services.vault_service import VaultService

        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.NORMAL
        config.vault.approved_folders.append(str(tmp_path))
        svc = VaultService(config, signal_bus)
        svc.start()

        src = tmp_path / "test.txt"
        src.write_text("content")
        vault_path = svc.import_file(str(src))
        assert vault_path is not None

        assert svc.remove_file(vault_path) is True
        assert svc.get_file_path(vault_path) is None

        svc.stop()

    def test_mode_switch(self, tmp_env, signal_bus):
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        config.vault.mode = VaultMode.PRIVATE
        svc = VaultService(config, signal_bus)
        svc.start()

        assert svc.is_private
        svc.set_mode(VaultMode.NORMAL)
        assert not svc.is_private
        svc.set_mode(VaultMode.PRIVATE)
        assert svc.is_private

        svc.stop()

    def test_audit_log(self, tmp_env, signal_bus):
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        svc = VaultService(config, signal_bus)
        svc.start()

        log = svc.get_audit_log()
        # At least the mode_change from import attempts
        assert isinstance(log, list)

        svc.stop()


class TestVaultGetFilePath:
    """get_file_path() must never hand back a path outside the vault.

    It grew up as plain concatenation plus .exists() — the only public
    method in VaultService that skipped PathValidator, so a "../"
    vault_path could reach any file on disk. These tests pin the
    containment check so the fix cannot silently vanish again.
    """

    def _make_service(self, tmp_env, signal_bus, approve_tmp=False):
        from ai_companion.services.vault_service import VaultService

        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.NORMAL
        if approve_tmp:
            # Only the import test needs tmp_path approved; the escape
            # tests deliberately leave it unapproved so files there are
            # NOT valid targets.
            config.vault.approved_folders.append(str(tmp_path))
        svc = VaultService(config, signal_bus)
        svc.start()
        return svc, tmp_path

    def test_real_vault_file_still_resolves(self, tmp_env, signal_bus):
        """The containment check must not break the legitimate path."""
        svc, tmp_path = self._make_service(tmp_env, signal_bus, approve_tmp=True)
        src = tmp_path / "note.txt"
        src.write_text("hello")
        stored = svc.import_file(str(src), category="test")
        assert stored is not None

        full = svc.get_file_path(stored)
        assert full is not None
        assert full.exists()

        svc.stop()

    def test_parent_traversal_is_refused(self, tmp_env, signal_bus):
        svc, _ = self._make_service(tmp_env, signal_bus)

        # Exists on POSIX; on Windows must_exist=True fails instead.
        # Either way the answer must be None, never a Path.
        assert svc.get_file_path("../../etc/passwd") is None

        svc.stop()

    def test_traversal_to_a_real_file_is_refused(self, tmp_env, signal_bus):
        """A crafted relative path that DOES resolve to a real file
        outside the vault must still return None."""
        svc, tmp_path = self._make_service(tmp_env, signal_bus)
        outside = tmp_path / "outside.txt"
        outside.write_text("not vault content")

        rel = os.path.relpath(outside, svc._vault_root)
        assert rel.startswith("..")  # sanity: it really escapes the vault
        assert svc.get_file_path(rel) is None

        svc.stop()

    def test_absolute_path_to_outside_file_is_refused(self, tmp_env, signal_bus):
        svc, tmp_path = self._make_service(tmp_env, signal_bus)
        outside = tmp_path / "outside.txt"
        outside.write_text("not vault content")

        assert svc.get_file_path(str(outside)) is None

        svc.stop()


class TestImageGenLoopback:
    """The ComfyUI worker must always bind to 127.0.0.1.

    start_worker() used to pass config.image_gen.host straight to the
    worker's --listen flag, and _submit_to_worker() used the same value
    to build the request URL — so any stray config value put the worker
    on the network while the class docstring claimed loopback-only. The
    bind/connect address is now pinned; only the port is configurable.
    """

    def _make_service(self, tmp_env, signal_bus, host="0.0.0.0"):
        from ai_companion.services.image_gen_service import ImageGenService

        config, tmp_path = tmp_env
        comfyui = tmp_path / "comfyui"
        comfyui.mkdir()
        config.image_gen.comfyui_path = str(comfyui)
        config.image_gen.host = host  # mistaken or hostile value
        svc = ImageGenService(config, signal_bus)
        svc.start()
        return svc

    def test_worker_always_binds_loopback(self, tmp_env, signal_bus):
        svc = self._make_service(tmp_env, signal_bus)

        with patch("subprocess.Popen") as popen:
            assert svc.start_worker() is True

        cmd = popen.call_args.args[0]
        assert cmd[cmd.index("--listen") + 1] == "127.0.0.1"

        svc.stop()

    def test_ignored_host_is_reported(self, tmp_env, signal_bus):
        """Silently overriding the user's config would be its own bug —
        the service must say why the value was ignored."""
        statuses = []
        signal_bus.service.status_changed.connect(
            lambda _name, msg: statuses.append(msg)
        )
        svc = self._make_service(tmp_env, signal_bus, host="0.0.0.0")

        with patch("subprocess.Popen"):
            svc.start_worker()

        assert any("ignored" in msg and "0.0.0.0" in msg for msg in statuses)

        svc.stop()

    def test_loopback_config_produces_no_warning(self, tmp_env, signal_bus):
        statuses = []
        signal_bus.service.status_changed.connect(
            lambda _name, msg: statuses.append(msg)
        )
        svc = self._make_service(tmp_env, signal_bus, host="127.0.0.1")

        with patch("subprocess.Popen"):
            svc.start_worker()

        assert not any("ignored" in msg for msg in statuses)

        svc.stop()

    def test_submissions_post_to_loopback(self, tmp_env, signal_bus):
        svc = self._make_service(tmp_env, signal_bus, host="192.168.1.50")
        svc._worker_running = True  # pretend a worker is up

        resp = MagicMock()
        resp.read.return_value = json.dumps({"prompt_id": "abc"}).encode()
        resp.__enter__.return_value = resp
        with patch("urllib.request.urlopen", return_value=resp) as urlopen:
            assert svc.generate("a lighthouse at dawn") is not None

        request = urlopen.call_args.args[0]
        assert request.full_url.startswith("http://127.0.0.1:")
        assert "192.168.1.50" not in request.full_url

        svc.stop()


class TestMemoryService:
    """Test the memory service."""

    def test_add_and_get(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.models.memory import MemorySource

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        mem = svc.add_memory(
            content="I prefer dark mode",
            tags=["preference"],
            confidence=0.8,
            auto_approve=True,
        )
        assert mem.content == "I prefer dark mode"
        assert "preference" in mem.tags

        retrieved = svc.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.content == "I prefer dark mode"

        svc.stop()

    def test_update_memory(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        mem = svc.add_memory(content="Original", auto_approve=True)
        updated = svc.update_memory(
            mem.id, content="Updated", confidence=0.9
        )
        assert updated.content == "Updated"
        assert updated.confidence == 0.9

        svc.stop()

    def test_delete_memory(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        mem = svc.add_memory(content="Delete me", auto_approve=True)
        assert svc.delete_memory(mem.id) is True
        assert svc.get_memory(mem.id) is None

        svc.stop()

    def test_search_filters(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.models.memory import MemoryFilter, MemoryStatus

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        svc.add_memory(content="Approved memory", auto_approve=True, tags=["a"])
        svc.add_memory(content="Pending memory", auto_approve=False, tags=["b"])

        # Search all
        all_mems = svc.search()
        assert len(all_mems) == 2

        # Search approved only
        approved = svc.search(MemoryFilter(status=MemoryStatus.APPROVED))
        assert len(approved) == 1
        assert approved[0].content == "Approved memory"

        # Search by tag
        tagged = svc.search(MemoryFilter(tags=["a"]))
        assert len(tagged) == 1

        # Search by query
        found = svc.search(MemoryFilter(query="Approved"))
        assert len(found) == 1

        svc.stop()

    def test_pending_approval(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        mem = svc.add_memory(content="Needs approval", auto_approve=False)
        pending = svc.get_pending()
        assert len(pending) == 1

        svc.approve_memory(mem.id)
        pending = svc.get_pending()
        assert len(pending) == 0

        svc.stop()

    def test_reject_memory(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.models.memory import MemoryStatus

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        mem = svc.add_memory(content="Reject me", auto_approve=False)
        svc.reject_memory(mem.id)

        updated = svc.get_memory(mem.id)
        assert updated.status == MemoryStatus.REJECTED

        svc.stop()

    def test_toggle_pin(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        mem = svc.add_memory(content="Pin me", auto_approve=True)
        assert mem.pinned is False

        svc.toggle_pin(mem.id)
        updated = svc.get_memory(mem.id)
        assert updated.pinned is True

        svc.toggle_pin(mem.id)
        updated = svc.get_memory(mem.id)
        assert updated.pinned is False

        svc.stop()

    def test_empty_content_rejected(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        with pytest.raises(ValueError, match="empty"):
            svc.add_memory(content="")

        svc.stop()

    def test_export_import(self, tmp_env, signal_bus):
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        svc = MemoryService(config, signal_bus)
        svc.start()

        svc.add_memory(content="Memory 1", auto_approve=True)
        svc.add_memory(content="Memory 2", auto_approve=True)

        exported = svc.export_all()
        assert len(exported) == 2

        svc.stop()


class TestGraphService:
    """Test the knowledge graph service."""

    def test_add_node(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        node = svc.add_node(NodeType.MEMORY, "Test Memory")
        assert node.label == "Test Memory"
        assert node.node_type == NodeType.MEMORY

        retrieved = svc.get_node(node.id)
        assert retrieved is not None

        svc.stop()

    def test_add_edge(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType, EdgeType

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        n1 = svc.add_node(NodeType.MEMORY, "Memory 1")
        n2 = svc.add_node(NodeType.PROJECT, "Project 1")

        edge = svc.add_edge(n1.id, n2.id, EdgeType.PART_OF)
        assert edge is not None
        assert edge.source_id == n1.id
        assert edge.target_id == n2.id

        svc.stop()

    def test_self_loop_blocked(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        n = svc.add_node(NodeType.MEMORY, "Self")
        edge = svc.add_edge(n.id, n.id)
        assert edge is None

        svc.stop()

    def test_delete_node_cascades(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        n1 = svc.add_node(NodeType.MEMORY, "A")
        n2 = svc.add_node(NodeType.MEMORY, "B")
        svc.add_edge(n1.id, n2.id)

        edges = svc.get_edges_for_node(n1.id)
        assert len(edges) == 1

        svc.delete_node(n1.id)
        edges = svc.get_edges_for_node(n2.id)
        assert len(edges) == 0

        svc.stop()

    def test_search(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType, GraphQuery

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        svc.add_node(NodeType.MEMORY, "Python Tips")
        svc.add_node(NodeType.PROJECT, "Web App")
        svc.add_node(NodeType.PERSON, "Alice")

        result = svc.search(GraphQuery(text="Python"))
        assert len(result["nodes"]) == 1
        assert result["nodes"][0].label == "Python Tips"

        result = svc.search(GraphQuery(node_types=[NodeType.PERSON]))
        assert len(result["nodes"]) == 1

        svc.stop()

    def test_compute_layout(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        svc.add_node(NodeType.MEMORY, "A")
        svc.add_node(NodeType.MEMORY, "B")
        svc.add_node(NodeType.MEMORY, "C")

        positions = svc.compute_layout()
        assert len(positions) == 3
        for node_id, (x, y) in positions.items():
            assert isinstance(x, float)
            assert isinstance(y, float)

        svc.stop()

    def test_stats(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        svc.add_node(NodeType.MEMORY, "A")
        svc.add_node(NodeType.PERSON, "B")

        stats = svc.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["node_types"]["memory"] == 1
        assert stats["node_types"]["person"] == 1

        svc.stop()

    def test_clear(self, tmp_env, signal_bus):
        from ai_companion.services.graph_service import GraphService
        from ai_companion.models.graph import NodeType

        config, _ = tmp_env
        svc = GraphService(config, signal_bus)
        svc.start()

        svc.add_node(NodeType.MEMORY, "A")
        svc.add_node(NodeType.MEMORY, "B")
        assert svc.get_stats()["total_nodes"] == 2

        svc.clear()
        assert svc.get_stats()["total_nodes"] == 0

        svc.stop()


class TestCodeWorkspace:
    """Test the code workspace service."""

    def test_write_and_read(self, tmp_env, signal_bus):
        from ai_companion.services.code_workspace import CodeWorkspaceService

        config, _ = tmp_env
        svc = CodeWorkspaceService(config, signal_bus)
        svc.start()

        assert svc.write_code("test.py", "print('hello')")
        content = svc.read_code("test.py")
        assert content == "print('hello')"

        svc.stop()

    def test_execute_code(self, tmp_env, signal_bus):
        from ai_companion.services.code_workspace import CodeWorkspaceService

        config, _ = tmp_env
        svc = CodeWorkspaceService(config, signal_bus)
        svc.start()

        exec_id = svc.execute(code="print(2 + 2)")
        assert exec_id

        # Wait for execution
        import time
        time.sleep(2)

        info = svc.get_execution(exec_id)
        assert info is not None
        assert "4" in info.get("output", "")

        svc.stop()

    def test_execute_error(self, tmp_env, signal_bus):
        from ai_companion.services.code_workspace import CodeWorkspaceService

        config, _ = tmp_env
        svc = CodeWorkspaceService(config, signal_bus)
        svc.start()

        exec_id = svc.execute(code="raise ValueError('test error')")
        assert exec_id

        import time
        time.sleep(2)

        info = svc.get_execution(exec_id)
        assert info is not None
        assert info["return_code"] != 0

        svc.stop()

    def test_snapshot_and_restore(self, tmp_env, signal_bus):
        from ai_companion.services.code_workspace import CodeWorkspaceService

        config, _ = tmp_env
        svc = CodeWorkspaceService(config, signal_bus)
        svc.start()

        # Write some files
        svc.write_code("file1.py", "x = 1")
        svc.write_code("file2.py", "y = 2")

        # Create snapshot
        snap_id = svc.create_snapshot("before_changes")
        assert snap_id

        # Modify files
        svc.write_code("file1.py", "x = 100")

        # Restore snapshot
        assert svc.restore_snapshot(snap_id)

        content = svc.read_code("file1.py")
        assert content == "x = 1"

        svc.stop()

    def test_diff_snapshots(self, tmp_env, signal_bus):
        from ai_companion.services.code_workspace import CodeWorkspaceService

        config, _ = tmp_env
        svc = CodeWorkspaceService(config, signal_bus)
        svc.start()

        svc.write_code("test.py", "print('v1')")
        snap1 = svc.create_snapshot("v1")

        svc.write_code("test.py", "print('v2')")
        snap2 = svc.create_snapshot("v2")

        diffs = svc.diff_snapshots(snap1, snap2)
        assert "test.py" in diffs
        assert any("v1" in line or "v2" in line for line in diffs["test.py"])

        svc.stop()

    def test_network_disabled_check(self, tmp_env, signal_bus):
        """Verify the service refuses to start if allow_network is True."""
        from ai_companion.services.code_workspace import CodeWorkspaceService

        config, _ = tmp_env
        config.code_workspace.allow_network = True

        svc = CodeWorkspaceService(config, signal_bus)
        svc.start()

        # Should have emitted an error
        # The service won't be fully operational
        assert not svc.is_running

        svc.stop()


class TestCameraService:
    """Test camera service (without real camera)."""

    def test_auto_record_blocked(self, tmp_env, signal_bus):
        from ai_companion.services.camera_service import CameraService

        config, _ = tmp_env
        config.camera.auto_record = True

        svc = CameraService(config, signal_bus)
        svc.start()

        # capture_frame should fail because auto_record is True
        result = svc.capture_frame()
        assert result is None

        svc.stop()

    def test_service_starts_without_camera(self, tmp_env, signal_bus):
        from ai_companion.services.camera_service import CameraService

        config, _ = tmp_env
        svc = CameraService(config, signal_bus)
        svc.start()

        assert not svc.is_open  # Camera not opened on start

        svc.stop()


class TestFileWorkspace:
    """Test file workspace service."""

    def test_write_and_read(self, tmp_env, signal_bus):
        from ai_companion.services.file_workspace import FileWorkspaceService

        config, _ = tmp_env
        svc = FileWorkspaceService(config, signal_bus)
        svc.start()

        assert svc.write_file("hello.txt", "Hello World")
        content = svc.read_file("hello.txt")
        assert content == "Hello World"

        svc.stop()

    def test_list_directory(self, tmp_env, signal_bus):
        from ai_companion.services.file_workspace import FileWorkspaceService

        config, _ = tmp_env
        svc = FileWorkspaceService(config, signal_bus)
        svc.start()

        svc.write_file("a.txt", "a")
        svc.write_file("b.txt", "b")
        svc.create_directory("subdir")

        entries = svc.list_directory()
        names = [e["name"] for e in entries]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names

        svc.stop()

    def test_delete(self, tmp_env, signal_bus):
        from ai_companion.services.file_workspace import FileWorkspaceService

        config, _ = tmp_env
        svc = FileWorkspaceService(config, signal_bus)
        svc.start()

        svc.write_file("delete_me.txt", "bye")
        assert svc.delete_path("delete_me.txt")
        assert svc.read_file("delete_me.txt") is None

        svc.stop()

    def test_audit_log(self, tmp_env, signal_bus):
        from ai_companion.services.file_workspace import FileWorkspaceService

        config, _ = tmp_env
        svc = FileWorkspaceService(config, signal_bus)
        svc.start()

        svc.write_file("audited.txt", "content")
        log = svc.get_audit_log()
        assert len(log) > 0
        assert any(e["operation"] == "write" for e in log)

        svc.stop()


class TestSandbox3D:
    """Test 3D sandbox service."""

    def test_service_starts(self, tmp_env, signal_bus):
        from ai_companion.services.sandbox3d_service import Sandbox3DService

        config, _ = tmp_env
        svc = Sandbox3DService(config, signal_bus)
        svc.start()

        assert svc.is_running
        assert len(svc.list_objects()) == 0

        svc.stop()

    def test_import_nonexistent_fails(self, tmp_env, signal_bus):
        from ai_companion.services.sandbox3d_service import Sandbox3DService

        config, _ = tmp_env
        svc = Sandbox3DService(config, signal_bus)
        svc.start()

        result = svc.import_model("/nonexistent/model.glb")
        assert result is None

        svc.stop()

    def test_scene_operations(self, tmp_env, signal_bus):
        from ai_companion.services.sandbox3d_service import Sandbox3DService

        config, _ = tmp_env
        svc = Sandbox3DService(config, signal_bus)
        svc.start()

        # Test export (empty scene)
        scene = svc.export_scene()
        assert "objects" in scene
        assert len(scene["objects"]) == 0

        # Test clear
        svc.clear_scene()
        assert len(svc.list_objects()) == 0

        svc.stop()


class TestServiceManager:
    """Test the service manager."""

    def test_register_and_start(self, tmp_env):
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config

        sm = ServiceManager(cm)
        sm.register(LlmService(config, sm.signal_bus))
        sm.register(MemoryService(config, sm.signal_bus))

        sm.start_all()

        status = sm.service_status()
        assert status["LLM"] is True
        assert status["Memory"] is True

        sm.stop_all()

    def test_duplicate_registration(self, tmp_env):
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config

        sm = ServiceManager(cm)
        sm.register(LlmService(config, sm.signal_bus))

        with pytest.raises(ValueError, match="already registered"):
            sm.register(LlmService(config, sm.signal_bus))


class TestConversationPersistence:
    """Chat history must survive restart — added 2026-09-17."""

    def _svc(self, tmp_env, signal_bus):
        from ai_companion.services.llm_service import LlmService
        config, tmp_path = tmp_env
        config.llm.conversations_path = str(tmp_path / "conversations.json")
        return LlmService(config, signal_bus)

    def test_conversations_survive_restart(self, tmp_env, signal_bus):
        from ai_companion.models.conversation import MessageRole

        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        cid = svc.create_conversation()
        conv = svc.get_conversation(cid)
        conv.add_message(MessageRole.USER, "remember this")
        conv.add_message(MessageRole.ASSISTANT, "ok")
        svc.stop()

        svc2 = self._svc(tmp_env, signal_bus)
        svc2.start()
        restored = svc2.get_conversation(cid)
        assert restored is not None
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "remember this"
        svc2.stop()

    def test_empty_conversations_not_persisted(self, tmp_env, signal_bus):
        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        svc.create_conversation()
        svc.create_conversation()
        svc.stop()

        svc2 = self._svc(tmp_env, signal_bus)
        svc2.start()
        assert svc2.list_conversations() == []
        svc2.stop()

    def test_delete_persists(self, tmp_env, signal_bus):
        from ai_companion.models.conversation import MessageRole

        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        cid = svc.create_conversation()
        svc.get_conversation(cid).add_message(MessageRole.USER, "hi")
        svc._save_conversations()
        svc.delete_conversation(cid)
        svc.stop()

        svc2 = self._svc(tmp_env, signal_bus)
        svc2.start()
        assert svc2.get_conversation(cid) is None
        svc2.stop()

    def test_corrupt_store_does_not_crash_startup(
        self, tmp_env, signal_bus
    ):
        path = tmp_env[1] / "conversations.json"
        path.write_text("{ this is not valid json", encoding="utf-8")
        svc = self._svc(tmp_env, signal_bus)
        svc.start()  # must not raise
        assert svc.list_conversations() == []
        svc.stop()

    def test_one_bad_record_skipped_others_kept(
        self, tmp_env, signal_bus
    ):
        import json
        from ai_companion.models.conversation import MessageRole

        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        cid = svc.create_conversation()
        svc.get_conversation(cid).add_message(MessageRole.USER, "good record")
        svc.stop()

        path = tmp_env[1] / "conversations.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["items"]["junk"] = {"not": "a conversation"}
        path.write_text(json.dumps(data), encoding="utf-8")

        svc2 = self._svc(tmp_env, signal_bus)
        svc2.start()
        assert svc2.get_conversation(cid) is not None
        assert svc2.get_conversation("junk") is None
        svc2.stop()

    def test_autotitle_from_first_user_message(
        self, tmp_env, signal_bus
    ):
        from ai_companion.models.conversation import MessageRole

        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        cid = svc.create_conversation()
        conv = svc.get_conversation(cid)
        conv.add_message(MessageRole.USER, "what is the capital of France")
        svc._autotitle(conv)
        assert conv.title == "what is the capital of France"
        svc.stop()

    def test_autotitle_truncates_long_message(
        self, tmp_env, signal_bus
    ):
        from ai_companion.models.conversation import MessageRole

        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        cid = svc.create_conversation()
        conv = svc.get_conversation(cid)
        conv.add_message(MessageRole.USER, "x" * 200)
        svc._autotitle(conv)
        assert len(conv.title) == 51
        assert conv.title.endswith("...")
        svc.stop()

    def test_trims_to_max_stored(self, tmp_env, signal_bus):
        from ai_companion.models.conversation import MessageRole

        tmp_env[0].llm.max_stored_conversations = 3
        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        for i in range(6):
            cid = svc.create_conversation()
            svc.get_conversation(cid).add_message(MessageRole.USER, f"m{i}")
        svc.stop()

        svc2 = self._svc(tmp_env, signal_bus)
        svc2.start()
        assert len(svc2.list_conversations()) == 3
        svc2.stop()


class TestPrivateModeChatPersistence:
    """Private mode must never write chat history to disk — 2026-09-17."""

    def _stack(self, tmp_env, signal_bus, private: bool):
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.config import ConfigManager
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.vault_service import VaultService
        from ai_companion.models.config_models import VaultMode

        config, tmp_path = tmp_env
        config.llm.conversations_path = str(tmp_path / "conversations.json")
        config.vault.mode = VaultMode.PRIVATE if private else VaultMode.NORMAL

        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        vault = VaultService(config, signal_bus)
        llm = LlmService(config, signal_bus)
        sm.register(vault)
        sm.register(llm)
        vault.start()
        llm.start()
        return sm, llm, vault

    def test_private_conversation_not_written(self, tmp_env, signal_bus):
        from ai_companion.models.conversation import MessageRole

        _, llm, _ = self._stack(tmp_env, signal_bus, private=True)
        cid = llm.create_conversation()
        assert llm.get_conversation(cid).private is True
        llm.get_conversation(cid).add_message(MessageRole.USER, "secret")
        llm.stop()

        path = tmp_env[1] / "conversations.json"
        if path.exists():
            assert "secret" not in path.read_text(encoding="utf-8")

    def test_normal_conversation_is_written(self, tmp_env, signal_bus):
        from ai_companion.models.conversation import MessageRole

        _, llm, _ = self._stack(tmp_env, signal_bus, private=False)
        cid = llm.create_conversation()
        assert llm.get_conversation(cid).private is False
        llm.get_conversation(cid).add_message(MessageRole.USER, "public")
        llm.stop()

        path = tmp_env[1] / "conversations.json"
        assert "public" in path.read_text(encoding="utf-8")

    def test_switching_to_private_purges_from_disk(self, tmp_env, signal_bus):
        """A chat started in normal mode, then continued in private, must have
        its already-saved half removed from disk."""
        from ai_companion.models.conversation import MessageRole
        from ai_companion.models.config_models import VaultMode

        _, llm, vault = self._stack(tmp_env, signal_bus, private=False)
        cid = llm.create_conversation()
        llm.get_conversation(cid).add_message(MessageRole.USER, "sensitive")
        llm._save_conversations()

        path = tmp_env[1] / "conversations.json"
        assert "sensitive" in path.read_text(encoding="utf-8")

        vault.set_mode(VaultMode.PRIVATE)
        llm.on_vault_mode_changed("private")

        assert "sensitive" not in path.read_text(encoding="utf-8")
        # still usable in memory
        assert llm.get_conversation(cid) is not None
        llm.stop()

    def test_private_conversation_gone_after_restart(self, tmp_env, signal_bus):
        from ai_companion.models.conversation import MessageRole

        _, llm, _ = self._stack(tmp_env, signal_bus, private=True)
        cid = llm.create_conversation()
        llm.get_conversation(cid).add_message(MessageRole.USER, "ephemeral")
        llm.stop()

        _, llm2, _ = self._stack(tmp_env, signal_bus, private=True)
        assert llm2.get_conversation(cid) is None
        llm2.stop()

    def test_fails_closed_when_vault_missing(self, tmp_env, signal_bus):
        """No vault service registered -> must not crash."""
        from ai_companion.services.llm_service import LlmService

        config, tmp_path = tmp_env
        config.llm.conversations_path = str(tmp_path / "conversations.json")
        llm = LlmService(config, signal_bus)
        llm.start()
        assert llm.is_private_mode() is False  # no manager -> normal
        llm.stop()

    def test_private_flag_defaults_false_for_old_records(self):
        """Backward compat: conversations saved before this field existed."""
        from ai_companion.models.conversation import Conversation

        conv = Conversation.model_validate(
            {"id": "x", "title": "old", "messages": []}
        )
        assert conv.private is False


class TestModelLoadDiagnostics:
    """Model load must fail loudly and specifically — 2026-09-17."""

    def _svc(self, tmp_env, signal_bus):
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        return LlmService(config, signal_bus)

    def test_missing_file_raises_clear_error(self, tmp_env, signal_bus):
        import pytest

        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            svc.load_model(r"C:\nope\missing.gguf")
        svc.stop()

    def test_directory_rejected(self, tmp_env, signal_bus):
        import pytest

        config, tmp_path = tmp_env
        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        with pytest.raises(IsADirectoryError):
            svc.load_model(str(tmp_path))
        svc.stop()

    def test_wrong_extension_rejected(self, tmp_env, signal_bus):
        import pytest

        config, tmp_path = tmp_env
        bogus = tmp_path / "model.bin"
        bogus.write_bytes(b"not a gguf")
        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        with pytest.raises(ValueError, match="gguf"):
            svc.load_model(str(bogus))
        svc.stop()

    def test_error_status_signal_emitted(self, tmp_env, signal_bus):
        import pytest

        received: list[tuple[str, str]] = []
        signal_bus.llm.model_status_changed.connect(
            lambda s, d: received.append((s, d))
        )
        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        with pytest.raises(FileNotFoundError):
            svc.load_model("/definitely/not/here.gguf")
        svc.stop()
        assert any(s == "error" for s, _ in received)
        assert any("not found" in d.lower() for s, d in received if s == "error")

    def test_model_not_marked_loaded_after_failure(self, tmp_env, signal_bus):
        import pytest

        svc = self._svc(tmp_env, signal_bus)
        svc.start()
        with pytest.raises(FileNotFoundError):
            svc.load_model("/definitely/not/here.gguf")
        assert svc.model_loaded is False
        svc.stop()


class TestPrivateModeMemoryAndGraph:
    """Private mode must reach memory and graph, not just chat — 2026-09-17."""

    def _stack(self, tmp_env, signal_bus, private: bool):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.graph_service import GraphService
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService

        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.PRIVATE if private else VaultMode.NORMAL
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        vault = VaultService(config, signal_bus)
        mem = MemoryService(config, signal_bus)
        graph = GraphService(config, signal_bus)
        for svc in (vault, mem, graph):
            sm.register(svc)
            svc.start()
        return sm, vault, mem, graph, tmp_path

    # ---------------- memory ----------------

    def test_private_memory_not_written(self, tmp_env, signal_bus):
        _, _, mem, _, tmp_path = self._stack(tmp_env, signal_bus, private=True)
        m = mem.add_memory("my therapist's name is Dr X")
        assert m.private is True
        mem.stop()
        path = tmp_path / "data" / "memories.json"
        if path.exists():
            assert "Dr X" not in path.read_text(encoding="utf-8")

    def test_normal_memory_is_written(self, tmp_env, signal_bus):
        _, _, mem, _, tmp_path = self._stack(tmp_env, signal_bus, private=False)
        m = mem.add_memory("I use PySide6")
        assert m.private is False
        mem.stop()
        path = tmp_path / "data" / "memories.json"
        assert "PySide6" in path.read_text(encoding="utf-8")

    def test_private_memory_usable_in_session(self, tmp_env, signal_bus):
        """Private does not mean crippled — it must work until restart."""
        from ai_companion.models.memory import MemoryFilter

        _, _, mem, _, _ = self._stack(tmp_env, signal_bus, private=True)
        m = mem.add_memory("ephemeral note")
        assert mem.get_memory(m.id) is not None
        assert any(
            r.id == m.id for r in mem.search(MemoryFilter(query="ephemeral"))
        )
        mem.stop()

    def test_private_memory_gone_after_restart(self, tmp_env, signal_bus):
        _, _, mem, _, _ = self._stack(tmp_env, signal_bus, private=True)
        m = mem.add_memory("vanishes")
        mem.stop()
        _, _, mem2, _, _ = self._stack(tmp_env, signal_bus, private=True)
        assert mem2.get_memory(m.id) is None
        mem2.stop()

    def test_existing_memories_survive_entering_private(
        self, tmp_env, signal_bus
    ):
        """Toggling Private must NOT delete curated long-term memories.

        This is the key difference from conversations: a memory is a stored
        fact the owner approved, not a live session.
        """
        from ai_companion.models.config_models import VaultMode

        _, vault, mem, _, tmp_path = self._stack(
            tmp_env, signal_bus, private=False
        )
        m = mem.add_memory("long term fact worth keeping")
        mem.stop()
        path = tmp_path / "data" / "memories.json"
        assert "long term fact" in path.read_text(encoding="utf-8")

        _, vault2, mem2, _, _ = self._stack(tmp_env, signal_bus, private=True)
        assert mem2.get_memory(m.id) is not None
        mem2.add_memory("secret")
        mem2.stop()
        text = path.read_text(encoding="utf-8")
        assert "long term fact" in text
        assert "secret" not in text

    def test_mixed_public_and_private_memories(self, tmp_env, signal_bus):
        from ai_companion.models.config_models import VaultMode

        sm, vault, mem, _, tmp_path = self._stack(
            tmp_env, signal_bus, private=False
        )
        mem.add_memory("public one")
        vault.set_mode(VaultMode.PRIVATE)
        mem.add_memory("private one")
        mem.stop()
        text = (tmp_path / "data" / "memories.json").read_text(encoding="utf-8")
        assert "public one" in text
        assert "private one" not in text

    # ---------------- graph ----------------

    def test_private_node_not_written(self, tmp_env, signal_bus):
        _, _, _, graph, tmp_path = self._stack(
            tmp_env, signal_bus, private=True
        )
        n = graph.add_node("person", "Secret Contact")
        assert n.private is True
        graph.stop()
        path = tmp_path / "data" / "graph.json"
        if path.exists():
            assert "Secret Contact" not in path.read_text(encoding="utf-8")

    def test_normal_node_is_written(self, tmp_env, signal_bus):
        _, _, _, graph, tmp_path = self._stack(
            tmp_env, signal_bus, private=False
        )
        graph.add_node("project", "Public Project")
        graph.stop()
        text = (tmp_path / "data" / "graph.json").read_text(encoding="utf-8")
        assert "Public Project" in text

    def test_private_edge_not_written(self, tmp_env, signal_bus):
        _, _, _, graph, tmp_path = self._stack(
            tmp_env, signal_bus, private=True
        )
        a = graph.add_node("person", "A")
        b = graph.add_node("person", "B")
        e = graph.add_edge(a.id, b.id, "related_to", label="SENSITIVE_LINK")
        assert e.private is True
        graph.stop()
        edge_path = tmp_path / "data" / "graph_edges.json"
        if edge_path.exists():
            assert "SENSITIVE_LINK" not in edge_path.read_text(encoding="utf-8")

    def test_private_graph_usable_in_session(self, tmp_env, signal_bus):
        _, _, _, graph, _ = self._stack(tmp_env, signal_bus, private=True)
        a = graph.add_node("idea", "temp idea")
        assert graph.get_node(a.id) is not None
        assert graph.get_stats()["total_nodes"] >= 1
        graph.stop()

    def test_private_graph_gone_after_restart(self, tmp_env, signal_bus):
        _, _, _, graph, _ = self._stack(tmp_env, signal_bus, private=True)
        a = graph.add_node("idea", "temp idea")
        graph.stop()
        _, _, _, graph2, _ = self._stack(tmp_env, signal_bus, private=True)
        assert graph2.get_node(a.id) is None
        graph2.stop()

    def test_existing_nodes_survive_entering_private(self, tmp_env, signal_bus):
        from ai_companion.models.config_models import VaultMode

        _, vault, _, graph, tmp_path = self._stack(
            tmp_env, signal_bus, private=False
        )
        n = graph.add_node("project", "Durable Project")
        vault.set_mode(VaultMode.PRIVATE)
        graph.add_node("idea", "Fleeting Idea")
        graph.stop()
        text = (tmp_path / "data" / "graph.json").read_text(encoding="utf-8")
        assert "Durable Project" in text
        assert "Fleeting Idea" not in text

    # ---------------- shared mixin ----------------

    def test_fails_closed_on_vault_error(self, tmp_env, signal_bus):
        """A broken vault must default to private, never to writing."""
        from ai_companion.infrastructure.privacy import PrivacyMixin

        class Boom:
            def get(self, _name):
                raise RuntimeError("vault exploded")

        class Svc(PrivacyMixin):
            _service_manager = Boom()

        assert Svc().is_private_mode() is True

    def test_no_manager_means_normal(self, tmp_env, signal_bus):
        from ai_companion.infrastructure.privacy import PrivacyMixin

        class Svc(PrivacyMixin):
            _service_manager = None

        assert Svc().is_private_mode() is False

    def test_all_three_services_share_one_implementation(self):
        """Guard rail: privacy logic must not be reimplemented per service."""
        from ai_companion.infrastructure.privacy import PrivacyMixin
        from ai_companion.services.graph_service import GraphService
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.memory_service import MemoryService

        for svc in (LlmService, MemoryService, GraphService):
            assert issubclass(svc, PrivacyMixin), svc.__name__
            assert "is_private_mode" not in svc.__dict__, (
                f"{svc.__name__} overrides the shared privacy check"
            )


class TestPrivateModeDoesNotDestroyHistory:
    """Regression: toggling Private must not wipe saved chats — 2026-09-17."""

    def _stack(self, tmp_env, signal_bus, private=False):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.vault_service import VaultService

        config, tmp_path = tmp_env
        config.vault.mode = VaultMode.PRIVATE if private else VaultMode.NORMAL
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        vault = VaultService(config, signal_bus)
        llm = LlmService(config, signal_bus)
        sm.register(vault)
        sm.register(llm)
        vault.start()
        llm.start()
        return vault, llm, tmp_path

    def test_old_saved_chats_survive_entering_private(
        self, tmp_env, signal_bus
    ):
        """The bug this replaces silently deleted ALL saved history."""
        from ai_companion.models.config_models import VaultMode
        from ai_companion.models.conversation import MessageRole

        _, llm, tmp_path = self._stack(tmp_env, signal_bus)
        cid = llm.create_conversation()
        llm.get_conversation(cid).add_message(MessageRole.USER, "KEEP_ME")
        llm.stop()

        path = tmp_path / "data" / "conversations.json"
        assert "KEEP_ME" in path.read_text(encoding="utf-8")

        # New session: restore from disk, then switch to private
        vault2, llm2, _ = self._stack(tmp_env, signal_bus)
        assert llm2.get_conversation(cid) is not None
        vault2.set_mode(VaultMode.PRIVATE)

        assert "KEEP_ME" in path.read_text(encoding="utf-8"), (
            "toggling Private deleted a previously saved conversation"
        )
        llm2.stop()
        assert "KEEP_ME" in path.read_text(encoding="utf-8")

    def test_active_chat_is_scrubbed_on_entering_private(
        self, tmp_env, signal_bus
    ):
        """The live conversation still gets protected — that part was right."""
        from ai_companion.models.config_models import VaultMode
        from ai_companion.models.conversation import MessageRole

        vault, llm, tmp_path = self._stack(tmp_env, signal_bus)
        cid = llm.create_conversation()
        llm._active_conversation_id = cid
        llm.get_conversation(cid).add_message(MessageRole.USER, "SENSITIVE_NOW")
        llm._save_conversations()

        path = tmp_path / "data" / "conversations.json"
        assert "SENSITIVE_NOW" in path.read_text(encoding="utf-8")

        vault.set_mode(VaultMode.PRIVATE)
        assert "SENSITIVE_NOW" not in path.read_text(encoding="utf-8")
        assert llm.get_conversation(cid) is not None  # still usable
        llm.stop()

    def test_only_active_chat_affected_not_siblings(
        self, tmp_env, signal_bus
    ):
        from ai_companion.models.config_models import VaultMode
        from ai_companion.models.conversation import MessageRole

        vault, llm, tmp_path = self._stack(tmp_env, signal_bus)
        old = llm.create_conversation()
        llm.get_conversation(old).add_message(MessageRole.USER, "SIBLING_CHAT")
        active = llm.create_conversation()
        llm._active_conversation_id = active
        llm.get_conversation(active).add_message(MessageRole.USER, "ACTIVE_CHAT")
        llm._save_conversations()

        vault.set_mode(VaultMode.PRIVATE)
        text = (tmp_path / "data" / "conversations.json").read_text(
            encoding="utf-8"
        )
        assert "SIBLING_CHAT" in text
        assert "ACTIVE_CHAT" not in text
        llm.stop()


class TestAttachmentReader:
    """Attachments must actually reach the model — 2026-09-17."""

    def test_plain_text_extracted(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "notes.txt"
        f.write_text("hello from a file", encoding="utf-8")
        r = extract(str(f))
        assert r.ok
        assert "hello from a file" in r.text

    def test_python_source_extracted(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "bot.py"
        f.write_text("import discord\nclient = discord.Client()\n", "utf-8")
        r = extract(str(f))
        assert r.ok
        assert "import discord" in r.text

    def test_json_is_pretty_printed(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "d.json"
        f.write_text('{"b":2,"a":1}', encoding="utf-8")
        r = extract(str(f))
        assert r.ok
        assert "\n" in r.text  # reformatted, not one line

    def test_csv_rendered_as_rows(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "d.csv"
        f.write_text("name,age\nsteven,40\n", encoding="utf-8")
        r = extract(str(f))
        assert r.ok
        assert "name | age" in r.text

    def test_image_is_refused_not_faked(self, tmp_path):
        """A text-only model must never be handed an image silently."""
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "photo.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
        r = extract(str(f))
        assert r.ok is False
        assert "text-only" in r.note
        assert "NOT been given" in r.note

    def test_binary_without_known_extension_refused(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "blob.dat"
        f.write_bytes(b"\x00\x01\x02\x03" * 100)
        r = extract(str(f))
        assert r.ok is False
        assert "binary" in r.note.lower()

    def test_missing_file_handled(self):
        from ai_companion.services.attachment_reader import extract

        r = extract("/no/such/file.txt")
        assert r.ok is False
        assert "exists" in r.note.lower()

    def test_empty_file_handled(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert extract(str(f)).ok is False

    def test_oversize_file_refused(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "big.txt"
        f.write_text("x" * 5000, encoding="utf-8")
        r = extract(str(f), max_bytes=1000)
        assert r.ok is False
        assert "limit" in r.note.lower()

    def test_long_text_truncated_to_budget(self, tmp_path):
        """A huge file must not blow the context window."""
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "long.txt"
        f.write_text("y" * 50_000, encoding="utf-8")
        r = extract(str(f), char_budget=1000)
        assert r.ok
        assert r.truncated
        assert len(r.text) == 1000

    def test_prompt_block_marks_truncation(self, tmp_path):
        from ai_companion.services.attachment_reader import extract

        f = tmp_path / "long.txt"
        f.write_text("z" * 5000, encoding="utf-8")
        block = extract(str(f), char_budget=100).as_prompt_block()
        assert "truncated" in block

    def test_prefix_combines_multiple_files(self, tmp_path):
        from ai_companion.services.attachment_reader import (
            build_prompt_prefix,
            extract,
        )

        a = tmp_path / "a.txt"
        a.write_text("AAA", encoding="utf-8")
        b = tmp_path / "b.txt"
        b.write_text("BBB", encoding="utf-8")
        prefix = build_prompt_prefix([extract(str(a)), extract(str(b))])
        assert "AAA" in prefix and "BBB" in prefix
        assert "a.txt" in prefix and "b.txt" in prefix

    def test_empty_attachment_list_yields_no_prefix(self):
        from ai_companion.services.attachment_reader import build_prompt_prefix

        assert build_prompt_prefix([]) == ""


class TestAttachmentsReachTheModel:
    """End-to-end: file text must land in the prompt history."""

    def test_prompt_content_used_for_model_not_display(self):
        from ai_companion.models.conversation import Conversation, MessageRole

        conv = Conversation()
        conv.add_message(MessageRole.USER, "what is this?")
        conv.messages[-1].metadata["prompt_content"] = (
            "[Attached file: a.txt]\n```\nSECRET_CONTENT\n```\n\nwhat is this?"
        )
        history = conv.get_history()
        assert "SECRET_CONTENT" in history[-1]["content"]
        # the transcript still shows only what the user typed
        assert conv.messages[-1].content == "what is this?"

    def test_history_falls_back_to_content(self):
        from ai_companion.models.conversation import Conversation, MessageRole

        conv = Conversation()
        conv.add_message(MessageRole.USER, "plain message")
        assert conv.get_history()[-1]["content"] == "plain message"

    def test_legacy_message_without_metadata_loads(self):
        """Backward compatibility with conversations saved before this field."""
        from ai_companion.models.conversation import Message

        m = Message.model_validate({"role": "user", "content": "old"})
        assert m.metadata == {}


class TestVaultModePersistence:
    """Header toggle must survive restart — 2026-09-18."""

    def test_default_is_normal_so_chats_actually_save(self):
        """PRIVATE-by-default silently discarded the user's conversations."""
        from ai_companion.models.config_models import AppConfig, VaultMode

        assert AppConfig().vault.mode == VaultMode.NORMAL

    def test_set_mode_writes_to_config(self, tmp_env, signal_bus, monkeypatch):
        """Persisting must NOT depend on the current working directory.

        config.json is anchored to the project root on purpose; chdir must
        not move it. This test patches the path explicitly instead.
        """
        import json

        from ai_companion import config as config_module
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        config, root = tmp_env
        target = root / "config.json"
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )

        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE)
        vault.stop()

        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["vault"]["mode"] == "private"

    def test_set_mode_updates_live_config_object(self, tmp_env, signal_bus):
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE, persist=False)
        assert config.vault.mode == VaultMode.PRIVATE
        assert vault.is_private is True
        vault.stop()

    def test_setting_same_mode_is_a_noop(self, tmp_env, signal_bus):
        """Avoids a redundant emit storm and a pointless config write."""
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        config.vault.mode = VaultMode.NORMAL
        vault = VaultService(config, signal_bus)
        vault.start()
        seen = []
        signal_bus.vault.mode_changed.connect(seen.append)
        vault.set_mode(VaultMode.NORMAL, persist=False)
        assert seen == []
        vault.stop()

    def test_toggle_round_trips(self, tmp_env, signal_bus):
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE, persist=False)
        assert vault.is_private
        vault.set_mode(VaultMode.NORMAL, persist=False)
        assert not vault.is_private
        vault.stop()

    def test_emits_signal_so_ui_cannot_drift(self, tmp_env, signal_bus):
        """The badge repaints from this signal, never optimistically."""
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        vault = VaultService(config, signal_bus)
        vault.start()
        seen = []
        signal_bus.vault.mode_changed.connect(seen.append)
        vault.set_mode(VaultMode.PRIVATE, persist=False)
        assert seen == ["private"]
        vault.stop()

    def test_normal_mode_actually_saves_a_conversation(
        self, tmp_env, signal_bus
    ):
        """End-to-end: the default install must persist chats."""
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.conversation import MessageRole
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.vault_service import VaultService

        config, root = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        vault = VaultService(config, signal_bus)
        llm = LlmService(config, signal_bus)
        sm.register(vault)
        sm.register(llm)
        vault.start()
        llm.start()

        cid = llm.create_conversation()
        llm.get_conversation(cid).add_message(MessageRole.USER, "PERSIST_ME")
        llm.stop()

        text = (root / "data" / "conversations.json").read_text(
            encoding="utf-8"
        )
        assert "PERSIST_ME" in text


class TestMemoryInjection:
    """Approved memories must reach the model — 2026-09-18."""

    def _mem(self, content, status="approved", pinned=False, conf=0.5,
             modified="2026-01-01"):
        from ai_companion.models.memory import Memory, MemoryStatus

        return Memory(
            content=content,
            status=MemoryStatus(status),
            pinned=pinned,
            confidence=conf,
            modified=modified,
        )

    def test_approved_memories_are_included(self):
        from ai_companion.services.memory_injection import build_memory_block

        block = build_memory_block([self._mem("Uses Python 3.12")])
        assert "Uses Python 3.12" in block

    def test_pending_memories_excluded(self):
        """Unreviewed guesses must not be fed to the model."""
        from ai_companion.services.memory_injection import build_memory_block

        block = build_memory_block([self._mem("GUESS", status="pending")])
        assert block == ""

    def test_rejected_memories_excluded(self):
        from ai_companion.services.memory_injection import build_memory_block

        block = build_memory_block([self._mem("NOPE", status="rejected")])
        assert block == ""

    def test_empty_content_ignored(self):
        from ai_companion.services.memory_injection import build_memory_block

        assert build_memory_block([self._mem("   ")]) == ""

    def test_no_memories_yields_empty_block(self):
        from ai_companion.services.memory_injection import build_memory_block

        assert build_memory_block([]) == ""

    def test_pinned_sorts_first(self):
        from ai_companion.services.memory_injection import select_memories

        chosen = select_memories([
            self._mem("ordinary", conf=0.9),
            self._mem("important", pinned=True, conf=0.1),
        ])
        assert chosen[0].content == "important"

    def test_confidence_orders_unpinned(self):
        from ai_companion.services.memory_injection import select_memories

        chosen = select_memories([
            self._mem("low", conf=0.2),
            self._mem("high", conf=0.9),
        ])
        assert [m.content for m in chosen] == ["high", "low"]

    def test_budget_is_enforced(self):
        """Memories must not crowd the conversation out of a 4096 window."""
        from ai_companion.services.memory_injection import select_memories

        many = [self._mem("x" * 100) for _ in range(50)]
        chosen = select_memories(many, char_budget=500)
        assert 0 < len(chosen) < 50
        assert sum(len(m.content) + 3 for m in chosen) <= 500

    def test_budget_keeps_pinned_even_when_crowded(self):
        from ai_companion.services.memory_injection import select_memories

        memories = [self._mem("y" * 200) for _ in range(10)]
        memories.append(self._mem("PINNED FACT", pinned=True))
        chosen = select_memories(memories, char_budget=250)
        assert any(m.content == "PINNED FACT" for m in chosen)

    def test_memories_never_truncated_mid_sentence(self):
        """Half a fact is worse than no fact."""
        from ai_companion.services.memory_injection import select_memories

        memories = [self._mem("a" * 400), self._mem("short one")]
        chosen = select_memories(memories, char_budget=100)
        for m in chosen:
            assert m.content in ("short one",)

    def test_base_prompt_comes_first(self):
        from ai_companion.services.memory_injection import compose_system_prompt

        out = compose_system_prompt("BE TERSE", [self._mem("A fact")])
        assert out.index("BE TERSE") < out.index("A fact")

    def test_compose_without_memories_returns_base(self):
        from ai_companion.services.memory_injection import compose_system_prompt

        assert compose_system_prompt("BE TERSE", []) == "BE TERSE"

    def test_compose_without_base_returns_block(self):
        from ai_companion.services.memory_injection import compose_system_prompt

        out = compose_system_prompt("", [self._mem("A fact")])
        assert "A fact" in out

    def test_newlines_flattened(self):
        """A multi-line memory must not break the bullet list."""
        from ai_companion.services.memory_injection import build_memory_block

        block = build_memory_block([self._mem("line one\nline two")])
        assert "- line one line two" in block

    # ---------- end to end through the service ----------

    def test_llm_sends_memories_in_system_prompt(self, tmp_env, signal_bus):
        import time

        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        config.llm.system_prompt = "BASE PROMPT"
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for svc in (
            VaultService(config, signal_bus),
            MemoryService(config, signal_bus),
            LlmService(config, signal_bus),
        ):
            sm.register(svc)
            svc.start()

        mem = sm.get("Memory")
        llm = sm.get("LLM")
        mem.add_memory("Steven uses Python 3.12", auto_approve=True)
        mem.add_memory("UNREVIEWED", auto_approve=False)

        captured = {}

        class Stub:
            def create_chat_completion(self, messages, **kw):
                captured["messages"] = messages
                return iter([{"choices": [{"delta": {"content": "ok"}}]}])

        llm._model = Stub()
        llm._model_loaded = True
        cid = llm.create_conversation()
        llm.send_message(cid, "hello")
        for _ in range(40):
            if "messages" in captured:
                break
            time.sleep(0.05)

        system = captured["messages"][0]["content"]
        assert "BASE PROMPT" in system
        assert "Python 3.12" in system
        assert "UNREVIEWED" not in system
        llm.stop()

    def test_chat_works_when_memory_service_absent(self, tmp_env, signal_bus):
        """Memory must never be able to break chat."""
        import time

        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.llm_service import LlmService

        config, _ = tmp_env
        config.llm.system_prompt = "BASE ONLY"
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        llm = LlmService(config, signal_bus)
        sm.register(llm)
        llm.start()

        captured = {}

        class Stub:
            def create_chat_completion(self, messages, **kw):
                captured["messages"] = messages
                return iter([{"choices": [{"delta": {"content": "ok"}}]}])

        llm._model = Stub()
        llm._model_loaded = True
        cid = llm.create_conversation()
        llm.send_message(cid, "hello")
        for _ in range(40):
            if "messages" in captured:
                break
            time.sleep(0.05)
        assert captured["messages"][0]["content"] == "BASE ONLY"
        llm.stop()


class TestBootStartsServices:
    """Boot screen drives real service startup — 2026-09-18.

    Lives here rather than in test_infrastructure because it needs the
    tmp_env / signal_bus fixtures defined in this module.
    """

    _KEEPALIVE: list = []

    def _app(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    def _make(self, steps):
        """Build a BootScreen and keep it alive for the module's lifetime.

        A dropped BootScreen with armed QTimers crashes the interpreter the
        next time any test pumps the event loop.
        """
        from ai_companion.ui.boot import BootScreen

        screen = BootScreen(steps)
        type(self)._KEEPALIVE.append(screen)
        return screen

    def test_build_steps_covers_every_service(self, tmp_env, signal_bus):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService
        from ai_companion.ui.boot import build_steps

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        sm.register(VaultService(config, signal_bus))
        sm.register(MemoryService(config, signal_bus))

        steps = build_steps(sm, config)
        labels = " ".join(s.label for s in steps)
        assert "VAULT" in labels
        assert "MEMORY" in labels
        # first and last are framing lines with no action
        assert steps[0].action is None
        assert steps[-1].action is None

    def test_boot_starts_real_services(self, tmp_env, signal_bus):
        """End to end: after boot, services are actually running."""
        app = self._app()

        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService
        from ai_companion.ui.boot import build_steps

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        sm.register(VaultService(config, signal_bus))
        sm.register(MemoryService(config, signal_bus))
        assert all(not s.is_running for s in sm.services.values())

        screen = self._make(build_steps(sm, config))
        screen.show()
        screen.skip()
        assert all(s.is_running for s in sm.services.values())
        sm.stop_all()

    def test_start_all_is_idempotent(self, tmp_env, signal_bus):
        """The boot screen starts services, then start_all runs as a net."""
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.memory_service import MemoryService

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        sm.register(MemoryService(config, signal_bus))
        sm.start_all()
        sm.start_all()  # must not raise or double-start
        assert all(s.is_running for s in sm.services.values())
        sm.stop_all()

    def test_services_property_is_a_copy(self):
        """Callers must not be able to mutate the registry by accident."""
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager

        sm = ServiceManager(ConfigManager())
        snapshot = sm.services
        snapshot["bogus"] = object()
        assert "bogus" not in sm.services


class TestContextBudget:
    """Prompt + reply must fit n_ctx — 2026-09-19."""

    def _msgs(self, n, chars=400, system=True):
        out = []
        if system:
            out.append({"role": "system", "content": "S" * 200})
        for i in range(n):
            out.append({"role": "user", "content": f"m{i} " + "x" * chars})
        return out

    def test_short_conversation_untouched(self):
        from ai_companion.services.context_budget import fit_to_context

        msgs = self._msgs(3, chars=100)
        r = fit_to_context(msgs, 4096, 512)
        assert r.dropped_messages == 0
        assert len(r.messages) == len(msgs)

    def test_long_history_is_trimmed(self):
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context(self._msgs(200, chars=400), 4096, 2048)
        assert r.truncated
        assert len(r.messages) < 200

    def test_result_actually_fits_the_window(self):
        """The whole point. Prompt + reply must not exceed n_ctx."""
        from ai_companion.services.context_budget import (
            estimate_messages,
            fit_to_context,
        )

        for ctx, reply in ((4096, 2048), (2048, 512), (512, 2048), (8192, 4096)):
            r = fit_to_context(self._msgs(120, chars=600), ctx, reply)
            assert estimate_messages(r.messages) + r.max_tokens <= ctx, (
                f"ctx={ctx} reply={reply}"
            )

    def test_system_prompt_is_never_dropped(self):
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context(self._msgs(200, chars=500), 2048, 512)
        assert r.messages[0]["role"] == "system"

    def test_newest_message_is_always_present(self):
        """It is the question being asked; losing it is nonsensical."""
        from ai_companion.services.context_budget import fit_to_context

        msgs = self._msgs(100, chars=400)
        msgs[-1]["content"] = "THE ACTUAL QUESTION"
        r = fit_to_context(msgs, 2048, 512)
        assert "THE ACTUAL QUESTION" in r.messages[-1]["content"]

    def test_oversized_newest_message_is_clipped_not_dropped(self):
        """A 12000-char attachment lives in ONE message."""
        from ai_companion.services.context_budget import (
            estimate_messages,
            fit_to_context,
        )

        msgs = [
            {"role": "system", "content": "S" * 200},
            {"role": "user", "content": "A" * 40000 + " FINAL QUESTION"},
        ]
        r = fit_to_context(msgs, 2048, 512)
        assert len(r.messages) == 2
        assert estimate_messages(r.messages) + r.max_tokens <= 2048
        assert "FINAL QUESTION" in r.messages[-1]["content"]

    def test_truncation_is_reported(self):
        """Silent amnesia is worse than a stated limit."""
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context(self._msgs(200, chars=500), 2048, 512)
        assert r.notes
        assert any("omitted" in n for n in r.notes)

    def test_reply_budget_shrinks_rather_than_starving_prompt(self):
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context(self._msgs(4, chars=200), 512, 4096)
        assert r.max_tokens < 4096
        assert r.max_tokens >= 128

    def test_reply_never_below_floor(self):
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context(self._msgs(50, chars=800), 512, 8)
        assert r.max_tokens >= 128

    def test_empty_conversation_is_safe(self):
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context([], 4096, 512)
        assert r.messages == []
        assert r.max_tokens > 0

    def test_only_a_system_message(self):
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context([{"role": "system", "content": "hi"}], 4096, 512)
        assert len(r.messages) == 1

    def test_huge_system_prompt_warns(self):
        from ai_companion.services.context_budget import fit_to_context

        r = fit_to_context(
            [{"role": "system", "content": "S" * 20000},
             {"role": "user", "content": "q"}],
            2048, 512,
        )
        assert any("context window" in n for n in r.notes)

    def test_order_is_preserved(self):
        from ai_companion.services.context_budget import fit_to_context

        msgs = self._msgs(6, chars=50)
        r = fit_to_context(msgs, 4096, 512)
        bodies = [m["content"] for m in r.messages if m["role"] == "user"]
        assert bodies == sorted(bodies, key=lambda b: int(b.split()[0][1:]))

    def test_estimate_is_conservative(self):
        """Under-counting would let a prompt slip over the real limit."""
        from ai_companion.services.context_budget import estimate_tokens

        assert estimate_tokens("") == 0
        assert estimate_tokens("a" * 400) >= 100

    # ---- end to end through the service ----

    def test_service_never_exceeds_n_ctx(self, tmp_env, signal_bus):
        import sys
        import time
        import types

        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.conversation import MessageRole
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService

        captured = {}

        class _Llama:
            def __init__(self, **kwargs):
                captured["n_ctx"] = kwargs.get("n_ctx")

            def create_chat_completion(self, messages, **kwargs):
                captured["messages"] = messages
                captured["max_tokens"] = kwargs.get("max_tokens")
                return iter([{"choices": [{"delta": {"content": "ok"}}]}])

        stub = types.ModuleType("llama_cpp")
        stub.Llama = _Llama
        sys.modules["llama_cpp"] = stub

        config, root = tmp_env
        config.llm.context_length = 4096
        config.llm.max_tokens = 2048
        model = root / "Q.gguf"
        model.write_bytes(b"\x00" * 2048)
        config.llm.model_path = str(model)

        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, MemoryService, LlmService):
            sm.register(cls(config, signal_bus))
        sm.start_all()

        mem = sm.get("Memory")
        llm = sm.get("LLM")
        for i in range(8):
            mem.add_memory(f"Fact {i}: " + "detail " * 12, auto_approve=True)

        attachment = root / "big.py"
        attachment.write_text("# comment\nx = 1\n" * 700, encoding="utf-8")

        cid = llm.create_conversation()
        for i in range(12):
            llm.get_conversation(cid).add_message(
                MessageRole.USER, f"Turn {i}: " + "explain in detail " * 20
            )
        llm.send_message(cid, "summarise this", [str(attachment)])
        for _ in range(60):
            if "messages" in captured:
                break
            time.sleep(0.05)

        from ai_companion.services.context_budget import estimate_messages

        total = estimate_messages(captured["messages"]) + captured["max_tokens"]
        assert total <= captured["n_ctx"], f"{total} > {captured['n_ctx']}"
        llm.stop()


class TestGraphSync:
    """Graph is derived from memories — 2026-09-19."""

    def _stack(self, tmp_env, signal_bus, private=False):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.graph_service import GraphService
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService

        config, root = tmp_env
        config.vault.mode = VaultMode.PRIVATE if private else VaultMode.NORMAL
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, MemoryService, GraphService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        return sm, sm.get("Memory"), sm.get("Graph"), root

    def test_memories_become_nodes(self, tmp_env, signal_bus):
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        mem.add_memory("Uses Python 3.12", tags=["python"], auto_approve=True)
        result = sync_memories_to_graph(mem.search(), graph)
        assert result.memories_added == 1
        labels = [n.label for n in graph.search()["nodes"]]
        assert any("Python 3.12" in l for l in labels)

    def test_tags_become_shared_nodes(self, tmp_env, signal_bus):
        """Two memories with the same tag must join at ONE node."""
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        mem.add_memory("A", tags=["python"], auto_approve=True)
        mem.add_memory("B", tags=["python"], auto_approve=True)
        sync_memories_to_graph(mem.search(), graph)
        tags = [n for n in graph.search()["nodes"]
                if n.node_type.value == "tag"]
        assert len(tags) == 1
        assert len(graph.get_edges_for_node(tags[0].id)) == 2

    def test_sync_is_idempotent(self, tmp_env, signal_bus):
        """Running twice must not duplicate anything."""
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        mem.add_memory("A", tags=["x", "y"], auto_approve=True)
        sync_memories_to_graph(mem.search(), graph)
        first = graph.get_stats()
        second = sync_memories_to_graph(mem.search(), graph)
        assert second.total_changes == 0
        assert graph.get_stats() == first

    def test_pending_memories_excluded(self, tmp_env, signal_bus):
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        mem.add_memory("UNREVIEWED", tags=["x"])
        result = sync_memories_to_graph(mem.search(), graph)
        assert result.memories_added == 0

    def test_private_memories_never_graphed(self, tmp_env, signal_bus):
        """A private memory must not reappear as a node written to disk."""
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.graph_sync import sync_memories_to_graph

        sm, mem, graph, _ = self._stack(tmp_env, signal_bus)
        mem.add_memory("PUBLIC", tags=["ok"], auto_approve=True)
        sm.get("Vault").set_mode(VaultMode.PRIVATE, persist=False)
        mem.add_memory("SECRET", tags=["hidden"], auto_approve=True)

        result = sync_memories_to_graph(mem.search(), graph)
        assert result.skipped_private == 1
        labels = [n.label for n in graph.search()["nodes"]]
        assert not any("SECRET" in l for l in labels)

    def test_deleted_memory_removes_its_node(self, tmp_env, signal_bus):
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        m = mem.add_memory("Temporary", tags=["x"], auto_approve=True)
        sync_memories_to_graph(mem.search(), graph)
        mem.delete_memory(m.id)
        result = sync_memories_to_graph(mem.search(), graph)
        assert result.removed >= 1
        assert graph.get_stats()["total_nodes"] == 0

    def test_hand_made_nodes_are_not_deleted(self, tmp_env, signal_bus):
        """Sync must only clean up what sync created."""
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        manual = graph.add_node("project", "My Manual Project")
        mem.add_memory("A", tags=["x"], auto_approve=True)
        sync_memories_to_graph(mem.search(), graph)
        sync_memories_to_graph(mem.search(), graph)
        assert graph.get_node(manual.id) is not None

    def test_edited_memory_updates_its_label(self, tmp_env, signal_bus):
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        m = mem.add_memory("Original text", tags=["x"], auto_approve=True)
        sync_memories_to_graph(mem.search(), graph)
        mem.update_memory(m.id, summary="Updated text")
        sync_memories_to_graph(mem.search(), graph)
        labels = [n.label for n in graph.search()["nodes"]]
        assert any("Updated text" in l for l in labels)

    def test_untagged_memory_still_gets_a_node(self, tmp_env, signal_bus):
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        mem.add_memory("No tags here", auto_approve=True)
        result = sync_memories_to_graph(mem.search(), graph)
        assert result.memories_added == 1
        assert result.tags_added == 0

    def test_empty_memory_set_is_safe(self, tmp_env, signal_bus):
        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        result = sync_memories_to_graph(mem.search(), graph)
        assert result.total_changes == 0
        assert "up to date" in result.summary()

    def test_layout_separates_every_node(self, tmp_env, signal_bus):
        """Overlapping nodes made the graph unreadable."""
        import itertools
        import math

        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        for i in range(8):
            mem.add_memory(
                f"Memory number {i}",
                tags=[f"tag{i % 3}", "shared"],
                auto_approve=True,
            )
        sync_memories_to_graph(mem.search(), graph)
        positions = list(graph.compute_layout().values())
        assert len(positions) > 5
        closest = min(
            math.dist(a, b) for a, b in itertools.combinations(positions, 2)
        )
        assert closest > 100, f"nodes only {closest:.1f}px apart"

    def test_layout_is_finite_and_positive(self, tmp_env, signal_bus):
        import math

        from ai_companion.services.graph_sync import sync_memories_to_graph

        _, mem, graph, _ = self._stack(tmp_env, signal_bus)
        for i in range(5):
            mem.add_memory(f"M{i}", tags=["a"], auto_approve=True)
        sync_memories_to_graph(mem.search(), graph)
        for x, y in graph.compute_layout().values():
            assert math.isfinite(x) and math.isfinite(y)
            assert x >= 0 and y >= 0


class TestMemoryRelevance:
    """Injection must send memories related to the question — 2026-09-19."""

    def _mk(self, content, tags=(), pinned=False, conf=0.5):
        from ai_companion.models.memory import Memory, MemoryStatus

        return Memory(
            content=content, summary=content[:80], tags=list(tags),
            status=MemoryStatus.APPROVED, pinned=pinned, confidence=conf,
        )

    def test_relevant_memory_beats_high_confidence_noise(self):
        """The bug: 150 memories delivered the same 11 regardless of query."""
        from ai_companion.services.memory_injection import select_memories

        store = [
            self._mk(f"Gardening fact {i}", ["misc"], conf=0.95)
            for i in range(60)
        ]
        store.append(
            self._mk("Discord bot needs message_content intent",
                     ["discord", "bots"], conf=0.2)
        )
        chosen = select_memories(store, query="why is my discord bot silent?")
        assert any("message_content" in m.content for m in chosen)

    def test_old_behaviour_would_have_missed_it(self):
        """Guard the premise: without a query it really is missed."""
        from ai_companion.services.memory_injection import select_memories

        store = [
            self._mk(f"Gardening fact {i}", ["misc"], conf=0.95)
            for i in range(60)
        ]
        store.append(self._mk("Discord intent fact", ["discord"], conf=0.2))
        chosen = select_memories(store)
        assert not any("Discord intent" in m.content for m in chosen)

    def test_pinned_always_first(self):
        """Pinning means 'always matters'; relevance must not override it."""
        from ai_companion.services.memory_injection import select_memories

        store = [
            self._mk("Highly relevant discord fact", ["discord"], conf=0.9),
            self._mk("Pinned unrelated", ["style"], pinned=True, conf=0.1),
        ]
        chosen = select_memories(store, query="discord bot")
        assert chosen[0].pinned

    def test_tags_outrank_incidental_words(self):
        from ai_companion.services.memory_relevance import score_memory

        tagged = self._mk("Some note", ["python"])
        prose = self._mk("I once mentioned python in passing here")
        assert score_memory(tagged, "python") > score_memory(prose, "python")

    def test_unrelated_memory_scores_zero(self):
        from ai_companion.services.memory_relevance import score_memory

        assert score_memory(self._mk("Gardening", ["garden"]), "discord") == 0

    def test_empty_query_scores_zero(self):
        from ai_companion.services.memory_relevance import score_memory

        assert score_memory(self._mk("anything", ["x"]), "") == 0.0

    def test_stop_words_do_not_match(self):
        """'the' matching everything would make ranking meaningless."""
        from ai_companion.services.memory_relevance import score_memory

        assert score_memory(self._mk("the quick brown fox"), "the") == 0

    def test_partial_tag_match_counts(self):
        from ai_companion.services.memory_relevance import score_memory

        assert score_memory(self._mk("note", ["discord-bot"]), "discord") > 0

    def test_ranking_excludes_pending(self):
        from ai_companion.models.memory import Memory, MemoryStatus
        from ai_companion.services.memory_relevance import rank_memories

        pending = Memory(content="pending discord fact",
                         status=MemoryStatus.PENDING)
        assert rank_memories([pending], "discord") == []

    def test_ranking_is_deterministic(self):
        from ai_companion.services.memory_relevance import rank_memories

        store = [self._mk(f"discord fact {i}", ["discord"]) for i in range(8)]
        first = [m.id for m, _ in rank_memories(store, "discord")]
        second = [m.id for m, _ in rank_memories(store, "discord")]
        assert first == second

    def test_budget_still_respected_with_query(self):
        from ai_companion.services.memory_injection import select_memories

        store = [self._mk("discord " + "x" * 200, ["discord"])
                 for _ in range(50)]
        chosen = select_memories(store, char_budget=600, query="discord")
        assert sum(len(m.content) + 3 for m in chosen) <= 600

    def test_graph_expansion_adds_sibling_tags(self, tmp_env, signal_bus):
        """Asking about 'discord' should reach memories tagged only 'bots'."""
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.graph_service import GraphService
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.memory_relevance import (
            expand_query_with_graph,
        )
        from ai_companion.services.graph_sync import sync_memories_to_graph
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, MemoryService, GraphService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        mem, graph = sm.get("Memory"), sm.get("Graph")
        mem.add_memory("Bot fact", tags=["discord", "bots"], auto_approve=True)
        sync_memories_to_graph(mem.search(), graph)

        expanded = expand_query_with_graph("discord", graph)
        assert "bots" in expanded

    def test_graph_expansion_never_raises(self):
        from ai_companion.services.memory_relevance import (
            expand_query_with_graph,
        )

        class Broken:
            def search(self):
                raise RuntimeError("boom")

        assert expand_query_with_graph("x", Broken()) == "x"
        assert expand_query_with_graph("x", None) == "x"


class TestCodeWorkspacePrivacy:
    """Snapshots must respect private mode — 2026-09-19."""

    def _stack(self, tmp_env, signal_bus, private):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.code_workspace import CodeWorkspaceService
        from ai_companion.services.vault_service import VaultService

        config, root = tmp_env
        config.vault.mode = VaultMode.PRIVATE if private else VaultMode.NORMAL
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, CodeWorkspaceService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        return sm.get("CodeWorkspace"), root

    def test_snapshot_refused_in_private_mode(self, tmp_env, signal_bus):
        cw, _ = self._stack(tmp_env, signal_bus, private=True)
        cw.write_code("secret.py", "API_KEY='hunter2'\n")
        assert cw.create_snapshot("nope") == ""
        assert cw.list_snapshots() == []

    def test_code_never_reaches_disk_in_private_mode(
        self, tmp_env, signal_bus
    ):
        """The actual privacy guarantee, checked against real files."""
        import os

        cw, root = self._stack(tmp_env, signal_bus, private=True)
        cw.write_code("secret.py", "API_KEY='hunter2'\n")
        cw.create_snapshot("nope")

        leaked = False
        for dirpath, _, files in os.walk(root):
            if "snapshot" not in dirpath:
                continue
            for name in files:
                try:
                    text = open(os.path.join(dirpath, name)).read()
                except OSError:
                    continue
                if "hunter2" in text:
                    leaked = True
        assert leaked is False

    def test_snapshot_works_in_normal_mode(self, tmp_env, signal_bus):
        """Guard the fix: refusing always would be a worse bug."""
        cw, _ = self._stack(tmp_env, signal_bus, private=False)
        cw.write_code("app.py", "print('hi')\n")
        assert cw.create_snapshot("ok") != ""
        assert len(cw.list_snapshots()) == 1

    def test_uses_the_shared_privacy_rule(self):
        """One implementation, not a fifth divergent copy."""
        from ai_companion.infrastructure.privacy import PrivacyMixin
        from ai_companion.services.code_workspace import CodeWorkspaceService

        assert issubclass(CodeWorkspaceService, PrivacyMixin)
        assert "is_private_mode" not in CodeWorkspaceService.__dict__


class TestIrrelevantMemoriesAreDropped:
    """Unrelated memories must not spend context — 2026-09-19."""

    def _mk(self, content, tags=(), pinned=False, conf=0.5):
        from ai_companion.models.memory import Memory, MemoryStatus

        return Memory(
            content=content, summary=content[:80], tags=list(tags),
            status=MemoryStatus.APPROVED, pinned=pinned, confidence=conf,
        )

    def test_zero_scoring_memories_excluded(self):
        """40 gardening notes must not crowd out the conversation."""
        from ai_companion.services.memory_injection import select_memories

        store = [self._mk(f"Gardening note {i}", ["garden"], conf=0.95)
                 for i in range(40)]
        store.append(self._mk("Discord intent fact", ["discord"], conf=0.2))
        chosen = select_memories(store, query="discord bot problem")
        assert len(chosen) == 1
        assert "Discord" in chosen[0].content

    def test_pinned_survive_even_when_irrelevant(self):
        from ai_companion.services.memory_injection import select_memories

        store = [
            self._mk("Discord intent fact", ["discord"]),
            self._mk("Terse answers please", ["style"], pinned=True),
        ]
        chosen = select_memories(store, query="discord bot")
        assert any(m.pinned for m in chosen)

    def test_unmatched_query_still_sends_something(self):
        """A general question should not produce an empty memory block."""
        from ai_companion.services.memory_injection import select_memories

        store = [self._mk(f"Fact {i}", ["misc"]) for i in range(5)]
        chosen = select_memories(store, query="hello there friend")
        assert len(chosen) > 0

    def test_no_query_keeps_old_behaviour(self):
        from ai_companion.services.memory_injection import select_memories

        store = [self._mk(f"Fact {i}", ["misc"]) for i in range(5)]
        assert len(select_memories(store)) == 5


class TestMemoryExtraction:
    """Candidates must be durable facts, not noise — 2026-09-19."""

    def test_explicit_remember_is_captured(self):
        from ai_companion.services.memory_extraction import extract_candidates

        found = extract_candidates("remember that my timezone is New_York")
        assert found
        assert "timezone" in found[0].content

    def test_dot_inside_a_token_is_not_a_sentence_end(self):
        """The old extractor produced 'my bot token is in .'"""
        from ai_companion.services.memory_extraction import extract_candidates

        found = extract_candidates("remember that my bot token is in .env")
        assert found
        assert found[0].content.endswith(".env")

    def test_version_numbers_survive(self):
        from ai_companion.services.memory_extraction import extract_candidates

        found = extract_candidates("my python version is 3.12")
        assert found
        assert "3.12" in found[0].content

    def test_transient_statements_rejected(self):
        """'the error is X' is true for ten minutes, not forever."""
        from ai_companion.services.memory_extraction import extract_candidates

        for text in (
            "the error is happening on line 42",
            "the problem is that my code is broken",
            "this bug is driving me crazy",
            "the issue is my function returns None",
        ):
            assert extract_candidates(text) == [], text

    def test_questions_produce_nothing(self):
        from ai_companion.services.memory_extraction import extract_candidates

        for text in (
            "what is the capital of France?",
            "can you explain how this works",
            "how do I fix this",
        ):
            assert extract_candidates(text) == [], text

    def test_vacuous_preferences_rejected(self):
        """'Prefers it to work' tells the model nothing."""
        from ai_companion.services.memory_extraction import extract_candidates

        assert extract_candidates("the thing is i want it to work") == []

    def test_one_bad_clause_does_not_suppress_a_good_fact(self):
        """Transience is judged per sentence, not per message."""
        from ai_companion.services.memory_extraction import extract_candidates

        found = extract_candidates(
            "my python version is 3.12. also the bug is annoying"
        )
        assert len(found) == 1
        assert "3.12" in found[0].content

    def test_duplicates_within_one_message_collapse(self):
        from ai_companion.services.memory_extraction import extract_candidates

        found = extract_candidates("i prefer terse answers, i prefer terse answers")
        assert len(found) == 1

    def test_existing_memories_are_not_reproposed(self):
        from ai_companion.models.memory import Memory, MemoryStatus
        from ai_companion.services.memory_extraction import extract_candidates

        existing = [Memory(content="my timezone is New_York",
                           status=MemoryStatus.APPROVED)]
        found = extract_candidates(
            "remember that my timezone is New_York", existing
        )
        assert found == []

    def test_tags_are_suggested(self):
        from ai_companion.services.memory_extraction import extract_candidates

        found = extract_candidates("remember that my discord bot uses python")
        assert found
        assert "discord" in found[0].tags

    def test_empty_input_is_safe(self):
        from ai_companion.services.memory_extraction import extract_candidates

        assert extract_candidates("") == []
        assert extract_candidates("   ") == []

    # ---- through the service ----

    def _stack(self, tmp_env, signal_bus, auto=True, private=False):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService

        config, root = tmp_env
        config.vault.mode = VaultMode.PRIVATE if private else VaultMode.NORMAL
        config.memory.auto_extract = auto
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, MemoryService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        return sm, sm.get("Memory")

    def test_candidates_are_pending_never_approved(self, tmp_env, signal_bus):
        """A misread sentence must not become something the model believes."""
        _, mem = self._stack(tmp_env, signal_bus)
        created = mem.extract_candidates("remember that my timezone is UTC")
        assert created
        assert all(m.status.value == "pending" for m in created)

    def test_nothing_extracted_in_private_mode(self, tmp_env, signal_bus):
        _, mem = self._stack(tmp_env, signal_bus, private=True)
        assert mem.extract_candidates("remember that my timezone is UTC") == []

    def test_pending_candidate_never_reaches_the_model(self):
        from ai_companion.models.memory import Memory, MemoryStatus
        from ai_companion.services.memory_injection import build_memory_block

        pending = Memory(content="my timezone is UTC",
                         status=MemoryStatus.PENDING)
        assert build_memory_block([pending]) == ""

    def test_extraction_is_off_by_default(self):
        from ai_companion.models.config_models import AppConfig

        assert AppConfig().memory.auto_extract is False


class TestMindMapStructure:
    """Concepts are nodes, memories are evidence — 2026-09-19."""

    def _mk(self, content, tags=(), private=False, status="approved"):
        from ai_companion.models.memory import Memory, MemoryStatus

        return Memory(content=content, summary=content[:80], tags=list(tags),
                      status=MemoryStatus(status), private=private)

    def _grandpa(self):
        return [
            self._mk("Grandpa specialized in satellites and NASA contracts",
                     ["family", "space"]),
            self._mk("Mom moved a lot because of Grandpa's job", ["family"]),
            self._mk("My interest in space came from Grandpa",
                     ["space", "engineering"]),
        ]

    def test_tags_become_concepts(self):
        from ai_companion.services.mind_map import build_mind_map

        mm = build_mind_map(self._grandpa())
        labels = {c.label for c in mm.concepts.values()}
        assert "family" in labels
        assert "space" in labels

    def test_repeated_names_become_concepts(self):
        """'Hector' should appear without anyone tagging it."""
        from ai_companion.services.mind_map import build_mind_map

        mm = build_mind_map(self._grandpa())
        assert any(c.label == "Grandpa" for c in mm.concepts.values())

    def test_single_mention_is_not_a_concept(self):
        """One passing mention must not create a bubble."""
        from ai_companion.services.mind_map import build_mind_map

        mm = build_mind_map([self._mk("I met Zebediah once", ["misc"])])
        assert not any(c.label == "Zebediah" for c in mm.concepts.values())

    def test_weight_counts_memories(self):
        from ai_companion.services.mind_map import build_mind_map

        mm = build_mind_map(self._grandpa())
        grandpa = next(c for c in mm.concepts.values() if c.label == "Grandpa")
        assert grandpa.weight == 3

    def test_links_come_from_shared_memories(self):
        from ai_companion.services.mind_map import build_mind_map

        mm = build_mind_map(self._grandpa())
        grandpa = next(c for c in mm.concepts.values() if c.label == "Grandpa")
        family = next(c for c in mm.concepts.values() if c.label == "family")
        link = next(
            (l for l in mm.links
             if {l.source, l.target} == {grandpa.key, family.key}), None
        )
        assert link is not None
        assert link.shared == 2

    def test_radius_uses_sqrt_not_linear(self):
        """Linear radius makes a 10-memory concept look 10x as important."""
        from ai_companion.services.mind_map import Concept

        small = Concept(key="a", label="a", memory_ids=["1"])
        big = Concept(key="b", label="b", memory_ids=[str(i) for i in range(16)])
        r_small = small.radius(max_weight=16)
        r_big = big.radius(max_weight=16)
        assert r_big > r_small
        assert r_big < r_small * 16

    def test_private_memories_excluded(self):
        from ai_companion.services.mind_map import build_mind_map

        mems = self._grandpa()
        mems.append(self._mk("SECRET about Vesper", ["secret"], private=True))
        mm = build_mind_map(mems)
        blob = " ".join(c.label for c in mm.concepts.values())
        assert "Vesper" not in blob
        assert "secret" not in blob
        assert mm.skipped_private == 1

    def test_pending_memories_excluded(self):
        from ai_companion.services.mind_map import build_mind_map

        mm = build_mind_map([self._mk("guess", ["x"], status="pending")])
        assert mm.concepts == {}

    def test_people_are_categorised(self):
        from ai_companion.services.mind_map import CATEGORY_PERSON, build_mind_map

        mm = build_mind_map(self._grandpa())
        grandpa = next(c for c in mm.concepts.values() if c.label == "Grandpa")
        assert grandpa.category == CATEGORY_PERSON

    def test_empty_input_is_safe(self):
        from ai_companion.services.mind_map import build_mind_map

        mm = build_mind_map([])
        assert mm.concepts == {}
        assert "No concepts" in mm.summary()

    def test_build_is_deterministic(self):
        from ai_companion.services.mind_map import build_mind_map

        mems = self._grandpa()
        first = [c.label for c in build_mind_map(mems).ordered()]
        second = [c.label for c in build_mind_map(mems).ordered()]
        assert first == second


class TestMindMapLayout:
    """Layout must be readable, not just correct — 2026-09-19."""

    def _many(self, count=16):
        from ai_companion.models.memory import Memory, MemoryStatus

        out = []
        for i in range(count):
            out.append(Memory(
                content=f"Fact {i} about thing {i % 4}",
                summary=f"Fact {i}",
                tags=[f"tag{i % 5}", "shared"],
                status=MemoryStatus.APPROVED,
            ))
        return out

    def test_bubbles_never_overlap(self):
        """Overlapping bubbles make the map unreadable."""
        import itertools
        import math

        from ai_companion.services.mind_map import build_mind_map, layout_mind_map

        mm = build_mind_map(self._many())
        pos = layout_mind_map(mm)
        radii = {
            k: c.radius(max_weight=mm.max_weight)
            for k, c in mm.concepts.items()
        }
        for a, b in itertools.combinations(pos, 2):
            gap = math.dist(pos[a], pos[b]) - (radii[a] + radii[b])
            assert gap > 0, f"{a} and {b} overlap by {-gap:.1f}px"

    def test_span_stays_bounded(self):
        """Unbounded repulsion once produced a 20000px span for 16 bubbles."""
        from ai_companion.services.mind_map import build_mind_map, layout_mind_map

        mm = build_mind_map(self._many())
        pos = layout_mind_map(mm)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        assert max(xs) - min(xs) < 4000
        assert max(ys) - min(ys) < 4000

    def test_coordinates_are_finite_and_positive(self):
        import math

        from ai_companion.services.mind_map import build_mind_map, layout_mind_map

        mm = build_mind_map(self._many())
        for x, y in layout_mind_map(mm).values():
            assert math.isfinite(x) and math.isfinite(y)
            assert x >= 0 and y >= 0

    def test_empty_layout_is_safe(self):
        from ai_companion.services.mind_map import MindMap, layout_mind_map

        assert layout_mind_map(MindMap()) == {}

    def test_single_concept_layout(self):
        from ai_companion.services.mind_map import build_mind_map, layout_mind_map
        from ai_companion.models.memory import Memory, MemoryStatus

        mm = build_mind_map([Memory(content="solo", tags=["only"],
                                    status=MemoryStatus.APPROVED)])
        assert len(layout_mind_map(mm)) == 1


class TestMindMapSynthesis:
    """Overviews are on demand, cached, and go stale — 2026-09-19."""

    def _stack(self, tmp_env, signal_bus, reply="A connected overview."):
        import sys
        import types

        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.mind_map_service import MindMapService
        from ai_companion.services.vault_service import VaultService

        class _Llama:
            def __init__(self, **kwargs):
                pass

            def create_chat_completion(self, messages, **kwargs):
                return {"choices": [{"message": {"content": reply}}]}

        stub = types.ModuleType("llama_cpp")
        stub.Llama = _Llama
        sys.modules["llama_cpp"] = stub

        config, root = tmp_env
        config.vault.mode = VaultMode.NORMAL
        model = root / "Q.gguf"
        model.write_bytes(b"\x00" * 2048)
        config.llm.model_path = str(model)
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, MemoryService, LlmService, MindMapService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        return sm, sm.get("Memory"), sm.get("MindMap")

    def _seed(self, mem):
        for content, tags in [
            ("Grandpa worked on NASA satellite contracts", ["family", "space"]),
            ("Mom moved a lot because of Grandpa's job", ["family"]),
            ("My interest in space came from Grandpa", ["space"]),
        ]:
            mem.add_memory(content, tags=tags, auto_approve=True)

    def _key(self, mms, label="Grandpa"):
        return next(c.key for c in mms.map.ordered() if c.label == label)

    def test_overview_is_written_and_cached(self, tmp_env, signal_bus):
        _, mem, mms = self._stack(tmp_env, signal_bus)
        self._seed(mem)
        mms.build()
        key = self._key(mms)
        result = mms.synthesise(key)
        assert result.ok
        assert mms.map.concepts[key].overview
        assert not mms.map.concepts[key].overview_stale

    def test_overview_goes_stale_when_memories_change(
        self, tmp_env, signal_bus
    ):
        """Serving a summary of facts that have changed would mislead."""
        _, mem, mms = self._stack(tmp_env, signal_bus)
        self._seed(mem)
        mms.build()
        key = self._key(mms)
        mms.synthesise(key)
        mem.add_memory("Grandpa retired in 1998", tags=["family"],
                       auto_approve=True)
        mms.build()
        assert mms.map.concepts[key].overview_stale is True

    def test_stale_overviews_are_not_injected(self, tmp_env, signal_bus):
        _, mem, mms = self._stack(tmp_env, signal_bus)
        self._seed(mem)
        mms.build()
        key = self._key(mms)
        mms.synthesise(key)
        assert mms.overviews_for_query("grandpa")
        mem.add_memory("Grandpa retired in 1998", tags=["family"],
                       auto_approve=True)
        mms.build()
        assert mms.overviews_for_query("grandpa") == []

    def test_fresh_overview_is_injected_for_matching_query(
        self, tmp_env, signal_bus
    ):
        _, mem, mms = self._stack(tmp_env, signal_bus)
        self._seed(mem)
        mms.build()
        mms.synthesise(self._key(mms))
        found = mms.overviews_for_query("tell me about grandpa")
        assert found
        assert found[0][0] == "Grandpa"

    def test_unrelated_query_gets_no_overview(self, tmp_env, signal_bus):
        _, mem, mms = self._stack(tmp_env, signal_bus)
        self._seed(mem)
        mms.build()
        mms.synthesise(self._key(mms))
        assert mms.overviews_for_query("docker kubernetes") == []

    def test_synthesis_without_a_model_fails_cleanly(
        self, tmp_env, signal_bus
    ):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.mind_map_service import MindMapService
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, MemoryService, MindMapService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        mem, mms = sm.get("Memory"), sm.get("MindMap")
        self._seed(mem)
        mms.build()
        result = mms.synthesise(self._key(mms))
        assert result.ok is False
        assert "model" in result.error.lower()

    def test_unknown_concept_fails_cleanly(self, tmp_env, signal_bus):
        _, _, mms = self._stack(tmp_env, signal_bus)
        mms.build()
        assert mms.synthesise("tag:nonexistent").ok is False

    def test_fingerprint_changes_with_membership(self):
        from ai_companion.services.mind_map_synthesis import fingerprint

        assert fingerprint(["a", "b"]) == fingerprint(["b", "a"])
        assert fingerprint(["a", "b"]) != fingerprint(["a", "b", "c"])

    def test_prompt_caps_fact_count(self):
        """Prompt length is the main driver of generation time on CPU."""
        from ai_companion.models.memory import Memory, MemoryStatus
        from ai_companion.services.mind_map_synthesis import build_prompt

        mems = [Memory(content=f"Fact {i}", status=MemoryStatus.APPROVED)
                for i in range(40)]
        prompt = build_prompt("Thing", mems, max_facts=5)
        assert prompt.count("- Fact") == 5


class TestMindMapInjection:
    """Overviews assist memories, never replace them — 2026-09-19."""

    def _mk(self, content, tags=()):
        from ai_companion.models.memory import Memory, MemoryStatus

        return Memory(content=content, summary=content[:80], tags=list(tags),
                      status=MemoryStatus.APPROVED)

    def test_overview_and_memories_both_present(self):
        from ai_companion.services.memory_injection import compose_system_prompt

        out = compose_system_prompt(
            "BE TERSE",
            [self._mk("Grandpa worked on NASA contracts", ["family"])],
            query="grandpa",
            overviews=[("Grandpa", "An engineer who inspired the user.")],
        )
        assert "BE TERSE" in out
        assert "An engineer who inspired" in out
        assert "NASA contracts" in out

    def test_base_prompt_comes_first(self):
        from ai_companion.services.memory_injection import compose_system_prompt

        out = compose_system_prompt(
            "BE TERSE", [self._mk("fact", ["x"])], query="x",
            overviews=[("X", "overview text")],
        )
        assert out.index("BE TERSE") < out.index("overview text")

    def test_overviews_cannot_consume_whole_budget(self):
        """The map must not crowd out the memories it is meant to assist."""
        from ai_companion.services.memory_injection import compose_system_prompt

        huge = [("Concept" + str(i), "x" * 400) for i in range(10)]
        out = compose_system_prompt(
            "", [self._mk("IMPORTANT MEMORY", ["x"])],
            char_budget=1200, query="x", overviews=huge,
        )
        assert "IMPORTANT MEMORY" in out

    def test_no_overviews_behaves_as_before(self):
        from ai_companion.services.memory_injection import compose_system_prompt

        out = compose_system_prompt(
            "BASE", [self._mk("a fact", ["x"])], query="x"
        )
        assert "Big-picture" not in out
        assert "a fact" in out


class TestSetModeCannotClobberRealConfig:
    """Regression: installing a patch wiped the user's settings.

    Lives here for the tmp_env / signal_bus fixtures.
    """

    def test_set_mode_cannot_reach_the_real_file(self, tmp_env, signal_bus):
        """The exact call that caused the bug, run unguarded."""
        from pathlib import Path

        import ai_companion
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        real = (
            Path(ai_companion.__file__).resolve().parent.parent / "config.json"
        )
        before = real.read_bytes() if real.exists() else None

        config, _ = tmp_env
        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE)   # persists by default
        vault.stop()

        after = real.read_bytes() if real.exists() else None
        assert before == after


class TestSpeechTextCleaning:
    """Text must be cleaned before it is read aloud — 2026-09-19."""

    def test_code_blocks_become_a_spoken_marker(self):
        """Reading 200 lines of Python aloud is useless and takes minutes."""
        from ai_companion.services.speech_engine import clean_for_speech

        spoken = clean_for_speech(
            "Here:\n```python\nimport discord\nx = 1\n```\nDone."
        )
        assert "import discord" not in spoken
        assert "code on screen" in spoken

    def test_unterminated_code_fence_handled(self):
        """Mid-stream the closing fence has not arrived yet."""
        from ai_companion.services.speech_engine import clean_for_speech

        spoken = clean_for_speech("Here:\n```python\nimport discord")
        assert "import discord" not in spoken

    def test_markdown_is_stripped(self):
        from ai_companion.services.speech_engine import clean_for_speech

        spoken = clean_for_speech("Use `pip` and **care**")
        assert "`" not in spoken and "*" not in spoken

    def test_urls_are_not_read_out(self):
        from ai_companion.services.speech_engine import clean_for_speech

        spoken = clean_for_speech("See https://example.com/a/b/c now")
        assert "https" not in spoken
        assert "a link" in spoken

    def test_no_double_punctuation(self):
        from ai_companion.services.speech_engine import clean_for_speech

        spoken = clean_for_speech("Fix:\n```py\nx=1\n```\nDone.")
        assert ". ." not in spoken
        assert ".." not in spoken

    def test_long_text_is_truncated_at_a_sentence(self):
        from ai_companion.services.speech_engine import clean_for_speech

        long_text = ". ".join(f"Sentence number {i}" for i in range(200))
        spoken = clean_for_speech(long_text, max_chars=200)
        assert len(spoken) < 300
        assert "on screen" in spoken

    def test_empty_input_is_safe(self):
        from ai_companion.services.speech_engine import clean_for_speech

        assert clean_for_speech("") == ""
        assert clean_for_speech("   ") == ""

    def test_code_only_reply_still_says_something(self):
        from ai_companion.services.speech_engine import clean_for_speech

        spoken = clean_for_speech("```python\nx = 1\n```")
        assert "code on screen" in spoken


class TestWakeWord:
    """Loose matching, because Whisper mishears — 2026-09-19."""

    def test_exact_match(self):
        from ai_companion.services.speech_engine import contains_wake_word

        assert contains_wake_word("Jarvis what time is it", "jarvis")

    def test_punctuation_and_case_ignored(self):
        from ai_companion.services.speech_engine import contains_wake_word

        assert contains_wake_word("JARVIS!", "jarvis")
        assert contains_wake_word("jarvis, status", "jarvis")

    def test_common_mishearings_accepted(self):
        """Whisper writes 'Travis' or 'Charvis' for 'Jarvis' routinely."""
        from ai_companion.services.speech_engine import contains_wake_word

        assert contains_wake_word("Travis how are you", "jarvis")
        assert contains_wake_word("Charvis hello", "jarvis")

    def test_unrelated_speech_rejected(self):
        from ai_companion.services.speech_engine import contains_wake_word

        for text in (
            "tell me about python",
            "what is the capital of France",
            "service the car",
            "harvest the crops",
        ):
            assert not contains_wake_word(text, "jarvis"), text

    def test_empty_wake_word_always_matches(self):
        """No wake word configured means everything is addressed to us."""
        from ai_companion.services.speech_engine import contains_wake_word

        assert contains_wake_word("anything at all", "")

    def test_wake_word_is_stripped_from_the_prompt(self):
        from ai_companion.services.speech_engine import strip_wake_word

        assert strip_wake_word("Jarvis, what time is it", "jarvis") == \
            "what time is it"

    def test_stripping_never_returns_empty(self):
        """'Jarvis' alone must not become an empty prompt."""
        from ai_companion.services.speech_engine import strip_wake_word

        assert strip_wake_word("Jarvis", "jarvis").strip()

    def test_edit_distance_helper(self):
        from ai_companion.services.speech_engine import _edit_distance_within

        assert _edit_distance_within("jarvis", "jarvis", 1)
        assert _edit_distance_within("charvis", "jarvis", 2)
        assert not _edit_distance_within("elephant", "jarvis", 2)


class TestVoiceService:
    """The hands-free loop — 2026-09-19."""

    def _stack(self, tmp_env, signal_bus):
        import sys
        import types

        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.vault_service import VaultService
        from ai_companion.services.voice_service import VoiceService

        # No audio hardware in CI; stub the device layer.
        if not isinstance(sys.modules.get("sounddevice"), types.ModuleType) \
                or not hasattr(sys.modules.get("sounddevice", None), "_stubbed"):
            stub = types.ModuleType("sounddevice")
            stub._stubbed = True

            class _Stream:
                def __init__(self, **kwargs):
                    pass

                def start(self):
                    pass

                def stop(self):
                    pass

                def close(self):
                    pass

            stub.InputStream = _Stream
            stub.play = lambda audio, rate: None
            stub.wait = lambda: None
            sys.modules["sounddevice"] = stub

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, VoiceService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        return sm, sm.get("Voice")

    def test_starts_disabled(self, tmp_env, signal_bus):
        from ai_companion.services.voice_service import VoiceState

        _, voice = self._stack(tmp_env, signal_bus)
        assert voice.state == VoiceState.OFF
        assert voice.enabled is False

    def test_enable_without_models_fails_cleanly(self, tmp_env, signal_bus):
        """No downloaded models must not crash the app."""
        config, _ = tmp_env
        config.speech.whisper_model = "definitely-not-a-model"
        _, voice = self._stack(tmp_env, signal_bus)
        ok, error = voice.enable()
        assert ok is False
        assert error

    def test_disable_is_idempotent(self, tmp_env, signal_bus):
        _, voice = self._stack(tmp_env, signal_bus)
        voice.disable()
        voice.disable()   # must not raise

    def test_available_voices_when_none_installed(self, tmp_env, signal_bus):
        config, root = tmp_env
        config.speech.voices_dir = str(root / "no_voices_here")
        _, voice = self._stack(tmp_env, signal_bus)
        assert voice.available_voices() == []

    def test_state_changes_are_signalled(self, tmp_env, signal_bus):
        """A voice UI that cannot see the real state is unusable."""
        from ai_companion.services.voice_service import VoiceState

        seen = []
        signal_bus.speech.state_changed.connect(seen.append)
        _, voice = self._stack(tmp_env, signal_bus)
        voice._set_state(VoiceState.CAPTURING)
        voice._set_state(VoiceState.IDLE)
        assert seen == ["capturing", "idle"]

    def test_repeated_state_does_not_re_signal(self, tmp_env, signal_bus):
        from ai_companion.services.voice_service import VoiceState

        seen = []
        _, voice = self._stack(tmp_env, signal_bus)
        signal_bus.speech.state_changed.connect(seen.append)
        voice._set_state(VoiceState.IDLE)
        voice._set_state(VoiceState.IDLE)
        assert seen == ["idle"]


class TestMicrophoneListener:
    """Utterance detection — 2026-09-19."""

    def _listener(self, captured):
        from ai_companion.services.speech_engine import MicrophoneListener

        return MicrophoneListener(
            on_utterance=captured.append,
            silence_threshold=0.05,
            silence_seconds=0.3,
        )

    def test_pause_clears_the_buffer(self, ):
        """Pausing mid-utterance must not leave half a sentence queued."""
        captured = []
        listener = self._listener(captured)
        listener._speaking = True
        listener._buffer = ["fake"]
        listener.pause()
        assert listener._buffer == []
        assert listener._speaking is False

    def test_resume_clears_the_pause_flag(self):
        captured = []
        listener = self._listener(captured)
        listener.pause()
        assert listener._paused is True
        listener.resume()
        assert listener._paused is False

    def test_stop_is_safe_when_never_started(self):
        captured = []
        self._listener(captured).stop()   # must not raise


class TestMicrophoneSelection:
    """Device picking and testing — 2026-09-19."""

    def test_device_listing_survives_missing_portaudio(self, monkeypatch):
        """sounddevice raises OSError, not ImportError, without PortAudio.

        Catching only ImportError crashed Settings on a fresh machine.
        """
        import builtins

        from ai_companion.services import speech_engine

        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        assert speech_engine.list_input_devices() == []
        assert speech_engine.default_input_device() is None

    def test_device_listing_survives_missing_library(self, monkeypatch):
        import builtins

        from ai_companion.services import speech_engine

        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "sounddevice":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        assert speech_engine.list_input_devices() == []

    def test_mic_test_reports_the_reason(self, monkeypatch):
        """A pass/fail light tells the user nothing actionable."""
        import builtins

        from ai_companion.services import speech_engine

        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", boom)
        result = speech_engine.test_microphone(seconds=0.1)
        assert result.ok is False
        assert "PortAudio" in result.error

    def test_silent_input_is_diagnosed(self, monkeypatch):
        import sys
        import types

        import numpy as np

        from ai_companion.services import speech_engine

        stub = types.ModuleType("sounddevice")
        stub.rec = lambda frames, **kwargs: np.zeros((frames, 1), dtype="float32")
        stub.wait = lambda: None
        monkeypatch.setitem(sys.modules, "sounddevice", stub)

        result = speech_engine.test_microphone(seconds=0.1)
        assert result.ok is False
        assert "silent" in result.verdict.lower()

    def test_quiet_input_suggests_raising_gain(self, monkeypatch):
        import sys
        import types

        import numpy as np

        from ai_companion.services import speech_engine

        stub = types.ModuleType("sounddevice")
        stub.rec = lambda frames, **kwargs: (
            np.full((frames, 1), 0.002, dtype="float32")
        )
        stub.wait = lambda: None
        monkeypatch.setitem(sys.modules, "sounddevice", stub)

        result = speech_engine.test_microphone(seconds=0.1, threshold=0.012)
        assert result.ok is True
        assert "quiet" in result.verdict.lower()

    def test_good_input_passes(self, monkeypatch):
        import sys
        import types

        import numpy as np

        from ai_companion.services import speech_engine

        stub = types.ModuleType("sounddevice")
        rng = np.random.default_rng(0)
        stub.rec = lambda frames, **kwargs: (
            rng.normal(0, 0.1, (frames, 1)).astype("float32")
        )
        stub.wait = lambda: None
        monkeypatch.setitem(sys.modules, "sounddevice", stub)

        result = speech_engine.test_microphone(seconds=0.1, threshold=0.012)
        assert result.ok is True
        assert "good" in result.verdict.lower()

    def test_listener_accepts_a_device_index(self):
        from ai_companion.services.speech_engine import MicrophoneListener

        listener = MicrophoneListener(lambda audio: None, device_index=3)
        assert listener._device_index == 3

    def test_config_stores_the_chosen_device(self):
        from ai_companion.models.config_models import AppConfig

        config = AppConfig()
        assert config.speech.input_device_index is None
        config.speech.input_device_index = 2
        assert AppConfig(**config.model_dump()).speech.input_device_index == 2


class TestSetModeWritesOnlyTheMode:
    """set_mode must not rewrite store paths — 2026-09-19.

    An older build saved the service's whole in-memory AppConfig. Under test
    that object has every path relocated to a pytest temp folder, so running
    the suite rewrote the real config to point at directories Windows later
    deletes. The app then found no memories and appeared to have wiped them.
    """

    def _vault(self, tmp_env, signal_bus, config_path):
        from ai_companion import config as config_module
        from ai_companion.services.vault_service import VaultService

        config, _ = tmp_env
        config_module.DEFAULT_CONFIG_PATH = str(config_path)
        vault = VaultService(config, signal_bus)
        vault.start()
        return vault

    def test_store_paths_are_not_rewritten(
        self, tmp_env, signal_bus, tmp_path, monkeypatch
    ):
        """The actual bug: temp paths leaking into the user's config."""
        import json

        from ai_companion import config as config_module
        from ai_companion.models.config_models import VaultMode

        target = tmp_path / "real_config.json"
        target.write_text(json.dumps({
            "llm": {"model_path": "C:/models/Qwen.gguf"},
            "memory": {"store_path": "data/memories.json"},
            "vault": {"mode": "normal"},
        }), encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )

        config, _ = tmp_env   # this config has temp paths in it
        from ai_companion.services.vault_service import VaultService

        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE)
        vault.stop()

        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["memory"]["store_path"] == "data/memories.json"
        assert "pytest" not in saved["memory"]["store_path"]
        assert "data_dir" not in saved or "pytest" not in str(
            saved.get("data_dir", "")
        )

    def test_unrelated_settings_are_preserved(
        self, tmp_env, signal_bus, tmp_path, monkeypatch
    ):
        import json

        from ai_companion import config as config_module
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        target = tmp_path / "real_config.json"
        target.write_text(json.dumps({
            "llm": {"model_path": "C:/models/Qwen.gguf", "max_tokens": 1536},
            "vault": {"mode": "normal"},
        }), encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )

        config, _ = tmp_env
        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE)
        vault.stop()

        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["llm"]["model_path"] == "C:/models/Qwen.gguf"
        assert saved["llm"]["max_tokens"] == 1536

    def test_the_mode_really_is_written(
        self, tmp_env, signal_bus, tmp_path, monkeypatch
    ):
        """Guard the fix: writing nothing at all would be worse."""
        import json

        from ai_companion import config as config_module
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        target = tmp_path / "real_config.json"
        target.write_text(json.dumps({"vault": {"mode": "normal"}}), "utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )

        config, _ = tmp_env
        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE)
        vault.stop()

        assert json.loads(target.read_text("utf-8"))["vault"]["mode"] == \
            "private"

    def test_corrupt_config_does_not_crash_the_toggle(
        self, tmp_env, signal_bus, tmp_path, monkeypatch
    ):
        from ai_companion import config as config_module
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        target = tmp_path / "real_config.json"
        target.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )

        config, _ = tmp_env
        vault = VaultService(config, signal_bus)
        vault.start()
        vault.set_mode(VaultMode.PRIVATE)   # must not raise
        vault.stop()
        assert vault.is_private is True


class TestWakeWordFeedback:
    """A rejected utterance must explain itself — 2026-09-19.

    Speech was transcribed perfectly, then silently dropped for lacking the
    wake word. With no feedback that is indistinguishable from a broken app.
    """

    def _stack(self, tmp_env, signal_bus, follow_up=12.0):
        import sys
        import types

        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.vault_service import VaultService
        from ai_companion.services.voice_service import VoiceService

        stub = types.ModuleType("sounddevice")

        class _Stream:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def close(self):
                pass

        stub.InputStream = _Stream
        stub.play = lambda audio, rate: None
        stub.wait = lambda: None
        sys.modules["sounddevice"] = stub

        llama = types.ModuleType("llama_cpp")

        class _Llama:
            def __init__(self, **kwargs):
                pass

            def create_chat_completion(self, messages, **kwargs):
                return iter([{"choices": [{"delta": {"content": "ok"}}]}])

        llama.Llama = _Llama
        sys.modules["llama_cpp"] = llama

        config, root = tmp_env
        config.speech.follow_up_seconds = follow_up
        model = root / "Q.gguf"
        model.write_bytes(b"\x00" * 2048)
        config.llm.model_path = str(model)
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        for cls in (VaultService, LlmService, VoiceService):
            sm.register(cls(config, signal_bus))
        sm.start_all()
        return sm, sm.get("Voice")

    def _drive(self, voice, text):
        import numpy as np

        from ai_companion.services.speech_engine import TranscriptionResult
        from ai_companion.services.voice_service import VoiceState

        class _Fake:
            def load(self):
                return True, ""

            def transcribe(self, audio):
                return TranscriptionResult(True, text=text)

        voice._transcriber = _Fake()
        voice._state = VoiceState.IDLE
        voice._on_utterance(np.zeros(16000, dtype="float32"))

    def test_missing_wake_word_is_reported(self, tmp_env, signal_bus):
        import time

        ignored = []
        signal_bus.speech.utterance_ignored.connect(
            lambda heard, reason: ignored.append((heard, reason))
        )
        _, voice = self._stack(tmp_env, signal_bus)
        self._drive(voice, "How good is your code?")
        time.sleep(0.3)
        assert ignored
        assert "wake word" in ignored[0][1]
        assert "How good is your code?" == ignored[0][0]

    def test_wake_word_utterance_is_sent(self, tmp_env, signal_bus):
        import time

        prompts = []
        signal_bus.speech.prompt_ready.connect(
            lambda conv, prompt: prompts.append(prompt)
        )
        _, voice = self._stack(tmp_env, signal_bus)
        self._drive(voice, "Jarvis how good is your code")
        time.sleep(0.3)
        assert prompts == ["how good is your code"]

    def test_follow_up_needs_no_wake_word(self, tmp_env, signal_bus):
        """Demanding the wake word every sentence is dictation, not talking."""
        import time

        prompts = []
        signal_bus.speech.prompt_ready.connect(
            lambda conv, prompt: prompts.append(prompt)
        )
        _, voice = self._stack(tmp_env, signal_bus)
        voice._follow_up_until = time.time() + 30
        self._drive(voice, "and what about performance")
        time.sleep(0.3)
        assert prompts == ["and what about performance"]

    def test_expired_window_requires_the_wake_word_again(
        self, tmp_env, signal_bus
    ):
        import time

        ignored = []
        signal_bus.speech.utterance_ignored.connect(
            lambda heard, reason: ignored.append(reason)
        )
        _, voice = self._stack(tmp_env, signal_bus)
        voice._follow_up_until = time.time() - 1
        self._drive(voice, "just talking to myself")
        time.sleep(0.3)
        assert ignored

    def test_follow_up_can_be_disabled(self, tmp_env, signal_bus):
        from ai_companion.models.config_models import AppConfig

        config = AppConfig()
        config.speech.follow_up_seconds = 0.0
        assert AppConfig(**config.model_dump()).speech.follow_up_seconds == 0.0

    def test_disable_clears_the_window(self, tmp_env, signal_bus):
        import time

        _, voice = self._stack(tmp_env, signal_bus)
        voice._follow_up_until = time.time() + 30
        voice.disable()
        assert voice._follow_up_until == 0.0


class TestMicrophonePreroll:
    """The first word must not be clipped — 2026-09-19.

    The listener only began buffering once the level crossed the threshold.
    The first word of a sentence is its quietest part, so "Jarvis, how good
    is your code?" arrived as "How good is your code?" and the wake word was
    never in the transcript to match against.
    """

    def _listener(self, captured):
        from ai_companion.services.speech_engine import MicrophoneListener

        listener = MicrophoneListener(
            on_utterance=captured.append,
            silence_threshold=0.05,
            silence_seconds=0.3,
        )
        listener._running = True
        return listener

    def _push(self, listener, level, blocks=1):
        import numpy as np

        from ai_companion.services.speech_engine import STT_SAMPLE_RATE

        size = int(STT_SAMPLE_RATE * 0.1)
        for _ in range(blocks):
            data = np.full((size, 1), level, dtype="float32")
            listener._callback(data, size, None, None)

    def test_quiet_first_word_is_preserved(self):
        import time

        import numpy as np

        from ai_companion.services.speech_engine import STT_SAMPLE_RATE

        captured = []
        listener = self._listener(captured)
        self._push(listener, 0.02, 4)    # quiet wake word, below threshold
        self._push(listener, 0.20, 10)   # the rest, loud
        self._push(listener, 0.001, 4)   # trailing silence
        time.sleep(0.4)

        assert captured, "no utterance captured"
        audio = captured[0]
        lead = float(np.mean(np.abs(audio[: int(STT_SAMPLE_RATE * 0.4)])))
        assert lead > 0.005, f"quiet lead-in was clipped (level {lead:.4f})"

    def test_preroll_is_bounded(self):
        """A rolling window, not an unbounded buffer."""
        captured = []
        listener = self._listener(captured)
        self._push(listener, 0.001, 200)   # 20s of silence
        assert len(listener._preroll) <= listener._preroll_blocks

    def test_preroll_cleared_on_pause(self):
        captured = []
        listener = self._listener(captured)
        self._push(listener, 0.001, 5)
        listener.pause()
        assert listener._preroll == []

    def test_preroll_is_consumed_not_repeated(self):
        """The pre-roll must not be prepended to every later utterance."""
        import time

        captured = []
        listener = self._listener(captured)
        self._push(listener, 0.001, 6)
        self._push(listener, 0.20, 6)
        self._push(listener, 0.001, 4)
        time.sleep(0.3)
        first = len(captured[0]) if captured else 0

        self._push(listener, 0.20, 6)
        self._push(listener, 0.001, 4)
        time.sleep(0.3)
        assert len(captured) == 2
        # Second utterance had no silence before it, so it must be shorter.
        assert len(captured[1]) < first


class TestVoiceFx:
    """Voice post-processing — 2026-09-19.

    Asking for "a JARVIS voice" is mostly a request for production, not a
    different speaker. Piper's dry output slopes off steeply above 3 kHz,
    which is why it reads as a TTS engine rather than a voice in a room.
    """

    def _tone(self, seconds=1.0, rate=22050):
        import numpy as np

        t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
        # Voice-like: fundamental plus harmonics.
        signal = sum(
            np.sin(2 * np.pi * f * t) / (i + 1)
            for i, f in enumerate((140, 280, 560, 1120, 2240))
        )
        return (signal / np.max(np.abs(signal))).astype("float32"), rate

    def test_presence_is_lifted(self):
        """The dull high end is the main reason it sounds synthetic."""
        import numpy as np
        from scipy import signal as sp

        from ai_companion.services.voice_fx import apply_voice_fx, settings_for

        audio, rate = self._tone()
        out = apply_voice_fx(audio, rate, settings_for("jarvis"))

        def presence(data):
            freqs, power = sp.welch(data, rate, nperseg=1024)
            mask = (freqs >= 3000) & (freqs < 6000)
            return float(np.mean(power[mask]))

        assert presence(out) > presence(audio)

    def test_output_never_clips(self):
        import numpy as np

        from ai_companion.services.voice_fx import apply_voice_fx, settings_for

        audio, rate = self._tone()
        for preset in ("jarvis", "hall", "intercom", "clean"):
            out = apply_voice_fx(audio, rate, settings_for(preset))
            assert float(np.max(np.abs(out))) <= 1.0, preset

    def test_output_is_finite(self):
        """A NaN would come out as a burst of noise."""
        import numpy as np

        from ai_companion.services.voice_fx import apply_voice_fx, settings_for

        audio, rate = self._tone()
        for preset in ("jarvis", "hall", "intercom", "clean"):
            out = apply_voice_fx(audio, rate, settings_for(preset))
            assert np.all(np.isfinite(out)), preset

    def test_off_preset_is_a_passthrough(self):
        import numpy as np

        from ai_companion.services.voice_fx import apply_voice_fx, settings_for

        audio, rate = self._tone()
        out = apply_voice_fx(audio, rate, settings_for("off"))
        assert np.array_equal(out, audio)

    def test_length_is_preserved(self):
        """A length change would desynchronise playback timing."""
        from ai_companion.services.voice_fx import apply_voice_fx, settings_for

        audio, rate = self._tone()
        for preset in ("jarvis", "hall", "intercom", "clean"):
            assert len(apply_voice_fx(audio, rate, settings_for(preset))) == \
                len(audio)

    def test_empty_input_is_safe(self):
        import numpy as np

        from ai_companion.services.voice_fx import apply_voice_fx

        empty = np.array([], dtype="float32")
        assert len(apply_voice_fx(empty, 22050)) == 0

    def test_unknown_preset_falls_back(self):
        from ai_companion.services.voice_fx import settings_for

        assert settings_for("nonsense").reverb_amount > 0

    def test_failure_returns_the_dry_signal(self, monkeypatch):
        """A broken effect must never silence the assistant."""
        import numpy as np

        from ai_companion.services import voice_fx

        def boom(*args, **kwargs):
            raise RuntimeError("dsp exploded")

        monkeypatch.setattr(voice_fx, "_high_pass", boom)
        audio, rate = self._tone()
        out = voice_fx.apply_voice_fx(audio, rate)
        assert np.array_equal(out, audio)

    def test_presets_are_all_described(self):
        from ai_companion.services.voice_fx import PRESETS, describe_presets

        described = {name for name, _ in describe_presets()}
        assert described == set(PRESETS)

    def test_config_stores_the_choice(self):
        from ai_companion.models.config_models import AppConfig

        config = AppConfig()
        # Default changed jarvis -> natural: the jarvis preset includes a
        # detune that makes the voice sound synthetic, which was the opposite
        # of what was wanted. It is still available, just not the default.
        assert config.speech.voice_fx == "natural"
        config.speech.voice_fx = "hall"
        assert AppConfig(**config.model_dump()).speech.voice_fx == "hall"

    def test_processing_is_fast_enough(self):
        """Must be negligible against Piper's own synthesis time."""
        import time

        from ai_companion.services.voice_fx import apply_voice_fx, settings_for

        audio, rate = self._tone(seconds=6.0)
        start = time.time()
        apply_voice_fx(audio, rate, settings_for("jarvis"))
        elapsed = time.time() - start
        assert elapsed < 1.0, f"took {elapsed:.2f}s for 6s of audio"


class TestTtsEngineChoice:
    """Piper vs Kokoro — 2026-09-20.

    Measured on this hardware:
        piper  ~9.8x realtime, 0.56s for a 5.5s reply
        kokoro ~1.2x realtime, 4.35s for a 5.1s reply

    Neither is objectively better. Piper is fast and slightly flatter; Kokoro
    is slower and generally judged more natural. The user picks.
    """

    def test_default_engine_is_piper(self):
        """The fast one stays default; slow-by-surprise is a bad default."""
        from ai_companion.models.config_models import AppConfig

        assert AppConfig().speech.tts_engine_name == "piper"

    def test_engine_choice_persists(self):
        from ai_companion.models.config_models import AppConfig

        config = AppConfig()
        config.speech.tts_engine_name = "kokoro"
        assert AppConfig(**config.model_dump()).speech.tts_engine_name == \
            "kokoro"

    def test_kokoro_missing_models_fails_cleanly(self, tmp_path):
        from ai_companion.services.speech_engine import KokoroSynthesiser

        ok, error = KokoroSynthesiser(str(tmp_path / "nope")).load()
        assert ok is False
        assert "not found" in error.lower()

    def test_kokoro_synthesis_reports_errors(self, tmp_path):
        from ai_companion.services.speech_engine import KokoroSynthesiser

        result = KokoroSynthesiser(str(tmp_path / "nope")).synthesise("hello")
        assert result.ok is False
        assert result.error

    def test_kokoro_rejects_empty_text(self, tmp_path):
        from ai_companion.services.speech_engine import KokoroSynthesiser

        assert KokoroSynthesiser(str(tmp_path)).synthesise("").ok is False

    def test_voice_list_is_populated(self):
        from ai_companion.services.speech_engine import KOKORO_VOICES

        assert len(KOKORO_VOICES) >= 4
        assert all(len(entry) == 2 for entry in KOKORO_VOICES)


class TestDetuneIsOffByDefault:
    """The robotic effect must not be on unless asked for — 2026-09-20."""

    def test_default_preset_has_no_detune(self):
        """A detuned double is literally a robot effect."""
        from ai_companion.services.voice_fx import settings_for

        assert settings_for("natural").chorus_amount == 0.0

    def test_most_presets_have_no_detune(self):
        from ai_companion.services.voice_fx import settings_for

        for preset in ("natural", "clean", "intercom", "hall"):
            assert settings_for(preset).chorus_amount == 0.0, preset

    def test_jarvis_preset_keeps_it_deliberately(self):
        """Still available for anyone who wants the film sound."""
        from ai_companion.services.voice_fx import settings_for

        assert settings_for("jarvis").chorus_amount > 0

    def test_natural_is_the_fallback_for_unknown(self):
        from ai_companion.services.voice_fx import settings_for

        assert settings_for("nonsense").chorus_amount == 0.0


class TestVaultPrivacyAudit:
    """The vault was the last store not using the shared rule — 2026-09-20.

    Chat, memory, graph and code snapshots all refuse to write in private
    mode. VaultService did not: removing a file during a private session
    still rewrote vault_index.json on disk.
    """

    def _session(self, tmp_env, signal_bus, mode):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import VaultMode
        from ai_companion.services.vault_service import VaultService

        config, root = tmp_env
        config.vault.mode = VaultMode(mode)
        config.vault.approved_folders = [str(root)]
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        vault = VaultService(config, signal_bus)
        sm.register(vault)
        vault.start()
        return vault, config, root

    def _index(self, config):
        import json
        from pathlib import Path

        path = Path(config.vault.vault_root) / "vault_index.json"
        if not path.exists():
            return []
        return list(json.loads(path.read_text(encoding="utf-8")).get(
            "items", {}
        ))

    def test_vault_uses_the_shared_privacy_rule(self):
        """One implementation, not a sixth divergent copy."""
        from ai_companion.infrastructure.privacy import PrivacyMixin
        from ai_companion.services.vault_service import VaultService

        assert issubclass(VaultService, PrivacyMixin)
        assert "is_private_mode" not in VaultService.__dict__

    def test_import_refused_in_private_mode(self, tmp_env, signal_bus):
        vault, config, root = self._session(tmp_env, signal_bus, "private")
        source = root / "secret.txt"
        source.write_text("CONFIDENTIAL", encoding="utf-8")
        assert vault.import_file(str(source), "docs") is None
        vault.stop()

    def test_nothing_from_a_private_import_reaches_disk(
        self, tmp_env, signal_bus
    ):
        import os

        vault, config, root = self._session(tmp_env, signal_bus, "private")
        source = root / "SECRET_DOC.txt"
        source.write_text("CONFIDENTIAL CONTENTS", encoding="utf-8")
        vault.import_file(str(source), "docs")
        vault.stop()

        blob = ""
        for folder, _dirs, files in os.walk(config.vault.vault_root):
            for name in files:
                blob += name
                try:
                    blob += open(
                        os.path.join(folder, name), encoding="utf-8",
                        errors="ignore",
                    ).read()
                except OSError:
                    pass
        assert "SECRET_DOC" not in blob
        assert "CONFIDENTIAL" not in blob

    def test_index_not_rewritten_on_private_removal(
        self, tmp_env, signal_bus
    ):
        """The actual hole: remove_file() persisted while private."""
        vault, config, root = self._session(tmp_env, signal_bus, "normal")
        source = root / "doc.txt"
        source.write_text("data", encoding="utf-8")
        relative = vault.import_file(str(source), "docs")
        vault.stop()
        before = self._index(config)
        assert before

        vault2, config, _ = self._session(tmp_env, signal_bus, "private")
        vault2.remove_file(relative)
        vault2.stop()

        assert self._index(config) == before

    def test_the_file_is_still_deleted_in_private_mode(
        self, tmp_env, signal_bus
    ):
        """Refusing deletion would trap data in the vault."""
        from pathlib import Path

        vault, config, root = self._session(tmp_env, signal_bus, "normal")
        source = root / "doc.txt"
        source.write_text("data", encoding="utf-8")
        relative = vault.import_file(str(source), "docs")
        vault.stop()

        vault2, config, _ = self._session(tmp_env, signal_bus, "private")
        vault2.remove_file(relative)
        vault2.stop()

        assert not (Path(config.vault.vault_root) / relative).exists()

    def test_index_reconciles_on_next_start(self, tmp_env, signal_bus):
        """A private deletion must not leave a ghost entry behind."""
        vault, config, root = self._session(tmp_env, signal_bus, "normal")
        source = root / "doc.txt"
        source.write_text("data", encoding="utf-8")
        relative = vault.import_file(str(source), "docs")
        vault.stop()

        vault2, config, _ = self._session(tmp_env, signal_bus, "private")
        vault2.remove_file(relative)
        vault2.stop()

        vault3, config, _ = self._session(tmp_env, signal_bus, "normal")
        assert vault3.list_files() == []
        vault3.stop()

    def test_normal_mode_still_persists(self, tmp_env, signal_bus):
        """Guard the fix: never writing at all would be a worse bug."""
        vault, config, root = self._session(tmp_env, signal_bus, "normal")
        source = root / "doc.txt"
        source.write_text("data", encoding="utf-8")
        vault.import_file(str(source), "docs")
        vault.stop()
        assert self._index(config)

    def test_audit_log_stays_in_memory(self, tmp_env, signal_bus):
        """It records file paths, so it must never be written to disk."""
        import os

        vault, config, root = self._session(tmp_env, signal_bus, "private")
        source = root / "TRACEABLE_NAME.txt"
        source.write_text("x", encoding="utf-8")
        vault.import_file(str(source), "docs")
        assert any(
            "TRACEABLE_NAME" in entry["detail"]
            for entry in vault.get_audit_log()
        )
        vault.stop()

        blob = ""
        for folder, _dirs, files in os.walk(config.data_dir):
            for name in files:
                try:
                    blob += open(
                        os.path.join(folder, name), encoding="utf-8",
                        errors="ignore",
                    ).read()
                except OSError:
                    pass
        assert "TRACEABLE_NAME" not in blob

    def test_all_five_stores_share_one_rule(self):
        """The audit's actual conclusion, as a regression guard."""
        from ai_companion.infrastructure.privacy import PrivacyMixin
        from ai_companion.services.code_workspace import CodeWorkspaceService
        from ai_companion.services.graph_service import GraphService
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.memory_service import MemoryService
        from ai_companion.services.vault_service import VaultService

        for service in (
            LlmService, MemoryService, GraphService,
            CodeWorkspaceService, VaultService,
        ):
            assert issubclass(service, PrivacyMixin), service.__name__
            assert "is_private_mode" not in service.__dict__, service.__name__
