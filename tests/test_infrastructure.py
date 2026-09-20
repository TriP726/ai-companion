"""Tests for core infrastructure components."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from ai_companion.models.config_models import AppConfig, LlmConfig, VaultMode


class TestConfigModels:
    """Test Pydantic configuration models."""

    def test_default_config(self):
        config = AppConfig()
        assert config.version == "1"
        assert config.llm.context_length == 4096
        assert config.llm.host == "127.0.0.1"
        # Default is NORMAL so a fresh install actually saves conversations.
        # Private is a deliberate one-click choice, not a silent default.
        assert config.vault.mode == VaultMode.NORMAL
        assert config.camera.auto_record is False
        assert config.code_workspace.allow_network is False

    def test_config_validation(self):
        # Temperature bounds
        with pytest.raises(Exception):
            LlmConfig(temperature=-0.1)
        with pytest.raises(Exception):
            LlmConfig(temperature=2.1)

        # Context length bounds
        with pytest.raises(Exception):
            LlmConfig(context_length=100)  # below min 512

    def test_config_serialization(self):
        config = AppConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["version"] == "1"
        assert data["llm"]["host"] == "127.0.0.1"

    def test_config_from_dict(self):
        data = {
            "version": "1",
            "llm": {"model_path": "/test/model.gguf", "context_length": 8192},
            "vault": {"mode": "normal"},
        }
        config = AppConfig(**data)
        assert config.llm.model_path == "/test/model.gguf"
        assert config.llm.context_length == 8192
        assert config.vault.mode == VaultMode.NORMAL

    def test_loopback_constraint_documented(self):
        """Verify that host fields default to loopback."""
        config = AppConfig()
        assert config.llm.host == "127.0.0.1"
        assert config.image_gen.host == "127.0.0.1"


class TestJsonStore:
    """Test the JSON persistence layer."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmp_dir, "test_store.json")

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_and_load(self):
        from ai_companion.infrastructure.json_store import JsonStore

        store = JsonStore(self.store_path)
        data = store.load()
        assert "_meta" in data
        assert "items" in data
        assert data["_meta"]["version"] == "1"

    def test_set_get_delete(self):
        from ai_companion.infrastructure.json_store import JsonStore

        store = JsonStore(self.store_path)
        store.load()

        store.set("key1", {"value": "hello"})
        assert store.get("key1") == {"value": "hello"}
        assert store.count() == 1

        store.set("key2", {"value": "world"})
        assert store.count() == 2

        assert store.delete("key1") is True
        assert store.get("key1") is None
        assert store.count() == 1

        assert store.delete("nonexistent") is False

    def test_persistence(self):
        from ai_companion.infrastructure.json_store import JsonStore

        store = JsonStore(self.store_path)
        store.load()
        store.set("persist", {"data": 42})
        store.save()

        # Load in new instance
        store2 = JsonStore(self.store_path)
        store2.load()
        assert store2.get("persist") == {"data": 42}

    def test_keys_values_items(self):
        from ai_companion.infrastructure.json_store import JsonStore

        store = JsonStore(self.store_path)
        store.load()
        store.set("a", 1)
        store.set("b", 2)
        store.set("c", 3)

        assert set(store.keys()) == {"a", "b", "c"}
        assert set(store.values()) == {1, 2, 3}
        assert len(store.items()) == 3

    def test_clear(self):
        from ai_companion.infrastructure.json_store import JsonStore

        store = JsonStore(self.store_path)
        store.load()
        store.set("x", 1)
        store.set("y", 2)
        assert store.count() == 2

        store.clear()
        assert store.count() == 0
        meta = store.get_meta()
        assert "version" in meta
        assert meta["version"] == "1"

    def test_backup(self):
        from ai_companion.infrastructure.json_store import JsonStore

        store = JsonStore(self.store_path)
        store.load()
        store.set("backup_test", True)
        store.save()

        backup_path = store.create_backup()
        assert backup_path is not None
        assert backup_path.exists()

    def test_atomic_write(self):
        """Verify no .tmp file left after save."""
        from ai_companion.infrastructure.json_store import JsonStore

        store = JsonStore(self.store_path)
        store.load()
        store.set("atomic", True)
        store.save()

        tmp_path = Path(self.store_path).with_suffix(".tmp")
        assert not tmp_path.exists()
        assert Path(self.store_path).exists()


class TestPathValidator:
    """Test path validation and security constraints."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.approved = [self.tmp_dir]

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_path(self):
        from ai_companion.infrastructure.path_validator import PathValidator

        pv = PathValidator(approved_roots=self.approved)
        test_path = os.path.join(self.tmp_dir, "test.txt")
        Path(test_path).touch()

        result = pv.validate_path(test_path, must_exist=True)
        assert result.exists()

    def test_blocks_traversal(self):
        from ai_companion.infrastructure.path_validator import (
            PathValidator,
            PathValidationError,
        )

        pv = PathValidator(approved_roots=self.approved)
        with pytest.raises(PathValidationError):
            pv.validate_path(os.path.join(self.tmp_dir, "../../../etc/passwd"))

    def test_blocks_outside_roots(self):
        from ai_companion.infrastructure.path_validator import (
            PathValidator,
            PathValidationError,
        )

        pv = PathValidator(approved_roots=self.approved)
        with pytest.raises(PathValidationError):
            pv.validate_path("/etc/passwd")

    def test_blocks_null_bytes(self):
        from ai_companion.infrastructure.path_validator import (
            PathValidator,
            PathValidationError,
        )

        pv = PathValidator(approved_roots=self.approved)
        with pytest.raises(PathValidationError, match="null bytes"):
            pv.validate_path("test\x00.txt")

    def test_extension_filter(self):
        from ai_companion.infrastructure.path_validator import (
            PathValidator,
            PathValidationError,
        )

        pv = PathValidator(
            approved_roots=self.approved,
            allowed_extensions=[".txt", ".md"],
        )
        # Valid extension
        result = pv.validate_path(
            os.path.join(self.tmp_dir, "readme.md")
        )
        assert result.suffix == ".md"

        # Invalid extension
        with pytest.raises(PathValidationError, match="Extension"):
            pv.validate_path(os.path.join(self.tmp_dir, "evil.exe"))

    def test_sanitize_filename(self):
        from ai_companion.infrastructure.path_validator import (
            PathValidator,
            PathValidationError,
        )

        pv = PathValidator(approved_roots=self.approved)

        # Normal filename
        assert pv.sanitize_filename("test.txt") == "test.txt"

        # Path traversal in filename
        assert pv.sanitize_filename("../../../etc/passwd") == "passwd"

        # Hidden file blocked
        with pytest.raises(PathValidationError):
            pv.sanitize_filename(".hidden")

    def test_file_size_check(self):
        from ai_companion.infrastructure.path_validator import (
            PathValidator,
            PathValidationError,
        )

        pv = PathValidator(
            approved_roots=self.approved,
            max_file_size_bytes=100,
        )
        small_file = Path(self.tmp_dir) / "small.txt"
        small_file.write_text("hello")

        # Should pass
        pv.validate_file_size(small_file)

        # Create a file that exceeds limit
        large_file = Path(self.tmp_dir) / "large.txt"
        large_file.write_text("x" * 200)

        with pytest.raises(PathValidationError, match="exceeds limit"):
            pv.validate_file_size(large_file)

    def test_empty_path_rejected(self):
        from ai_companion.infrastructure.path_validator import (
            PathValidator,
            PathValidationError,
        )

        pv = PathValidator(approved_roots=self.approved)
        with pytest.raises(PathValidationError, match="Empty"):
            pv.validate_path("")
        with pytest.raises(PathValidationError, match="Empty"):
            pv.validate_path("   ")

    def test_is_within_roots(self):
        from ai_companion.infrastructure.path_validator import PathValidator

        pv = PathValidator(approved_roots=self.approved)
        assert pv.is_within_roots(os.path.join(self.tmp_dir, "test.txt"))
        assert not pv.is_within_roots("/etc/passwd")


class TestConversationModels:
    """Test conversation and message models."""

    def test_create_conversation(self):
        from ai_companion.models.conversation import Conversation, MessageRole

        conv = Conversation()
        assert conv.title == "New Conversation"
        assert len(conv.messages) == 0

    def test_add_message(self):
        from ai_companion.models.conversation import Conversation, MessageRole

        conv = Conversation()
        msg = conv.add_message(MessageRole.USER, "Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert len(conv.messages) == 1

    def test_get_history(self):
        from ai_companion.models.conversation import Conversation, MessageRole

        conv = Conversation()
        conv.add_message(MessageRole.USER, "Hi")
        conv.add_message(MessageRole.ASSISTANT, "Hello!")
        conv.add_message(MessageRole.USER, "How are you?")

        history = conv.get_history()
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["content"] == "How are you?"

    def test_get_history_limit(self):
        from ai_companion.models.conversation import Conversation, MessageRole

        conv = Conversation()
        for i in range(100):
            conv.add_message(MessageRole.USER, f"msg {i}")

        history = conv.get_history(max_messages=10)
        assert len(history) == 10


class TestMemoryModels:
    """Test memory data models."""

    def test_create_memory(self):
        from ai_companion.models.memory import Memory, MemoryStatus, MemorySource

        mem = Memory(content="Test memory")
        assert mem.content == "Test memory"
        assert mem.status == MemoryStatus.PENDING
        assert mem.source == MemorySource.USER_EXPLICIT
        assert 0.0 <= mem.confidence <= 1.0

    def test_memory_filter(self):
        from ai_companion.models.memory import MemoryFilter, MemoryStatus

        f = MemoryFilter(query="test", status=MemoryStatus.APPROVED)
        assert f.query == "test"
        assert f.status == MemoryStatus.APPROVED
        assert f.min_confidence == 0.0


class TestGraphModels:
    """Test graph data models."""

    def test_create_node(self):
        from ai_companion.models.graph import GraphNode, NodeType

        node = GraphNode(node_type=NodeType.MEMORY, label="Test")
        assert node.node_type == NodeType.MEMORY
        assert node.label == "Test"
        assert node.id  # auto-generated

    def test_create_edge(self):
        from ai_companion.models.graph import GraphEdge, EdgeType

        edge = GraphEdge(source_id="a", target_id="b")
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.edge_type == EdgeType.RELATED_TO


class TestSystemPromptConfig:
    """LlmConfig.system_prompt — added 2026-09-17."""

    def test_default_system_prompt_present(self):
        from ai_companion.models.config_models import LlmConfig
        cfg = LlmConfig()
        assert cfg.system_prompt
        assert "concisely" in cfg.system_prompt

    def test_system_prompt_is_overridable(self):
        from ai_companion.models.config_models import LlmConfig
        cfg = LlmConfig(system_prompt="Be terse.")
        assert cfg.system_prompt == "Be terse."

    def test_system_prompt_length_capped(self):
        import pytest
        from pydantic import ValidationError
        from ai_companion.models.config_models import LlmConfig
        with pytest.raises(ValidationError):
            LlmConfig(system_prompt="x" * 4001)

    def test_old_config_without_system_prompt_still_loads(self):
        """Backward compatibility: configs saved before this field existed."""
        from ai_companion.models.config_models import LlmConfig
        cfg = LlmConfig.model_validate({"model_path": "m.gguf", "threads": 6})
        assert cfg.threads == 6
        assert cfg.system_prompt


class TestPathRelocation:
    """AppConfig.relocate() and its self-enforcing coverage — 2026-09-17."""

    def test_relocate_rewrites_every_registered_path(self, tmp_path):
        from ai_companion.models.config_models import AppConfig

        cfg = AppConfig().relocate(tmp_path)
        for section, field, _ in AppConfig.PATH_FIELDS:
            obj = cfg if section is None else getattr(cfg, section)
            value = getattr(obj, field)
            assert str(tmp_path) in value, f"{section}.{field} not relocated"

    def test_relocate_leaves_user_chosen_paths_alone(self, tmp_path):
        from ai_companion.models.config_models import AppConfig

        cfg = AppConfig()
        cfg.llm.model_path = r"C:\models\phi3.gguf"
        cfg.image_gen.comfyui_path = r"C:\tools\comfy"
        cfg.vault.approved_folders = [r"C:\Users\me\Documents"]
        cfg.relocate(tmp_path)
        assert cfg.llm.model_path == r"C:\models\phi3.gguf"
        assert cfg.image_gen.comfyui_path == r"C:\tools\comfy"
        assert cfg.vault.approved_folders == [r"C:\Users\me\Documents"]

    def test_relocate_is_idempotent(self, tmp_path):
        from ai_companion.models.config_models import AppConfig

        cfg = AppConfig().relocate(tmp_path).relocate(tmp_path)
        assert cfg.memory.store_path == str(tmp_path / "data" / "memories.json")

    def test_relocate_returns_self_for_chaining(self, tmp_path):
        from ai_companion.models.config_models import AppConfig

        cfg = AppConfig()
        assert cfg.relocate(tmp_path) is cfg

    def test_every_path_field_is_registered_or_exempt(self):
        """Guard rail: adding a new path-like config field without registering
        it in PATH_FIELDS (or PATH_FIELDS_EXEMPT) fails HERE, loudly, instead of
        silently letting tests read and overwrite real user data.

        This is the check that would have caught the conversations_path leak.
        """
        from pydantic import BaseModel

        from ai_companion.models.config_models import AppConfig

        registered = {
            f"{section}.{field}" if section else field
            for section, field, _ in AppConfig.PATH_FIELDS
        }
        exempt = set(AppConfig.PATH_FIELDS_EXEMPT)

        suspicious_suffixes = ("_path", "_dir", "_root", "_folders")
        missing: list[str] = []

        cfg = AppConfig()
        for section_name in AppConfig.model_fields:
            value = getattr(cfg, section_name)
            if not isinstance(value, BaseModel):
                # scalar on AppConfig itself
                if section_name.endswith(suspicious_suffixes):
                    if section_name not in registered | exempt:
                        missing.append(section_name)
                continue
            for sub_name in type(value).model_fields:
                if not sub_name.endswith(suspicious_suffixes):
                    continue
                key = f"{section_name}.{sub_name}"
                if key not in registered and key not in exempt:
                    missing.append(key)

        assert not missing, (
            "Unregistered path-like config field(s): "
            + ", ".join(sorted(missing))
            + ". Add each to AppConfig.PATH_FIELDS so tests relocate it, or to "
            "PATH_FIELDS_EXEMPT with a reason if it points outside the app."
        )


class TestThemeTokens:
    """Token-based theming — added 2026-09-17."""

    def test_both_modes_produce_stylesheets(self):
        """Light mode was deliberately dropped for the HUD redesign.

        LIGHT is now an alias of DARK so existing references keep working.
        Both modes must still render a usable stylesheet; they are no longer
        required to differ.
        """
        from ai_companion.ui.theme import Theme, ThemeMode

        dark = Theme(ThemeMode.DARK).stylesheet()
        light = Theme(ThemeMode.LIGHT).stylesheet()
        assert dark and light

    def test_no_unresolved_format_placeholders(self):
        """A stray {token} means a colour silently renders as literal text."""
        import re

        from ai_companion.ui.theme import Theme, ThemeMode

        for mode in ThemeMode:
            css = Theme(mode).stylesheet()
            leftovers = re.findall(r"\{[a-z_]+\}", css)
            assert not leftovers, f"{mode}: {leftovers}"

    def test_toggle_round_trips(self):
        from ai_companion.ui.theme import Theme, ThemeMode

        t = Theme(ThemeMode.DARK)
        assert t.toggle() is ThemeMode.LIGHT
        assert t.toggle() is ThemeMode.DARK

    def test_palettes_define_identical_token_sets(self):
        """Light must not be missing a token that dark defines."""
        import dataclasses

        from ai_companion.ui.theme import DARK, LIGHT

        assert {f.name for f in dataclasses.fields(DARK)} == {
            f.name for f in dataclasses.fields(LIGHT)
        }

    def test_every_colour_token_is_non_empty(self):
        import dataclasses

        from ai_companion.ui.theme import DARK, LIGHT

        for palette in (DARK, LIGHT):
            for f in dataclasses.fields(palette):
                assert getattr(palette, f.name), f.name

    def test_ui_modules_contain_no_hardcoded_hex(self):
        """Regression guard: a hardcoded colour breaks the light theme.

        Every colour must come from the token palette, otherwise switching
        themes leaves dark-mode colours stranded in light mode.
        """
        import pathlib
        import re

        ui_dir = pathlib.Path("ai_companion/ui")
        hex_pattern = re.compile(r"#[0-9a-fA-F]{6}\b")
        offenders: list[str] = []
        for path in ui_dir.glob("*.py"):
            if path.name == "theme.py":
                continue  # the palette definition is the one allowed place
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if hex_pattern.search(line):
                    offenders.append(f"{path.name}:{lineno}")
        assert not offenders, f"hardcoded colours: {offenders}"


class TestTranscriptRendering:
    """Chat transcript HTML — testable without a display server."""

    def _pm(self):
        from ai_companion.ui.theme import DARK, METRICS

        return DARK, METRICS

    def test_splits_prose_and_code(self):
        from ai_companion.ui.transcript import split_segments

        segs = split_segments("intro\n```python\nx = 1\n```\noutro")
        assert [s.kind for s in segs] == ["text", "code", "text"]
        assert segs[1].language == "python"
        assert segs[1].text == "x = 1"

    def test_unterminated_fence_still_renders_as_code(self):
        """Mid-stream the closing fence has not arrived yet."""
        from ai_companion.ui.transcript import split_segments

        segs = split_segments("here:\n```py\nhalf_a_line")
        assert segs[-1].kind == "code"

    def test_indentation_preserved(self):
        from ai_companion.ui.transcript import render_body

        p, m = self._pm()
        html = render_body("```python\ndef f():\n    return 1\n```", p, m)
        assert "&nbsp;&nbsp;&nbsp;&nbsp;return 1" in html

    def test_angle_brackets_escaped(self):
        """Unescaped output is parsed as markup and silently swallowed."""
        from ai_companion.ui.transcript import render_body

        p, m = self._pm()
        html = render_body("if a < b and c > d", p, m)
        assert "&lt;" in html and "&gt;" in html

    def test_ampersand_escaped(self):
        from ai_companion.ui.transcript import render_body

        p, m = self._pm()
        assert "&amp;" in render_body("Tom & Jerry", p, m)

    def test_html_injection_is_inert(self):
        from ai_companion.ui.transcript import render_body

        p, m = self._pm()
        html = render_body("<script>alert(1)</script>", p, m)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_roles_get_distinct_labels(self):
        from ai_companion.ui.transcript import render_message

        p, m = self._pm()
        assert "YOU" in render_message("user", "hi", p, m)
        assert "ASSISTANT" in render_message("assistant", "hi", p, m)

    def test_system_messages_render_compactly(self):
        from ai_companion.ui.transcript import render_message

        p, m = self._pm()
        html = render_message("system", "New conversation started.", p, m)
        assert "<table" not in html

    def test_empty_text_does_not_crash(self):
        from ai_companion.ui.transcript import render_body, render_message

        p, m = self._pm()
        assert render_body("", p, m) == ""
        assert render_message("user", "", p, m)

    def test_inline_code_and_bold(self):
        from ai_companion.ui.transcript import render_body

        p, m = self._pm()
        html = render_body("use `pip` and **care**", p, m)
        assert "monospace" in html
        assert "<b>care</b>" in html


class TestSpellCheck:
    """Composer spell checking — 2026-09-18."""

    def _sp(self):
        from ai_companion.ui.spellcheck import SpellChecker_

        return SpellChecker_()

    def test_catches_common_typos(self):
        sp = self._sp()
        if not sp.available:
            import pytest

            pytest.skip("pyspellchecker not installed")
        for word in ("disord", "recieve", "mispelled", "teh"):
            assert sp.is_misspelled(word), word

    def test_suggests_the_right_correction(self):
        sp = self._sp()
        if not sp.available:
            import pytest

            pytest.skip("pyspellchecker not installed")
        assert "discord" in sp.suggestions("disord")
        assert "receive" in sp.suggestions("recieve")

    def test_technical_words_not_flagged(self):
        """A dictionary that underlines 'json' becomes noise and gets ignored."""
        sp = self._sp()
        if not sp.available:
            import pytest

            pytest.skip("pyspellchecker not installed")
        for word in (
            "json", "async", "python", "gguf", "pyside", "def", "bot",
            "discord", "llm", "repo", "config", "kwargs",
        ):
            assert not sp.is_misspelled(word), word

    def test_code_shapes_are_skipped(self):
        sp = self._sp()
        if not sp.available:
            import pytest

            pytest.skip("pyspellchecker not installed")
        for token in ("myVar", "snake_case", "C:/models", "file.txt", "x1"):
            assert not sp.is_misspelled(token), token

    def test_short_words_ignored(self):
        sp = self._sp()
        assert not sp.is_misspelled("ok")
        assert not sp.is_misspelled("a")

    def test_custom_word_persists(self, tmp_path):
        from ai_companion.ui.spellcheck import SpellChecker_

        path = tmp_path / "custom.json"
        sp = SpellChecker_(custom_path=path)
        if not sp.available:
            import pytest

            pytest.skip("pyspellchecker not installed")
        assert sp.is_misspelled("zzyzx")
        sp.add_word("zzyzx")
        assert not sp.is_misspelled("zzyzx")
        # a new instance must still know it
        assert not SpellChecker_(custom_path=path).is_misspelled("zzyzx")

    def test_capitalisation_mirrored_in_suggestions(self):
        sp = self._sp()
        if not sp.available:
            import pytest

            pytest.skip("pyspellchecker not installed")
        assert all(s[0].isupper() for s in sp.suggestions("Recieve"))

    def test_word_at_cursor_finds_the_word(self):
        from ai_companion.ui.spellcheck import word_at_cursor

        text = "show me a disord bot"
        word, start, end = word_at_cursor(text, 13)
        assert word == "disord"
        assert text[start:end] == "disord"

    def test_word_at_cursor_handles_edges(self):
        from ai_companion.ui.spellcheck import word_at_cursor

        assert word_at_cursor("", 0)[0] == ""
        assert word_at_cursor("hi", 0)[0] == "hi"
        assert word_at_cursor("hi", 2)[0] == "hi"

    def test_missing_library_degrades_quietly(self, monkeypatch):
        """No spellchecker must mean no squiggle, never a crash."""
        from ai_companion.ui import spellcheck

        monkeypatch.setattr(spellcheck, "_AVAILABLE", False)
        monkeypatch.setattr(spellcheck, "SpellChecker", None)
        sp = spellcheck.SpellChecker_()
        assert sp.available is False
        assert sp.is_misspelled("disord") is False
        assert sp.suggestions("disord") == []


class TestConfigPathIsCwdIndependent:
    """Settings must survive being launched from anywhere — 2026-09-18."""

    def test_default_path_is_absolute(self):
        """A relative 'config.json' follows the working directory around."""
        import os

        from ai_companion.config import DEFAULT_CONFIG_PATH

        assert os.path.isabs(DEFAULT_CONFIG_PATH)

    def test_default_path_sits_at_project_root(self):
        """Verify the real default, not the test sandbox.

        conftest redirects DEFAULT_CONFIG_PATH for the whole session so no
        test can clobber the developer's config. This one is specifically
        about the shipped default, so it recomputes it from source.
        """
        from pathlib import Path

        import ai_companion
        from ai_companion.config import _default_config_path

        root = Path(ai_companion.__file__).resolve().parent.parent
        assert _default_config_path() == root / "config.json"

    def test_same_path_regardless_of_cwd(self, tmp_path, monkeypatch):
        """This is the actual bug: two cwds gave two different config files."""
        from ai_companion.config import ConfigManager

        first = ConfigManager()._path
        monkeypatch.chdir(tmp_path)
        second = ConfigManager()._path
        assert first == second

    def test_settings_round_trip_from_a_different_cwd(
        self, tmp_path, monkeypatch
    ):
        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager
        from ai_companion.models.config_models import AppConfig, VaultMode

        target = tmp_path / "config.json"
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )

        cm = ConfigManager()
        cm._config = AppConfig()
        cm._config.llm.model_path = "C:/models/Qwen.gguf"
        cm._config.vault.mode = VaultMode.PRIVATE
        cm.save()

        # "restart" the app from an unrelated directory
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        loaded = ConfigManager().load()
        assert loaded.llm.model_path == "C:/models/Qwen.gguf"
        assert loaded.vault.mode == VaultMode.PRIVATE

    def test_save_is_atomic(self, tmp_path, monkeypatch):
        """A truncated config.json would look like every setting resetting."""
        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager
        from ai_companion.models.config_models import AppConfig

        target = tmp_path / "config.json"
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )
        cm = ConfigManager()
        cm._config = AppConfig()
        cm.save()
        assert target.exists()
        # the temp file must not be left behind
        assert not (tmp_path / "config.json.tmp").exists()

    def test_saved_config_is_valid_json_and_reloads(self, tmp_path,
                                                    monkeypatch):
        import json

        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager
        from ai_companion.models.config_models import AppConfig

        target = tmp_path / "config.json"
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )
        cm = ConfigManager()
        cm._config = AppConfig()
        cm._config.llm.max_tokens = 1536
        cm.save()

        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["llm"]["max_tokens"] == 1536
        assert ConfigManager().load().llm.max_tokens == 1536

    def test_explicit_relative_path_is_resolved(self, tmp_path, monkeypatch):
        from ai_companion.config import ConfigManager

        monkeypatch.chdir(tmp_path)
        cm = ConfigManager("custom.json")
        assert cm._path.is_absolute()


class TestHudWidgets:
    """HUD chrome and its performance contract — 2026-09-18."""

    # Widgets that own QTimers must outlive the test. If Python frees one
    # while a timer is still armed, the timer fires into freed memory and the
    # interpreter dies with a bus error the next time any test pumps events.
    _KEEPALIVE: list = []

    def _keep(self, widget):
        type(self)._KEEPALIVE.append(widget)
        return widget

    def _app(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    # ---- the performance contract ----

    def test_reactor_idle_runs_no_timer(self):
        """Idle animation would steal CPU from token generation."""
        self._app()
        from ai_companion.ui.hud import ReactorCore

        core = self._keep(ReactorCore())
        assert core._timer.isActive() is False

    def test_reactor_only_animates_while_active(self):
        self._app()
        from ai_companion.ui.hud import ReactorCore

        core = self._keep(ReactorCore())
        core.show()
        core.set_active(True)
        assert core._timer.isActive() is True
        core.set_active(False)
        assert core._timer.isActive() is False

    def test_reactor_stops_when_hidden(self):
        """A timer running behind a hidden widget is pure waste."""
        self._app()
        from ai_companion.ui.hud import ReactorCore

        core = self._keep(ReactorCore())
        core.show()
        core.set_active(True)
        core.hide()
        assert core._timer.isActive() is False

    def test_pulse_bar_idle_runs_no_timer(self):
        self._app()
        from ai_companion.ui.hud import PulseBar

        bar = self._keep(PulseBar())
        assert bar._timer.isActive() is False
        bar.show()
        bar.set_active(True)
        assert bar._timer.isActive() is True
        bar.hide()
        assert bar._timer.isActive() is False

    def test_set_active_is_idempotent(self):
        """Repeated signals must not stack timers or restart the animation."""
        self._app()
        from ai_companion.ui.hud import ReactorCore

        core = self._keep(ReactorCore())
        core.show()
        core.set_active(True)
        angle = core._angle
        core.set_active(True)
        assert core._angle == angle

    # ---- painting ----

    def test_widgets_paint_without_error(self):
        self._app()
        from PySide6.QtGui import QPixmap

        from ai_companion.ui.hud import (
            GridBackground,
            HudPanel,
            PulseBar,
            ReactorCore,
        )

        for factory in (HudPanel, GridBackground, ReactorCore, PulseBar):
            widget = self._keep(factory())
            widget.resize(120, 60)
            assert not QPixmap(widget.grab()).isNull()

    def test_telemetry_updates_values(self):
        self._app()
        from ai_companion.ui.hud import TelemetryStrip

        strip = TelemetryStrip()
        strip.add_field("model", "MODEL", "OFFLINE")
        strip.set_value("model", "QWEN")
        assert strip._items["model"].text() == "QWEN"

    def test_telemetry_unknown_key_is_ignored(self):
        self._app()
        from ai_companion.ui.hud import TelemetryStrip

        strip = TelemetryStrip()
        strip.set_value("nope", "x")  # must not raise

    # ---- nav rail ----

    def test_nav_rail_selection_is_exclusive(self):
        self._app()
        from ai_companion.ui.nav_rail import NavRail

        rail = NavRail()
        for icon, label in (("chat", "CHAT"), ("memory", "MEM")):
            rail.add_destination(icon, label)
        rail.set_current(1)
        assert [c._selected for c in rail._cells] == [False, True]

    def test_nav_rail_emits_index(self):
        self._app()
        from ai_companion.ui.nav_rail import NavRail

        rail = NavRail()
        rail.add_destination("chat", "CHAT")
        rail.add_destination("memory", "MEM")
        seen = []
        rail.navigated.connect(seen.append)
        rail._cells[1].activated.emit()
        assert seen == [1]

    def test_fade_in_clears_its_effect(self, qtbot=None):
        """A lingering opacity effect forces offscreen rendering forever."""
        import time

        app = self._app()
        from PySide6.QtWidgets import QWidget

        from ai_companion.ui.hud import fade_in

        widget = self._keep(QWidget())
        widget.resize(50, 50)
        widget.show()
        fade_in(widget, duration=30)
        deadline = time.time() + 2.0
        while time.time() < deadline and widget.graphicsEffect() is not None:
            app.processEvents()
            time.sleep(0.01)
        assert widget.graphicsEffect() is None

    def test_fade_never_starts_fully_transparent(self):
        """Starting at 0 makes the panel look broken mid-switch."""
        self._app()
        from PySide6.QtWidgets import QWidget

        from ai_companion.ui.hud import fade_in

        widget = self._keep(QWidget())
        widget.show()
        anim = fade_in(widget)
        assert anim is not None
        assert anim.startValue() >= 0.5


class TestBootSequence:
    """Boot screen must report real work, never fake it — 2026-09-18."""

    def _app(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        return QApplication.instance() or QApplication([])

    _KEEPALIVE: list = []

    def _make(self, steps):
        """Build a BootScreen and keep it alive.

        A BootScreen owns running QTimers. If Python collects the widget while
        Qt still holds timer callbacks into it, the interpreter dies with a
        bus error - which is exactly what happened the first time these tests
        ran. Holding a reference for the module's lifetime avoids it.
        """
        from ai_companion.ui.boot import BootScreen

        screen = BootScreen(steps)
        type(self)._KEEPALIVE.append(screen)
        return screen

    def _drain(self, app, screen, limit=600):
        import time

        for _ in range(limit):
            app.processEvents()
            time.sleep(0.005)
            if screen._done:
                return True
        return False

    def test_steps_actually_run_their_actions(self):
        """The whole point: the log reflects work, not a scripted timer."""
        app = self._app()
        from ai_companion.ui.boot import BootStep

        ran = []
        screen = self._make([
            BootStep("ONE", lambda: ran.append(1)),
            BootStep("TWO", lambda: ran.append(2)),
        ])
        screen.show()
        screen.start()
        assert self._drain(app, screen)
        assert ran == [1, 2]

    def test_failing_step_is_marked_and_does_not_abort(self):
        """A cosmetic screen must never stop the app from opening."""
        app = self._app()
        from ai_companion.ui.boot import BootStep

        def boom():
            raise RuntimeError("nope")

        screen = self._make([
            BootStep("GOOD"),
            BootStep("BAD", boom),
            BootStep("AFTER"),
        ])
        screen.show()
        screen.start()
        assert self._drain(app, screen)
        states = dict(screen._results)
        assert states["BAD"] == "FAIL"
        assert states["AFTER"] == "OK"

    def test_finished_signal_emitted_once(self):
        app = self._app()
        from ai_companion.ui.boot import BootStep

        seen = []
        screen = self._make([BootStep("ONE")])
        screen.finished.connect(lambda: seen.append(1))
        screen.show()
        screen.start()
        assert self._drain(app, screen)
        screen._complete()  # must not fire again
        assert seen == [1]

    def test_skip_still_runs_remaining_work(self):
        """Skipping skips the WAITING, never the work.

        Otherwise an impatient click leaves services unstarted.
        """
        self._app()
        from ai_companion.ui.boot import BootStep

        ran = []
        screen = self._make([
            BootStep("ONE", lambda: ran.append(1)),
            BootStep("TWO", lambda: ran.append(2)),
            BootStep("THREE", lambda: ran.append(3)),
        ])
        screen.show()
        screen.skip()
        assert ran == [1, 2, 3]
        assert screen._done is True

    def test_skip_survives_a_failing_step(self):
        self._app()
        from ai_companion.ui.boot import BootStep

        ran = []

        def boom():
            raise RuntimeError("nope")

        screen = self._make([
            BootStep("BAD", boom),
            BootStep("GOOD", lambda: ran.append(1)),
        ])
        screen.show()
        screen.skip()
        assert ran == [1]

    def test_timer_stops_when_finished(self):
        """A spinner left running behind a closed screen burns CPU forever."""
        app = self._app()
        from ai_companion.ui.boot import BootStep

        screen = self._make([BootStep("ONE")])
        screen.show()
        screen.start()
        assert self._drain(app, screen)
        assert screen._spin.isActive() is False

    def test_timer_stops_when_hidden(self):
        self._app()
        from ai_companion.ui.boot import BootStep

        screen = self._make([BootStep("ONE")])
        screen.show()
        screen.start()
        screen.hide()
        assert screen._spin.isActive() is False

    def test_paints_without_error(self):
        self._app()
        from PySide6.QtGui import QPixmap

        from ai_companion.ui.boot import BootStep

        screen = self._make([BootStep("ONE"), BootStep("TWO")])
        screen.resize(800, 600)
        assert not QPixmap(screen.grab()).isNull()


class TestVaultModeMigration:
    """Existing configs must not stay stuck on the old private default."""

    def _cm(self, monkeypatch, tmp_path, data):
        import json

        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager

        target = tmp_path / "config.json"
        target.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )
        return ConfigManager(), target

    def test_old_private_config_is_flipped_once(self, tmp_path, monkeypatch):
        """The whole point: private-by-default silently discarded data."""
        from ai_companion.models.config_models import VaultMode

        cm, _ = self._cm(monkeypatch, tmp_path, {"vault": {"mode": "private"}})
        assert cm.load().vault.mode == VaultMode.NORMAL

    def test_migration_explains_itself(self, tmp_path, monkeypatch):
        cm, _ = self._cm(monkeypatch, tmp_path, {"vault": {"mode": "private"}})
        cm.load()
        assert cm.migration_notes
        assert "SAVING" in cm.migration_notes[0]

    def test_other_settings_survive_migration(self, tmp_path, monkeypatch):
        cm, _ = self._cm(monkeypatch, tmp_path, {
            "llm": {
                "model_path": "C:/models/Qwen.gguf",
                "context_length": 4096,
                "max_tokens": 1536,
                "system_prompt": "BE TERSE",
            },
            "vault": {"mode": "private"},
        })
        cfg = cm.load()
        assert cfg.llm.model_path == "C:/models/Qwen.gguf"
        assert cfg.llm.context_length == 4096
        assert cfg.llm.max_tokens == 1536
        assert cfg.llm.system_prompt == "BE TERSE"

    def test_migration_runs_only_once(self, tmp_path, monkeypatch):
        """A later, deliberate choice of private must be respected."""
        from ai_companion.config import ConfigManager
        from ai_companion.models.config_models import VaultMode

        cm, _ = self._cm(monkeypatch, tmp_path, {"vault": {"mode": "private"}})
        cfg = cm.load()
        assert cfg.vault.mode == VaultMode.NORMAL

        cfg.vault.mode = VaultMode.PRIVATE
        cm.save()

        assert ConfigManager().load().vault.mode == VaultMode.PRIVATE

    def test_normal_config_untouched(self, tmp_path, monkeypatch):
        from ai_companion.models.config_models import VaultMode

        cm, _ = self._cm(monkeypatch, tmp_path, {"vault": {"mode": "normal"}})
        assert cm.load().vault.mode == VaultMode.NORMAL
        assert cm.migration_notes == []

    def test_marker_written_and_not_a_config_field(self, tmp_path,
                                                   monkeypatch):
        """The marker lives in the file, never on AppConfig."""
        import json

        cm, target = self._cm(
            monkeypatch, tmp_path, {"vault": {"mode": "private"}}
        )
        cfg = cm.load()
        cm.save()
        saved = json.loads(target.read_text(encoding="utf-8"))
        assert saved["migration_version"] >= 2
        assert not hasattr(cfg, "migration_version")

    def test_corrupt_config_still_loads_defaults(self, tmp_path, monkeypatch):
        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager
        from ai_companion.models.config_models import VaultMode

        target = tmp_path / "config.json"
        target.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )
        assert ConfigManager().load().vault.mode == VaultMode.NORMAL


class TestSettingsDialogDoesNotClobber:
    """Settings must not silently undo changes made elsewhere — 2026-09-19."""

    _KEEPALIVE: list = []

    def _stack(self, tmp_path, monkeypatch, mode="normal", model=""):
        import sys
        import types

        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication([])

        # Stub llama_cpp so a load attempt cannot raise, and neuter modal
        # popups so a headless run never blocks.
        if "llama_cpp" not in sys.modules:
            stub = types.ModuleType("llama_cpp")

            class _Llama:
                def __init__(self, **kwargs):
                    pass

            stub.Llama = _Llama
            sys.modules["llama_cpp"] = stub
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.models.config_models import AppConfig, VaultMode
        from ai_companion.services.llm_service import LlmService
        from ai_companion.services.vault_service import VaultService

        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH",
            str(tmp_path / "config.json"),
        )
        cfg = AppConfig().relocate(tmp_path)
        cfg.vault.mode = VaultMode(mode)
        cfg.llm.model_path = model
        cm = ConfigManager()
        cm._config = cfg
        sm = ServiceManager(cm)
        for cls in (LlmService, VaultService):
            sm.register(cls(cfg, sm.signal_bus))
        sm.start_all()
        return app, sm, cfg

    def _dialog(self, sm):
        from ai_companion.ui.settings_panel import SettingsPanel

        dlg = SettingsPanel(sm)
        dlg.accept = lambda: None  # do not close during the test
        type(self)._KEEPALIVE.append(dlg)
        return dlg

    def test_header_badge_change_survives_settings_save(
        self, tmp_path, monkeypatch
    ):
        """The exact bug: Save reverted a badge click made while open."""
        from ai_companion.models.config_models import VaultMode

        _, sm, cfg = self._stack(tmp_path, monkeypatch, mode="normal")
        dlg = self._dialog(sm)
        sm.get("Vault").set_mode(VaultMode.PRIVATE)
        dlg._save_settings()
        assert cfg.vault.mode == VaultMode.PRIVATE

    def test_reverse_direction_also_survives(self, tmp_path, monkeypatch):
        from ai_companion.models.config_models import VaultMode

        _, sm, cfg = self._stack(tmp_path, monkeypatch, mode="private")
        dlg = self._dialog(sm)
        sm.get("Vault").set_mode(VaultMode.NORMAL)
        dlg._save_settings()
        assert cfg.vault.mode == VaultMode.NORMAL

    def test_dropdown_change_is_still_applied(self, tmp_path, monkeypatch):
        """Guard the fix: ignoring the dropdown entirely would be worse."""
        from ai_companion.models.config_models import VaultMode

        _, sm, cfg = self._stack(tmp_path, monkeypatch, mode="normal")
        dlg = self._dialog(sm)
        dlg._vault_mode.setCurrentText("private")
        dlg._save_settings()
        assert cfg.vault.mode == VaultMode.PRIVATE

    def test_dropdown_change_reaches_the_vault_service(
        self, tmp_path, monkeypatch
    ):
        """Changing it here must move the badge too, not just config."""
        _, sm, _ = self._stack(tmp_path, monkeypatch, mode="normal")
        dlg = self._dialog(sm)
        dlg._vault_mode.setCurrentText("private")
        dlg._save_settings()
        assert sm.get("Vault").is_private is True

    def test_empty_field_cannot_wipe_saved_model_path(
        self, tmp_path, monkeypatch
    ):
        """An empty box must never erase a working model path."""
        model = tmp_path / "Qwen.gguf"
        model.write_bytes(b"\x00" * 2048)
        _, sm, cfg = self._stack(tmp_path, monkeypatch, model=str(model))
        dlg = self._dialog(sm)
        dlg._model_path.setText("")
        dlg._save_settings()
        assert cfg.llm.model_path == str(model)

    def test_whitespace_only_path_is_ignored(self, tmp_path, monkeypatch):
        model = tmp_path / "Qwen.gguf"
        model.write_bytes(b"\x00" * 2048)
        _, sm, cfg = self._stack(tmp_path, monkeypatch, model=str(model))
        dlg = self._dialog(sm)
        dlg._model_path.setText("   ")
        dlg._save_settings()
        assert cfg.llm.model_path == str(model)

    def test_a_real_path_change_is_applied(self, tmp_path, monkeypatch):
        old = tmp_path / "Old.gguf"
        old.write_bytes(b"\x00" * 2048)
        new = tmp_path / "New.gguf"
        new.write_bytes(b"\x00" * 2048)
        _, sm, cfg = self._stack(tmp_path, monkeypatch, model=str(old))
        dlg = self._dialog(sm)
        dlg._model_path.setText(str(new))
        dlg._save_settings()
        assert cfg.llm.model_path == str(new)

    def test_other_llm_settings_still_save(self, tmp_path, monkeypatch):
        _, sm, cfg = self._stack(tmp_path, monkeypatch)
        dlg = self._dialog(sm)
        dlg._max_tokens.setValue(1536)
        dlg._ctx_length.setValue(2048)
        dlg._save_settings()
        assert cfg.llm.max_tokens == 1536
        assert cfg.llm.context_length == 2048


class TestTestsCannotClobberRealConfig:
    """The suite must never touch the developer's config — 2026-09-19.

    Running the tests used to rewrite the real config.json: model_path wiped
    to "" and vault flipped to private. install_patch.py runs the suite, so
    every install silently destroyed the user's settings.
    """

    def test_config_path_is_redirected_during_tests(self):
        """conftest points DEFAULT_CONFIG_PATH at a sandbox for the session."""
        from pathlib import Path

        import ai_companion
        from ai_companion.config import DEFAULT_CONFIG_PATH

        root = Path(ai_companion.__file__).resolve().parent.parent
        assert Path(DEFAULT_CONFIG_PATH) != root / "config.json"

    def test_config_manager_writes_go_to_the_sandbox(self):
        from pathlib import Path

        from ai_companion.config import ConfigManager

        cm = ConfigManager()
        cm.load()
        cm.save()
        assert "config_sandbox" in str(Path(cm._path))


class TestGraphLinkContrast:
    """Links must be visible against the background — 2026-09-19.

    They previously rendered at 1.08:1 effective contrast, which is invisible
    in practice. Measured, not eyeballed.
    """

    @staticmethod
    def _rgb(value):
        value = value.lstrip("#")
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    @classmethod
    def _luminance(cls, rgb):
        def channel(v):
            v /= 255
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

        r, g, b = (channel(x) for x in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @classmethod
    def _contrast(cls, a, b):
        la, lb = cls._luminance(a), cls._luminance(b)
        hi, lo = max(la, lb), min(la, lb)
        return (hi + 0.05) / (lo + 0.05)

    @classmethod
    def _composite(cls, fg_hex, alpha, bg_hex):
        ratio = alpha / 255
        return tuple(
            ratio * f + (1 - ratio) * b
            for f, b in zip(cls._rgb(fg_hex), cls._rgb(bg_hex))
        )

    def test_link_token_exists(self):
        from ai_companion.ui.theme import theme

        assert theme.p.link

    def test_weakest_link_is_visible(self):
        """A single shared memory must still draw a traceable line."""
        from ai_companion.ui.theme import theme

        p = theme.p
        alpha = 170 + int(85 * (1 / 5.0))   # strength for shared=1
        blended = self._composite(p.link, alpha, p.bg_base)
        ratio = self._contrast(blended, self._rgb(p.bg_base))
        assert ratio > 2.0, f"weakest link only {ratio:.2f}:1"

    def test_strongest_link_is_clearly_visible(self):
        from ai_companion.ui.theme import theme

        p = theme.p
        blended = self._composite(p.link, 255, p.bg_base)
        ratio = self._contrast(blended, self._rgb(p.bg_base))
        assert ratio > 4.0, f"strongest link only {ratio:.2f}:1"

    def test_dimmed_link_still_traceable(self):
        """Dimming must not look like the graph deleted itself."""
        from ai_companion.ui.theme import theme

        p = theme.p
        alpha = max(80, (170 + int(85 * 0.2)) // 3)
        blended = self._composite(p.link, alpha, p.bg_base)
        assert self._contrast(blended, self._rgb(p.bg_base)) > 1.25

    def test_link_does_not_outshine_the_accent(self):
        """Selection highlight must remain the brightest thing on screen."""
        from ai_companion.ui.theme import theme

        p = theme.p
        bg = self._rgb(p.bg_base)
        assert self._contrast(self._rgb(p.accent), bg) > self._contrast(
            self._rgb(p.link), bg
        ) * 0.9

    def test_link_brighter_than_structural_border(self):
        from ai_companion.ui.theme import theme

        p = theme.p
        bg = self._rgb(p.bg_base)
        assert self._contrast(self._rgb(p.link), bg) > self._contrast(
            self._rgb(p.border_strong), bg
        )


class TestConfigRejectsTempPaths:
    """A config pointing at temp dirs must self-heal — 2026-09-19.

    An earlier build let the test suite write its relocated config over the
    real one. Windows deletes those temp folders, so the app silently
    recreated them and wrote memories somewhere that vanishes. The user saw
    an empty MEM tab and lost everything added afterwards.
    """

    def _load(self, tmp_path, monkeypatch, data):
        import json

        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager

        target = tmp_path / "config.json"
        target.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )
        manager = ConfigManager()
        return manager, manager.load()

    def test_pytest_temp_store_path_is_reset(self, tmp_path, monkeypatch):
        """The exact damage Steven's diagnostic reported."""
        gone = (
            "C:/Users/jooe3/AppData/Local/Temp/pytest-of-jooe3/pytest-27/"
            "test_only_active_chat_affected0"
        )
        _, config = self._load(tmp_path, monkeypatch, {
            "data_dir": f"{gone}/data",
            "memory": {"store_path": f"{gone}/data/memories.json"},
            "vault": {"mode": "normal"},
        })
        # Paths are now ANCHORED to the app root rather than left relative,
        # so a frozen exe reads its stores from beside the executable rather
        # than from whatever directory it was launched in.
        # Compare path COMPONENTS, not the raw string: Windows produces
        # backslashes and an endswith("data/memories.json") check fails there
        # even though the value is correct.
        from pathlib import Path

        assert Path(config.memory.store_path).parts[-2:] == (
            "data", "memories.json"
        )
        assert "pytest" not in config.memory.store_path
        assert Path(config.data_dir).name == "data"

    def test_unix_tmp_path_is_reset(self, tmp_path, monkeypatch):
        _, config = self._load(tmp_path, monkeypatch, {
            "memory": {"store_path": "/tmp/pytest-of-user/x/memories.json"},
        })
        from pathlib import Path

        assert Path(config.memory.store_path).parts[-2:] == (
            "data", "memories.json"
        )
        assert "pytest" not in config.memory.store_path

    def test_every_store_field_is_covered(self, tmp_path, monkeypatch):
        gone = "/tmp/pytest-of-user/pytest-1/t0"
        _, config = self._load(tmp_path, monkeypatch, {
            "llm": {"conversations_path": f"{gone}/conversations.json"},
            "memory": {"store_path": f"{gone}/memories.json"},
            "graph": {"store_path": f"{gone}/graph.json"},
            "mind_map": {"store_path": f"{gone}/mind_map.json"},
            "vault": {"vault_root": f"{gone}/vault", "mode": "normal"},
            "code_workspace": {"sandbox_dir": f"{gone}/sandbox"},
            "image_gen": {"output_dir": f"{gone}/images"},
            "speech": {"voices_dir": f"{gone}/piper"},
        })
        # Check the pytest FIXTURE path specifically. Paths are now anchored
        # to the app root, and on Linux CI that root can legitimately live
        # under /tmp - so a blanket "/tmp/" ban fails for the wrong reason.
        for value in (
            config.llm.conversations_path,
            config.memory.store_path,
            config.graph.store_path,
            config.mind_map.store_path,
            config.vault.vault_root,
            config.code_workspace.sandbox_dir,
            config.image_gen.output_dir,
            config.speech.voices_dir,
        ):
            assert "pytest-of-" not in value, value
            assert gone not in value, value

    def test_legitimate_paths_are_untouched(self, tmp_path, monkeypatch):
        """Guard the fix: resetting a valid custom path would be a new bug."""
        _, config = self._load(tmp_path, monkeypatch, {
            "memory": {"store_path": "D:/MyData/memories.json"},
            "llm": {"model_path": "C:/models/Qwen.gguf"},
        })
        # Absolute custom paths must pass through untouched - anchoring
        # applies only to relative ones.
        assert config.memory.store_path == "D:/MyData/memories.json"
        assert config.llm.model_path == "C:/models/Qwen.gguf"

    def test_other_settings_survive_the_reset(self, tmp_path, monkeypatch):
        _, config = self._load(tmp_path, monkeypatch, {
            "data_dir": "/tmp/pytest-of-user/x/data",
            "llm": {"model_path": "C:/models/Qwen.gguf", "max_tokens": 1536},
            "memory": {"store_path": "/tmp/pytest-of-user/x/memories.json"},
        })
        assert config.llm.model_path == "C:/models/Qwen.gguf"
        assert config.llm.max_tokens == 1536

    def test_the_user_is_told(self, tmp_path, monkeypatch):
        """Silent repair would leave them wondering where memories went."""
        manager, _ = self._load(tmp_path, monkeypatch, {
            "memory": {"store_path": "/tmp/pytest-of-user/x/memories.json"},
        })
        assert manager.migration_notes
        assert "temp folder" in manager.migration_notes[0]

    def test_clean_config_produces_no_notice(self, tmp_path, monkeypatch):
        manager, _ = self._load(tmp_path, monkeypatch, {
            "memory": {"store_path": "data/memories.json"},
            "vault": {"mode": "normal"},
        })
        assert manager.migration_notes == []


class TestDataSnapshots:
    """Every launch keeps a recoverable copy — 2026-09-19.

    The user lost memories to a storage path that silently pointed at a
    deleted temp folder. A rolling on-disk history means that class of
    failure can never be unrecoverable again.
    """

    def _config(self, tmp_path):
        from ai_companion.models.config_models import AppConfig

        config = AppConfig().relocate(tmp_path)
        data = tmp_path / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "memories.json").write_text(
            '{"_meta": {}, "items": {"m1": {"id": "m1"}}}', encoding="utf-8"
        )
        return config

    def test_snapshot_is_created(self, tmp_path, monkeypatch):
        import ai_companion.main as main_module

        config = self._config(tmp_path)
        monkeypatch.setattr(
            main_module, "__file__", str(tmp_path / "ai_companion" / "main.py")
        )
        main_module._snapshot_data(config)

        backups = tmp_path / "data_backups"
        assert backups.is_dir()
        snapshots = list(backups.iterdir())
        assert len(snapshots) == 1
        assert (snapshots[0] / "memories.json").exists()

    def test_snapshot_contents_match(self, tmp_path, monkeypatch):
        import ai_companion.main as main_module

        config = self._config(tmp_path)
        monkeypatch.setattr(
            main_module, "__file__", str(tmp_path / "ai_companion" / "main.py")
        )
        main_module._snapshot_data(config)

        original = (tmp_path / "data" / "memories.json").read_text("utf-8")
        snapshot = next((tmp_path / "data_backups").iterdir())
        assert (snapshot / "memories.json").read_text("utf-8") == original

    def test_old_snapshots_are_pruned(self, tmp_path, monkeypatch):
        """Unbounded growth would eventually fill the disk."""
        import ai_companion.main as main_module

        config = self._config(tmp_path)
        monkeypatch.setattr(
            main_module, "__file__", str(tmp_path / "ai_companion" / "main.py")
        )
        backups = tmp_path / "data_backups"
        backups.mkdir(parents=True, exist_ok=True)
        for i in range(15):
            (backups / f"2026010{i:02d}-000000").mkdir()

        main_module._snapshot_data(config)
        assert len(list(backups.iterdir())) <= main_module.MAX_DATA_SNAPSHOTS

    def test_empty_data_dir_is_skipped(self, tmp_path, monkeypatch):
        """No point snapshotting nothing."""
        import ai_companion.main as main_module

        from ai_companion.models.config_models import AppConfig

        config = AppConfig().relocate(tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            main_module, "__file__", str(tmp_path / "ai_companion" / "main.py")
        )
        main_module._snapshot_data(config)
        assert not (tmp_path / "data_backups").exists()

    def test_missing_data_dir_does_not_raise(self, tmp_path, monkeypatch):
        import ai_companion.main as main_module

        from ai_companion.models.config_models import AppConfig

        config = AppConfig().relocate(tmp_path / "nonexistent")
        monkeypatch.setattr(
            main_module, "__file__", str(tmp_path / "ai_companion" / "main.py")
        )
        main_module._snapshot_data(config)   # must not raise

    def test_json_store_warns_on_temp_path(self, tmp_path, caplog):
        """A tripwire: production stores must never live in temp."""
        import logging

        from ai_companion.infrastructure.json_store import JsonStore

        with caplog.at_level(logging.WARNING):
            JsonStore("/tmp/pytest-of-user/x/memories.json")
        assert any(
            "temporary directory" in record.message for record in caplog.records
        )

    def test_json_store_silent_for_normal_path(self, tmp_path, caplog):
        import logging

        from ai_companion.infrastructure.json_store import JsonStore

        with caplog.at_level(logging.WARNING):
            JsonStore(tmp_path / "data" / "memories.json", allow_temp=True)
        assert not any(
            "temporary directory" in record.message for record in caplog.records
        )


class TestPackagingSpec:
    """The .exe spec must not drift from the real dependencies — 2026-09-20.

    The previous spec predated the speech work: it listed 14 hidden imports,
    missed nine real dependencies, and EXCLUDED scipy, which voice_fx needs.
    A stale spec produces a build that compiles cleanly and then crashes the
    first time you use the affected feature.
    """

    def _spec(self):
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent.parent
        return (root / "ai_companion.spec").read_text(encoding="utf-8")

    def test_spec_exists_and_parses(self):
        import ast

        ast.parse(self._spec())

    def test_scipy_is_not_excluded(self):
        """It was, and voice post-processing would have failed at runtime.

        Checks actual quoted entries, not a substring of the whole block:
        the block contains an explanatory comment mentioning scipy, and
        naive matching flagged that as an exclusion.
        """
        import re

        spec = self._spec()
        block = spec.split("excludes=[")[1].split("]")[0]
        entries = re.findall(r'["\']([a-zA-Z_][\w.]*)["\']', block)
        assert "scipy" not in entries, f"scipy excluded, entries={entries}"

    def test_speech_packages_are_hidden_imports(self):
        """These are imported lazily, so static analysis never sees them."""
        spec = self._spec()
        for package in (
            "faster_whisper", "piper", "kokoro_onnx", "sounddevice",
            "onnxruntime", "ctranslate2", "scipy", "spellchecker",
        ):
            assert f'"{package}"' in spec, package

    def test_runtime_data_files_are_collected(self):
        """Files opened by path, which PyInstaller cannot infer."""
        spec = self._spec()
        assert "collect_data_files" in spec
        for package in (
            "faster_whisper", "piper", "kokoro_onnx", "spellchecker",
            "espeakng_loader",
        ):
            assert package in spec, package

    def test_upx_is_disabled(self):
        """UPX corrupts Qt and ONNX DLLs, and it fails at launch not build."""
        assert "upx=False" in self._spec()
        assert "upx=True" not in self._spec()

    def test_one_folder_not_one_file(self):
        """One-file unpacks ~1 GB to temp on every launch."""
        assert "COLLECT" in self._spec()

    def test_build_script_exists_and_parses(self):
        import ast
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent.parent
        script = root / "build_exe.py"
        assert script.is_file()
        ast.parse(script.read_text(encoding="utf-8"))

    def test_every_app_dependency_appears_in_the_spec(self):
        """Guard against the spec rotting as the app grows again."""
        import re
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent
        spec = self._spec()

        third_party = set()
        pattern = re.compile(r"^\s*(?:from|import)\s+([a-z_][a-z0-9_]*)", re.M)
        stdlib_or_local = {
            "ai_companion", "abc", "argparse", "base64", "collections",
            "contextlib", "csv", "dataclasses", "datetime", "difflib", "enum",
            "functools", "hashlib", "io", "itertools", "json", "logging",
            "math", "os", "pathlib", "queue", "random", "re", "shutil",
            "subprocess", "sys", "tempfile", "threading", "time", "traceback",
            "types", "typing", "uuid", "wave", "warnings", "zipfile", "html",
            "urllib", "injection", "whisper", "pyttsx3", "__future__",
            "importlib", "inspect", "copy", "string", "textwrap", "glob",
            "signal", "platform", "socket", "struct", "binascii", "gzip",
        }
        for path in root.rglob("*.py"):
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                name = match.group(1)
                if name not in stdlib_or_local:
                    third_party.add(name)

        # Check the hiddenimports list specifically. Searching the whole file
        # gave false passes: a package named only in a comment or in the
        # collect_data_files loop still "appeared", so removing it from
        # hiddenimports went undetected.
        hidden_block = spec.split("hiddenimports = [")[1].split("\n]")[0]
        hidden = set(re.findall(r'["\']([a-zA-Z_][\w.]*)["\']', hidden_block))
        hidden_roots = {name.split(".")[0] for name in hidden}

        missing = sorted(third_party - hidden_roots)
        assert not missing, f"missing from hiddenimports: {missing}"


class TestFrozenPaths:
    """Packaged exe must find its own data — 2026-09-20.

    The first .exe build launched but reported "no voices installed" and
    failed to load the model, because paths anchored to config.py's location
    resolve to `_internal\\` inside a PyInstaller bundle rather than the
    folder holding the executable.
    """

    def test_app_root_from_source(self):
        from pathlib import Path

        import ai_companion
        from ai_companion.config import app_root

        expected = Path(ai_companion.__file__).resolve().parent.parent
        assert app_root() == expected

    def test_app_root_when_frozen(self, monkeypatch, tmp_path):
        """Frozen: beside the exe, NOT inside _internal."""
        import sys

        from ai_companion.config import app_root

        exe = tmp_path / "AICompanion" / "AICompanion.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))

        assert app_root() == exe.parent
        assert "_internal" not in str(app_root())

    def test_windows_drive_letters_count_as_absolute(self):
        """`Path('D:/x').is_absolute()` is False on Linux.

        Without this check a real Windows path was mangled into
        '/home/user/D:/MyData'.
        """
        import os

        from ai_companion.config import _is_absolute

        # Drive-letter forms must be absolute on EVERY platform - that is the
        # bug this guards. A bare "/abs/path" is absolute on POSIX but only
        # drive-RELATIVE on Windows, so it is asserted per-platform.
        for value in ("D:/MyData/x.json", "C:\\models\\q.gguf"):
            assert _is_absolute(value), value
        if os.name != "nt":
            assert _is_absolute("/abs/path")

    def test_relative_paths_are_not_absolute(self):
        from ai_companion.config import _is_absolute

        for value in ("data/x.json", "models/piper", ""):
            assert not _is_absolute(value), value

    def test_relative_stores_are_anchored_on_load(self, tmp_path, monkeypatch):
        import json

        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager

        target = tmp_path / "config.json"
        target.write_text(json.dumps({
            "memory": {"store_path": "data/memories.json"},
            "speech": {"voices_dir": "models/piper"},
        }), encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )
        config = ConfigManager().load()

        from pathlib import Path

        assert Path(config.memory.store_path).is_absolute()
        assert Path(config.speech.voices_dir).is_absolute()
        assert Path(config.memory.store_path).parts[-2:] == (
            "data", "memories.json"
        )

    def test_absolute_stores_are_left_alone(self, tmp_path, monkeypatch):
        import json

        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager

        target = tmp_path / "config.json"
        target.write_text(json.dumps({
            "memory": {"store_path": "D:/MyData/memories.json"},
        }), encoding="utf-8")
        monkeypatch.setattr(
            config_module, "DEFAULT_CONFIG_PATH", str(target)
        )
        config = ConfigManager().load()
        assert config.memory.store_path == "D:/MyData/memories.json"

    def test_spec_bundles_llama_cpp_libraries(self):
        """llama-cpp ships prebuilt DLLs loaded by ctypes, not imported.

        Missing them produced:
          [WinError 3] cannot find the path specified:
          '...\\_internal\\llama_cpp\\lib'
        """
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent.parent
        spec = (root / "ai_companion.spec").read_text(encoding="utf-8")
        assert "llama_cpp/lib" in spec


class TestBuildScriptGuards:
    """The build must verify its own output — 2026-09-20.

    The first exe compiled cleanly and then failed with WinError 3 because
    llama-cpp's native libraries were not bundled. A build that reports
    success while producing a broken binary is worse than one that fails.
    """

    def _script(self):
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent.parent
        return (root / "build_exe.py").read_text(encoding="utf-8")

    def test_verifies_llama_libraries(self):
        """The exact thing that was missing on the first build."""
        assert "llama_cpp/lib" in self._script()

    def test_verifies_speech_assets(self):
        script = self._script()
        assert "faster_whisper/assets" in script
        assert "spellchecker/resources" in script

    def test_copies_user_data_beside_the_exe(self):
        """Requiring a manual xcopy was a step easy to miss."""
        script = self._script()
        assert "COPYING YOUR DATA" in script
        for name in ("config.json", "models", "data"):
            assert name in script

    def test_verify_build_script_exists(self):
        import ast
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent.parent
        script = root / "verify_build.py"
        assert script.is_file()
        ast.parse(script.read_text(encoding="utf-8"))

    def test_verify_build_checks_staleness(self):
        """A stale exe is the likeliest reason a fix appears not to work."""
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent.parent
        script = (root / "verify_build.py").read_text(encoding="utf-8")
        assert "STALE" in script
        assert "st_mtime" in script


class TestNoLinuxOnlyPathAssumptions:
    """Tests must pass on Windows too — 2026-09-20.

    Four tests shipped that failed only on Steven's machine, because they
    asserted `endswith("data/memories.json")`. Windows produces backslashes,
    so a correct value failed a correct-looking assertion.

    The app runs on Windows; the tests run here. That mismatch has now
    produced the same class of bug twice, so it gets a guard.
    """

    def _test_source(self) -> str:
        from pathlib import Path

        import ai_companion

        root = Path(ai_companion.__file__).resolve().parent.parent
        return "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((root / "tests").glob("test_*.py"))
        )

    def test_no_forward_slash_path_endswith(self):
        """`endswith("a/b.json")` is false on Windows for a correct path."""
        import ast
        import re
        from pathlib import Path

        import ai_companion

        # Parse the AST rather than grepping text: the first version matched
        # its own docstring and comments explaining the rule, which made the
        # guard fail for the wrong reason.
        root = Path(ai_companion.__file__).resolve().parent.parent
        offenders = []
        for path in sorted((root / "tests").glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr != "endswith" or not node.args:
                    continue
                arg = node.args[0]
                if not isinstance(arg, ast.Constant):
                    continue
                value = arg.value
                if not isinstance(value, str):
                    continue
                if "/" in value and ("." in value or "data" in value):
                    offenders.append(f"{path.name}:{node.lineno} {value!r}")
        assert not offenders, (
            f"path endswith() with forward slashes will fail on Windows: "
            f"{offenders}. Compare Path(...).parts instead."
        )

    # REMOVED: test_windows_semantics_are_simulated
    #
    # It asserted ntpath.isabs("/abs/path") is False. That is a statement
    # about CPython, not about this app, and it is version-dependent:
    # gh-104614 changed the behaviour in 3.13, so the test passed on the
    # 3.13 sandbox it was written on and failed on Steven's 3.12.
    #
    # A test that cannot catch a bug in our code, and breaks when the stdlib
    # changes, is worse than no test. The behaviour that actually matters -
    # that _is_absolute() treats drive letters as absolute on every platform
    # and Python version - is covered below.

    def test_is_absolute_agrees_with_both_platforms(self):
        """Drive letters are absolute on every platform AND Python version.

        This is the behaviour that matters: config may hold a Windows path
        while the tests run on Linux. Asserting our own function rather than
        a stdlib constant keeps it stable across Python releases - an earlier
        version of this class asserted ntpath behaviour that changed in 3.13
        and broke on Steven's 3.12.
        """
        from ai_companion.config import _is_absolute

        for value in (
            "C:/models/q.gguf", "C:\\models\\q.gguf",
            "D:/MyData/x.json", "Z:\\share\\y.json",
        ):
            assert _is_absolute(value), value
        for value in ("data/x.json", "models\\piper", "", "x.json"):
            assert not _is_absolute(value), value

    def test_anchoring_never_mangles_a_windows_path(self):
        """The bug this guards: '/home/user/D:/MyData'.

        Exercises the real anchoring code with a Windows-style path, on
        whatever platform the tests happen to run.
        """
        import json
        from pathlib import Path

        from ai_companion import config as config_module
        from ai_companion.config import ConfigManager

        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "config.json"
            target.write_text(json.dumps({
                "memory": {"store_path": "D:/MyData/memories.json"},
                "llm": {"model_path": "C:/models/Qwen.gguf"},
            }), encoding="utf-8")
            original = config_module.DEFAULT_CONFIG_PATH
            config_module.DEFAULT_CONFIG_PATH = str(target)
            try:
                config = ConfigManager().load()
            finally:
                config_module.DEFAULT_CONFIG_PATH = original

        assert config.memory.store_path == "D:/MyData/memories.json"
        assert config.llm.model_path == "C:/models/Qwen.gguf"
