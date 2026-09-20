"""Global test safety net.

WHY THIS EXISTS
---------------
`config.json` is anchored to the project root, and `VaultService.set_mode()`
persists to it. Together that meant running the test suite rewrote the
developer's REAL config: model path wiped to "", vault flipped to private.

Because `install_patch.py` runs the suite after copying files, every install
silently destroyed the user's settings. Reproduced:

    BEFORE  model_path: C:/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
            vault     : normal
    AFTER   model_path: ''
            vault     : private

Four test classes were responsible, but fixing them one by one is the wrong
remedy - the next test that calls `set_mode()` without `persist=False` would
reintroduce it. This redirects the config path for the WHOLE session, so no
test can reach the real file regardless of what it does.

The guard is autouse and session-scoped: it cannot be forgotten.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _never_touch_the_real_config(tmp_path_factory):
    """Point ConfigManager at a throwaway file for the entire test session."""
    from ai_companion import config as config_module

    sandbox = tmp_path_factory.mktemp("config_sandbox") / "config.json"
    original = config_module.DEFAULT_CONFIG_PATH
    config_module.DEFAULT_CONFIG_PATH = str(sandbox)
    try:
        yield
    finally:
        config_module.DEFAULT_CONFIG_PATH = original


@pytest.fixture(autouse=True)
def _fail_if_real_config_written(request):
    """Fail loudly if a test writes to the project's config.json.

    The session fixture above prevents it, but this catches any future code
    path that hardcodes the location instead of reading the module attribute.
    A silent regression here costs the user their settings.
    """
    from pathlib import Path

    import ai_companion

    real = Path(ai_companion.__file__).resolve().parent.parent / "config.json"
    before = real.stat().st_mtime_ns if real.exists() else None

    yield

    after = real.stat().st_mtime_ns if real.exists() else None
    if before != after:
        pytest.fail(
            f"{request.node.name} modified the real config.json at {real}. "
            "Tests must never touch it - patch "
            "ai_companion.config.DEFAULT_CONFIG_PATH instead."
        )
