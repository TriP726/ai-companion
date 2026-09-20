"""Diagnose why settings do not stick between launches.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe tools/diagnose_config.py

Read-only except for one clearly-labelled write test. Prints exactly which
config.json the app uses, what is in it, and whether it can be written.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    print("=" * 66)
    print("  CONFIG PERSISTENCE DIAGNOSTIC")
    print("=" * 66)

    print(f"\n[1] Interpreter\n    {sys.executable}")
    if ".venv" not in sys.executable:
        print("    WARNING: not the venv Python. Use .venv\\Scripts\\python.exe")

    print(f"\n[2] Working directory\n    {Path.cwd()}")

    try:
        from ai_companion.config import ConfigManager, DEFAULT_CONFIG_PATH
    except Exception as exc:  # noqa: BLE001
        print(f"\n    ERROR importing ai_companion: {exc}")
        print("    Run this from C:\\dev\\ai_companion")
        return 1

    target = Path(DEFAULT_CONFIG_PATH)
    print(f"\n[3] Config file the app will use\n    {target}")
    print(f"    exists   : {target.exists()}")
    if target.exists():
        stat = target.stat()
        print(f"    size     : {stat.st_size} bytes")
        print(f"    writable : {os.access(target, os.W_OK)}")
        import datetime
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        print(f"    modified : {mtime:%Y-%m-%d %H:%M:%S}")

    print("\n[4] Other config.json files that could be shadowing it")
    seen = 0
    for candidate in [
        Path.cwd() / "config.json",
        Path.home() / "config.json",
        Path.home() / "Desktop" / "config.json",
        Path("C:/config.json"),
    ]:
        if candidate.exists() and candidate.resolve() != target.resolve():
            print(f"    STRAY: {candidate}")
            seen += 1
    if seen == 0:
        print("    none found (good)")

    print("\n[5] Current contents")
    if not target.exists():
        print("    File does not exist yet.")
    else:
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"    UNREADABLE: {exc}")
            print("    This is the bug: an unparseable config silently")
            print("    resets every setting on launch. Delete it and redo")
            print("    your settings.")
            return 1
        llm = data.get("llm", {})
        print(f"    model_path        : {llm.get('model_path') or '(EMPTY)'}")
        print(f"    context_length    : {llm.get('context_length')}")
        print(f"    max_tokens        : {llm.get('max_tokens')}")
        print(f"    threads           : {llm.get('threads')}")
        print(f"    vault.mode        : {data.get('vault', {}).get('mode')}")
        print(f"    migration_version : {data.get('migration_version')}")

        path = llm.get("model_path") or ""
        if path:
            p = Path(path)
            print(f"\n    model file exists : {p.exists()}")
            if not p.exists():
                print("    ^ the path is saved but the FILE is missing, so the")
                print("      app loads the config fine and the model load fails.")

    print("\n[6] Write test (writes, then restores your original)")
    original = target.read_bytes() if target.exists() else None
    try:
        cm = ConfigManager()
        cfg = cm.load()
        marker = cfg.llm.max_tokens
        cfg.llm.max_tokens = 4321
        cm.save()
        check = json.loads(target.read_text(encoding="utf-8"))
        ok = check.get("llm", {}).get("max_tokens") == 4321
        print(f"    write succeeded   : {ok}")
        if not ok:
            print("    ^ THIS IS THE BUG: save() reported success but the")
            print("      value did not reach disk.")
        cfg.llm.max_tokens = marker
        cm.save()
    except Exception as exc:  # noqa: BLE001
        print(f"    WRITE FAILED: {type(exc).__name__}: {exc}")
        print("    Likely cause: the folder is read-only, or antivirus /")
        print("    OneDrive is locking config.json.")
        if original is not None:
            target.write_bytes(original)
        return 1

    leftover = target.with_suffix(".json.tmp")
    print(f"\n[7] Leftover temp file present: {leftover.exists()}")
    if leftover.exists():
        print("    ^ a previous save was interrupted mid-write.")

    print("\nDone. Paste this whole output back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
