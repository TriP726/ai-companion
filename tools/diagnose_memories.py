"""Find out what happened to the memories.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe tools/diagnose_memories.py

READ-ONLY. It never writes to your stores. It reports what is on disk, what
the app would load, and whether anything looks corrupted or misplaced.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def stamp(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except OSError:
        return "?"


def describe_store(path: Path, label: str) -> int:
    """Print what is in a JsonStore file. Returns the record count."""
    print(f"\n[{label}]")
    print(f"  path     : {path}")
    if not path.exists():
        print("  MISSING - this file does not exist")
        return 0

    size = path.stat().st_size
    print(f"  size     : {size} bytes")
    print(f"  modified : {stamp(path)}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  UNREADABLE: {exc}")
        print("  ^ A corrupt store loads as empty. This is recoverable only")
        print("    from a .bak file - see below.")
        return -1

    if not isinstance(raw, dict):
        print(f"  UNEXPECTED TYPE: {type(raw).__name__}")
        return -1

    items = raw.get("items")
    if items is None:
        # Records stored at the top level instead of under "items" would be
        # invisible to JsonStore, which is worth flagging loudly.
        stray = [k for k in raw if k != "_meta"]
        if stray:
            print(f"  WRONG SHAPE: {len(stray)} record(s) at the top level,")
            print("    but JsonStore reads them from an 'items' key.")
            print(f"    keys: {stray[:6]}")
            return -1
        print("  0 records (file has no 'items' key and no stray records)")
        return 0

    count = len(items)
    print(f"  records  : {count}")
    meta = raw.get("_meta", {})
    if meta:
        print(f"  created  : {meta.get('created', '?')}")
        print(f"  modified : {meta.get('modified', '?')}")

    if count:
        print("  sample   :")
        for key, value in list(items.items())[:3]:
            if isinstance(value, dict):
                text = str(value.get("content") or value.get("label") or "")
                status = value.get("status", "")
                private = value.get("private", False)
                flag = " [PRIVATE]" if private else ""
                print(f"    - {status:9} {text[:48]}{flag}")
            else:
                print(f"    - {key}")

        statuses: dict[str, int] = {}
        private_count = 0
        for value in items.values():
            if isinstance(value, dict):
                statuses[value.get("status", "?")] = statuses.get(
                    value.get("status", "?"), 0
                ) + 1
                if value.get("private"):
                    private_count += 1
        print(f"  by status: {statuses}")
        if private_count:
            print(f"  private  : {private_count} (these are never written to disk)")

    return count


def main() -> int:
    root = Path(__file__).resolve().parents[1]  # repo root (this file lives in tools/)
    print("=" * 66)
    print("  MEMORY DIAGNOSTIC (read-only)")
    print("=" * 66)
    print(f"\nProject root: {root}")

    try:
        from ai_companion.config import ConfigManager, DEFAULT_CONFIG_PATH
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR importing ai_companion: {exc}")
        print("Run this from C:\\dev\\ai_companion with the venv Python.")
        return 1

    print(f"Config file : {DEFAULT_CONFIG_PATH}")
    config = ConfigManager().load()

    mode = config.vault.mode.value
    print(f"Vault mode  : {mode.upper()}")
    if mode == "private":
        print("  ^ IMPORTANT: in private mode NOTHING new is written to disk.")
        print("    Memories added now vanish when the app closes. Click the")
        print("    badge in the chat header so it reads SAVING.")

    total = 0
    for attr, label in (
        ("memory", "MEMORIES"),
        ("graph", "GRAPH NODES"),
        ("mind_map", "MIND MAP OVERVIEWS"),
    ):
        section = getattr(config, attr, None)
        if section is None:
            continue
        count = describe_store(Path(section.store_path), label)
        if attr == "memory":
            total = count

    conversations = Path(config.llm.conversations_path)
    describe_store(conversations, "CONVERSATIONS")

    # Backups and strays
    print("\n[BACKUPS AND STRAY FILES]")
    data_dir = Path(config.data_dir)
    found = False
    if data_dir.exists():
        for pattern in ("*.bak", "*.tmp", "*.json.part"):
            for item in data_dir.glob(pattern):
                print(f"  {item.name:28} {item.stat().st_size:>8} bytes  "
                      f"{stamp(item)}")
                found = True
    for other in root.rglob("memories.json"):
        if other != Path(config.memory.store_path):
            print(f"  STRAY STORE: {other}")
            found = True
    if not found:
        print("  none")

    print("\n" + "=" * 66)
    if total > 0:
        print(f"RESULT: {total} memories are present and readable.")
    elif total == 0:
        print("RESULT: the memory store is empty.")
        print("\nMost likely causes, in order:")
        print("  1. The vault was in PRIVATE mode when they were added.")
        print("     Private memories are never written to disk by design.")
        print("  2. They were added, but the app was killed before it saved")
        print("     (memories save immediately, so this is unlikely).")
        print("  3. A different config.json is in use - check the path above.")
    else:
        print("RESULT: the memory store is corrupt or wrongly shaped.")
        print("  Look for a .bak file above; renaming it over memories.json")
        print("  may recover them.")

    print("\nPaste this whole output back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
