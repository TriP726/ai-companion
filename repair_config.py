"""Repair a config.json whose store paths point into a pytest temp folder.

WHAT WENT WRONG
---------------
An older build let the test suite persist its own relocated config over the
real one. Tests run against a throwaway directory, so every store path was
rewritten to something like:

    C:\\Users\\you\\AppData\\Local\\Temp\\pytest-of-you\\pytest-27\\
        test_only_active_chat_affected0\\data\\memories.json

Windows deletes those folders. The app then looked for memories somewhere
that no longer exists, found nothing, and showed an empty Memory tab.

**Your memories were never deleted.** They are still in
`C:\\dev\\ai_companion\\data\\memories.json`. This script points the config
back at them.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe repair_config.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

# (section, field, correct relative default)
STORE_FIELDS = (
    ("llm", "conversations_path", "data/conversations.json"),
    ("memory", "store_path", "data/memories.json"),
    ("graph", "store_path", "data/graph.json"),
    ("mind_map", "store_path", "data/mind_map.json"),
    ("vault", "vault_root", "vault"),
    ("code_workspace", "sandbox_dir", "sandbox"),
    ("code_workspace", "snapshot_dir", "sandbox/.snapshots"),
    ("image_gen", "output_dir", "generated_images"),
    ("speech", "voices_dir", "models/piper"),
)

TEMP_MARKERS = ("pytest-of-", "pytest-", "/tmp/", "\\temp\\", "appdata\\local\\temp")


def looks_like_temp(value: str) -> bool:
    lowered = str(value).lower().replace("\\", "/")
    return any(marker.replace("\\", "/") in lowered for marker in TEMP_MARKERS)


def main() -> int:
    root = Path(__file__).resolve().parent
    config_path = root / "config.json"

    print("=" * 66)
    print("  CONFIG REPAIR")
    print("=" * 66)
    print(f"\nProject root : {root}")
    print(f"Config file  : {config_path}")

    if not config_path.exists():
        print("\nNo config.json here. Nothing to repair.")
        return 0

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"\nconfig.json is unreadable: {exc}")
        print("Delete it and the app will write a fresh one.")
        return 1

    # ---- find the damage ----
    broken: list[tuple[str, str, str, str]] = []
    if looks_like_temp(str(data.get("data_dir", ""))):
        broken.append(("", "data_dir", str(data["data_dir"]), "data"))

    for section, field, default in STORE_FIELDS:
        value = data.get(section, {}).get(field)
        if value and looks_like_temp(str(value)):
            broken.append((section, field, str(value), default))

    if not broken:
        print("\nAll store paths look correct. Nothing to repair.")
        print("\nCurrent paths:")
        for section, field, _default in STORE_FIELDS:
            value = data.get(section, {}).get(field)
            if value:
                print(f"  {section}.{field:20} {value}")
        return 0

    print(f"\nFound {len(broken)} path(s) pointing into a temp folder:\n")
    for section, field, value, default in broken:
        label = f"{section}.{field}" if section else field
        print(f"  {label}")
        print(f"     now : {value}")
        print(f"     ->  : {default}")

    # ---- check the real data is there ----
    print("\nYour actual data:")
    recovered = 0
    for name, relative in (
        ("memories", "data/memories.json"),
        ("conversations", "data/conversations.json"),
        ("graph", "data/graph.json"),
    ):
        path = root / relative
        if not path.exists():
            print(f"  {name:14} not present")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            count = len(payload.get("items", {}))
        except Exception:  # noqa: BLE001
            count = -1
        if count >= 0:
            print(f"  {name:14} {count} record(s)  ({path})")
            if name == "memories":
                recovered = count
        else:
            print(f"  {name:14} unreadable ({path})")

    answer = input("\nRepair config.json now? [Y/n] ").strip().lower()
    if answer not in ("", "y", "yes"):
        print("No changes made.")
        return 0

    backup = config_path.with_suffix(
        f".json.broken-{datetime.now():%Y%m%d-%H%M%S}"
    )
    shutil.copy2(config_path, backup)
    print(f"\nBacked up the old config to {backup.name}")

    if "data_dir" in data and looks_like_temp(str(data["data_dir"])):
        data["data_dir"] = "data"
    for section, field, default in STORE_FIELDS:
        value = data.get(section, {}).get(field)
        if value and looks_like_temp(str(value)):
            data.setdefault(section, {})[field] = default

    tmp = config_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(config_path)
    print("config.json repaired.")

    if recovered:
        print(f"\n{recovered} memories should reappear when you start the app.")
    print("\nStart with run.bat and check the MEM tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
