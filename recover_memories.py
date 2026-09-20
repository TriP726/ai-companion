"""Find and restore memory stores from anywhere on disk.

`repair_config.py` only fixed the POINTER in config.json. If the file it now
points at is itself empty, that is not enough - the records may be sitting in
a different file entirely (a pytest temp folder that still exists, a stale
copy, or a .bak).

This script searches, shows you every store it finds with its record count,
and offers to restore the best one.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe recover_memories.py

Nothing is overwritten without a backup and an explicit yes.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# Files that hold app records, with the config field that points at each.
STORE_NAMES = (
    "memories.json",
    "conversations.json",
    "graph.json",
    "graph_edges.json",
    "mind_map.json",
)


def record_count(path: Path) -> int:
    """Records in a JsonStore file. -1 if unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return -1
    if not isinstance(data, dict):
        return -1
    items = data.get("items")
    if isinstance(items, dict):
        return len(items)
    stray = [k for k in data if k != "_meta"]
    return len(stray) if stray else 0


def search_roots() -> list[Path]:
    """Places worth searching, deepest-value first."""
    roots: list[Path] = []
    project = Path(__file__).resolve().parent
    roots.append(project)

    home = Path.home()
    for candidate in (
        home / "AppData" / "Local" / "Temp",
        Path(os.environ.get("TEMP", "")) if os.environ.get("TEMP") else None,
        Path("/tmp"),
        home,
    ):
        if candidate and candidate.exists() and candidate not in roots:
            roots.append(candidate)
    return roots


# Paths under a pytest temp tree contain fixture data, not the user's. Ranking
# those above the real store would "restore" synthetic test records over
# genuine memories - worse than finding nothing.
TEST_MARKERS = ("pytest-of-", "pytest-", "/test_", "\\test_")


def looks_like_test_data(path: Path) -> bool:
    lowered = str(path).lower().replace("\\", "/")
    return any(marker.replace("\\", "/") in lowered for marker in TEST_MARKERS)


def find_stores(name: str) -> list[tuple[int, Path, float]]:
    """Every file called `name`, as (records, path, mtime).

    Real user data sorts above pytest fixture data regardless of record count.
    """
    seen: set[Path] = set()
    found: list[tuple[int, Path, float]] = []
    for root in search_roots():
        try:
            # Temp trees can be enormous; cap the walk depth by pattern.
            candidates = list(root.rglob(name))
        except (OSError, PermissionError):
            continue
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                found.append(
                    (record_count(resolved), resolved,
                     resolved.stat().st_mtime)
                )
            except OSError:
                continue
    return sorted(
        found,
        key=lambda item: (
            looks_like_test_data(item[1]),   # real data first
            -item[0],                        # then most records
            -item[2],                        # then most recent
        ),
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    print("=" * 66)
    print("  MEMORY RECOVERY")
    print("=" * 66)
    print(f"\nProject root: {root}")
    print("Searching (this can take a minute on a large Temp folder)...\n")

    try:
        from ai_companion.config import ConfigManager

        config = ConfigManager().load()
        targets = {
            "memories.json": Path(config.memory.store_path),
            "conversations.json": Path(config.llm.conversations_path),
            "graph.json": Path(config.graph.store_path),
        }
        mind_map = getattr(config, "mind_map", None)
        if mind_map is not None:
            targets["mind_map.json"] = Path(mind_map.store_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read config: {exc}")
        targets = {"memories.json": root / "data" / "memories.json"}

    restored_any = False

    for name in STORE_NAMES:
        target = targets.get(name)
        if target is None:
            continue

        found = find_stores(name)
        if not found:
            continue

        print(f"\n[{name}]")
        print(f"  app reads from: {target}")
        for count, path, mtime in found[:8]:
            when = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            marker = ""
            try:
                if path.resolve() == target.resolve():
                    marker = "  <- currently in use"
            except OSError:
                pass
            state = f"{count} records" if count >= 0 else "unreadable"
            if looks_like_test_data(path):
                marker += "  [test data - ignored]"
            print(f"    {state:14} {when}  {path}{marker}")

        real = [f for f in found if not looks_like_test_data(f[1])]
        if not real:
            print("  Only test-fixture copies found - not offering those.")
            continue
        best_count, best_path, _ = real[0]
        try:
            current = record_count(target) if target.exists() else 0
        except Exception:  # noqa: BLE001
            current = 0

        if best_count <= 0:
            print("  Nothing with records found.")
            continue
        if current >= best_count:
            print(f"  Current file already has {current} records. Leaving it.")
            continue

        print(f"\n  Best candidate: {best_count} records")
        print(f"    {best_path}")
        answer = input(
            f"  Restore this over the current {name}? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("  Skipped.")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            backup = target.with_suffix(
                f".json.replaced-{datetime.now():%Y%m%d-%H%M%S}"
            )
            shutil.copy2(target, backup)
            print(f"  Backed up the old file to {backup.name}")
        shutil.copy2(best_path, target)
        print(f"  Restored {best_count} records.")
        restored_any = True

    print("\n" + "=" * 66)
    if restored_any:
        print("Done. Start the app with run.bat and check the MEM tab.")
    else:
        print("Nothing was restored.")
        print("\nIf your memories are genuinely gone, the likely cause is that")
        print("they were added while the vault was in PRIVATE mode - those are")
        print("never written to disk by design, and cannot be recovered.")
        print("\nCheck the badge in the chat header reads SAVING before adding")
        print("memories you want to keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
