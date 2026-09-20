"""Check whether the built exe actually contains the current fixes.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe tools/verify_build.py

Answers one question: is the exe you are running built from the patched
source, or is it a stale binary from an earlier build?
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (this file lives in tools/)


def stamp(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except OSError:
        return "?"


def main() -> int:
    print("=" * 66)
    print("  BUILD VERIFICATION")
    print("=" * 66)

    # --- 1. is the SOURCE patched? ---
    print("\n[1] Source code")
    config = ROOT / "ai_companion" / "config.py"
    spec = ROOT / "ai_companion.spec"

    has_app_root = config.is_file() and "def app_root" in config.read_text(
        encoding="utf-8"
    )
    has_llama = spec.is_file() and "llama_cpp/lib" in spec.read_text(
        encoding="utf-8"
    )
    print(f"    config.py has app_root()   : {has_app_root}")
    print(f"    spec bundles llama_cpp/lib : {has_llama}")
    if config.is_file():
        print(f"    config.py modified         : {stamp(config)}")

    if not (has_app_root and has_llama):
        print("\n    The source tree is out of date. Update it (git pull)")
        print("    before building.")
        return 1

    # --- 2. does a build exist, and is it newer than the source? ---
    print("\n[2] Built executable")
    dist = ROOT / "dist" / "AICompanion"
    exe = dist / "AICompanion.exe"

    if not exe.is_file():
        print("    No exe found. Run:")
        print("        .venv\\Scripts\\python.exe tools/build_exe.py")
        return 1

    print(f"    exe built    : {stamp(exe)}")
    print(f"    source edited: {stamp(config)}")

    if exe.stat().st_mtime < config.stat().st_mtime:
        print("\n    STALE: the exe is OLDER than the patched source.")
        print("    You are running a binary built before the fixes.")
        print("    Rebuild:")
        print("        .venv\\Scripts\\python.exe tools/build_exe.py")
        return 1
    print("    exe is newer than the source (good)")

    # --- 3. does the BUILD contain the llama DLLs? ---
    print("\n[3] Bundled llama-cpp libraries")
    internal = dist / "_internal"
    search = internal if internal.is_dir() else dist
    lib = search / "llama_cpp" / "lib"
    if lib.is_dir():
        files = [f.name for f in lib.iterdir() if f.is_file()]
        print(f"    {lib}")
        print(f"    {len(files)} file(s): {', '.join(files[:5])}")
    else:
        print(f"    MISSING: {lib}")
        print("    This is the WinError 3 you are seeing. Rebuild.")
        return 1

    # --- 4. is the user data beside the exe? ---
    print("\n[4] Data beside the executable")
    problems = 0
    for name, why in (
        ("config.json", "settings and model path"),
        ("models", "Piper/Kokoro voices"),
        ("data", "memories and conversations"),
    ):
        target = dist / name
        exists = target.exists()
        print(f"    {name:12} {'present' if exists else 'MISSING'}  ({why})")
        if not exists:
            problems += 1

    if problems:
        print("\n    Copy them across:")
        print("        copy config.json    dist\\AICompanion\\")
        print("        xcopy /E /I models  dist\\AICompanion\\models")
        print("        xcopy /E /I data    dist\\AICompanion\\data")
        return 1

    voices = list((dist / "models" / "piper").glob("*.onnx")) \
        if (dist / "models" / "piper").is_dir() else []
    print(f"    piper voices : {len(voices)}")

    print("\n" + "=" * 66)
    print("Everything checks out. Launch dist\\AICompanion\\AICompanion.exe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
