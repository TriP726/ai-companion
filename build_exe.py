"""Build the Windows executable, with preflight checks.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe build_exe.py

WHY A SCRIPT RATHER THAN RUNNING PYINSTALLER DIRECTLY
-----------------------------------------------------
A PyInstaller run for this app takes 5-15 minutes. Most failures are things
that can be detected in two seconds beforehand: a missing package, the wrong
interpreter, no icon. This checks those first, then builds, then verifies the
result actually contains the data files the app needs at runtime.

The last part matters most. The classic failure here is a build that compiles
cleanly, launches, and crashes the first time you press the mic - because a
model file the speech engine opens by path was never bundled.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (import name, pip name, why it is needed)
REQUIRED = [
    ("PySide6", "PySide6", "the entire UI"),
    ("pydantic", "pydantic", "configuration models"),
    ("numpy", "numpy", "audio and graph maths"),
    ("networkx", "networkx", "knowledge graph"),
    ("PyInstaller", "pyinstaller", "this build"),
]

OPTIONAL = [
    ("llama_cpp", "llama-cpp-python", "local model inference"),
    ("faster_whisper", "faster-whisper", "speech to text"),
    ("piper", "piper-tts", "speech synthesis"),
    ("kokoro_onnx", "kokoro-onnx", "higher quality speech"),
    ("sounddevice", "sounddevice", "microphone and playback"),
    ("scipy", "scipy", "voice post-processing"),
    ("spellchecker", "pyspellchecker", "composer spell check"),
    ("cv2", "opencv-python", "camera"),
]

# Files that must exist in the finished build. These are the ones opened by
# path at runtime, which PyInstaller cannot infer.
EXPECTED_ARTIFACTS = [
    ("faster_whisper/assets", "Whisper voice-activity model"),
    ("spellchecker/resources", "spell check dictionaries"),
    # The one that produced WinError 3 on the first build. llama-cpp loads
    # these by path via ctypes, so a missing directory is invisible until the
    # user tries to load a model.
    ("llama_cpp/lib", "llama-cpp native libraries"),
]


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  - {detail}" if detail else ""))
    return ok


def preflight() -> bool:
    print("=" * 66)
    print("  PREFLIGHT")
    print("=" * 66 + "\n")

    ok = True

    in_venv = ".venv" in sys.executable or hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    ok &= check(
        "running inside the venv", in_venv,
        sys.executable if not in_venv else "",
    )
    if not in_venv:
        print("\n      Use: .venv\\Scripts\\python.exe build_exe.py")
        print("      Building with the wrong interpreter bundles the wrong")
        print("      packages, and the failure only appears at runtime.\n")

    ok &= check("spec file present", (ROOT / "ai_companion.spec").is_file())
    ok &= check("icon present", (ROOT / "assets" / "app.ico").is_file(),
                "run create_icon.py" if not
                (ROOT / "assets" / "app.ico").is_file() else "")
    ok &= check("entry point present",
                (ROOT / "ai_companion" / "main.py").is_file())

    print("\n  Required packages:")
    import importlib.util

    for module, package, why in REQUIRED:
        found = importlib.util.find_spec(module) is not None
        ok &= check(f"{package:22} ({why})", found,
                    "" if found else f"pip install {package}")

    print("\n  Optional packages (missing ones disable a feature):")
    for module, package, why in OPTIONAL:
        found = importlib.util.find_spec(module) is not None
        check(f"{package:22} ({why})", found,
              "" if found else "feature will be unavailable in the .exe")

    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    ok &= check(f"disk space ({free_gb:.1f} GB free)", free_gb > 5.0,
                "need at least 5 GB" if free_gb <= 5.0 else "")

    return ok


def verify(dist: Path) -> bool:
    """Check the build actually contains what the app opens at runtime."""
    print("\n" + "=" * 66)
    print("  VERIFYING THE BUILD")
    print("=" * 66 + "\n")

    exe = dist / "AICompanion.exe"
    ok = check("AICompanion.exe exists", exe.is_file())
    if exe.is_file():
        print(f"         {exe.stat().st_size / 1024 / 1024:.1f} MB")

    internal = dist / "_internal"
    search_root = internal if internal.is_dir() else dist

    for relative, why in EXPECTED_ARTIFACTS:
        target = search_root / relative
        found = target.exists()
        ok &= check(f"{relative:34} ({why})", found)

    # Qt platform plugin: without it the window never appears and there is no
    # error message at all.
    qt_platforms = list(search_root.rglob("qwindows.dll"))
    ok &= check("Qt Windows platform plugin", bool(qt_platforms))

    total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    print(f"\n  Total size: {total / 1024 / 1024:.0f} MB")

    return ok


def main() -> int:
    if not preflight():
        print("\nPreflight failed. Fix the items above and re-run.")
        return 1

    print("\n" + "=" * 66)
    print("  BUILDING  (5-15 minutes - this is normal)")
    print("=" * 66 + "\n")

    for stale in ("build", "dist"):
        path = ROOT / stale
        if path.exists():
            print(f"  removing stale {stale}/ ...")
            shutil.rmtree(path, ignore_errors=True)

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "ai_companion.spec",
         "--noconfirm", "--clean"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("\nBuild failed. The PyInstaller output above has the reason.")
        print("Most common causes:")
        print("  - a package installed outside the venv")
        print("  - antivirus locking files in build/ mid-write")
        return result.returncode

    dist = ROOT / "dist" / "AICompanion"
    if not verify(dist):
        print("\nThe build completed but is missing files it needs at")
        print("runtime. It will launch and then fail when you use the")
        print("affected feature. Report the FAIL lines above.")
        return 1

    # --- place user data beside the executable --------------------------
    #
    # The exe resolves config.json, models/ and data/ relative to its own
    # folder. Requiring a manual xcopy afterwards was a step easy to miss,
    # and missing it looks like a broken build ("no voices installed").
    print("\n" + "=" * 66)
    print("  COPYING YOUR DATA BESIDE THE EXE")
    print("=" * 66 + "\n")
    for name in ("config.json", "models", "data"):
        source = ROOT / name
        target = dist / name
        if not source.exists():
            print(f"  {name:12} not present in the project, skipping")
            continue
        if target.exists():
            print(f"  {name:12} already there, left alone")
            continue
        try:
            if source.is_dir():
                shutil.copytree(source, target)
                count = sum(1 for _ in target.rglob("*") if _.is_file())
                print(f"  {name:12} copied ({count} files)")
            else:
                shutil.copy2(source, target)
                print(f"  {name:12} copied")
        except Exception as exc:  # noqa: BLE001 - never fail the build here
            print(f"  {name:12} could not copy: {exc}")

    print("\n" + "=" * 66)
    print("  DONE")
    print("=" * 66)
    print(f"\n  {dist}\\AICompanion.exe")
    print("\n  Test it BEFORE deleting your venv:")
    print("    1. Launch it and check the model loads")
    print("    2. Press the mic badge and say something")
    print("    3. Open Settings and press Preview on a voice")
    print("\n  config.json, models\\ and data\\ now sit beside the exe.")
    print("  Move that whole folder anywhere; it is self-contained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
