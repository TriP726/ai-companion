# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the AI Companion Windows executable.

WHY THIS IS NOT A SIMPLE SPEC
-----------------------------
Five of this app's dependencies ship data files that PyInstaller's static
analysis cannot see, because they are opened at runtime by path rather than
imported:

    faster_whisper   assets/silero_vad_v6.onnx     (VAD model)
    piper            espeakbridge + tashkeel ONNX  (phonemiser)
    espeakng_loader  libespeak-ng native library
    kokoro_onnx      config.json
    spellchecker     resources/*.json.gz           (dictionaries)

Miss any of them and you get a build that compiles cleanly, launches, and
then crashes the first time you press the mic. That failure mode is the whole
reason this file collects them programmatically with collect_data_files
rather than hardcoding paths - the layout differs between platforms and
package versions.

The previous version of this spec predated the speech work entirely. It
listed 14 hidden imports, missed nine real dependencies, and actively
EXCLUDED scipy, which voice_fx now requires for its EQ and reverb.

SIZE
----
Expect 700 MB - 1.2 GB. PySide6, onnxruntime, ctranslate2 and OpenCV are each
large. This is normal for a frozen Qt + ML application and is not a sign
something has gone wrong.

BUILD
-----
    .venv\\Scripts\\python.exe build_exe.py

Do not run pyinstaller directly unless you know why - build_exe.py performs
preflight checks that catch most failures before a 10 minute build.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# --- data files that must travel with the executable -------------------
#
# collect_data_files walks the installed package, so this keeps working when
# a dependency reorganises its internals.
datas = [
    ("assets/app.ico", "assets"),
]

for package in (
    "faster_whisper",    # silero VAD model
    "piper",             # espeak bridge, tashkeel models
    "kokoro_onnx",       # config.json
    "spellchecker",      # dictionary .json.gz files
    "espeakng_loader",   # native espeak-ng library
    "language_tags",     # used by piper for locale handling
):
    try:
        datas += collect_data_files(package)
    except Exception:
        # A missing optional package is not fatal: the app degrades to
        # "feature unavailable" rather than failing to build.
        pass

# llama-cpp-python ships PREBUILT DLLs in llama_cpp/lib, loaded by path at
# runtime via ctypes. Neither collect_data_files nor collect_dynamic_libs
# finds them reliably, because they are not imported and not all are .dll on
# every platform. Without this the exe launches fine and then fails with:
#
#   [WinError 3] cannot find the path specified:
#   '...\\_internal\\llama_cpp\\lib'
#
# Copy the whole directory verbatim, preserving the layout ctypes expects.
try:
    import llama_cpp
    from pathlib import Path as _Path

    _lib = _Path(llama_cpp.__file__).parent / "lib"
    if _lib.is_dir():
        for _f in _lib.iterdir():
            if _f.is_file():
                datas.append((str(_f), "llama_cpp/lib"))
except Exception:
    pass

# --- native libraries --------------------------------------------------
binaries = []
for package in ("onnxruntime", "ctranslate2", "espeakng_loader", "piper"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

# --- imports PyInstaller cannot infer ----------------------------------
hiddenimports = [
    # Qt
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "shiboken6",
    # Core
    "pydantic",
    "pydantic.deprecated.decorator",
    "numpy",
    "networkx",
    # Model
    "llama_cpp",
    # Speech - these are imported lazily inside functions, so static
    # analysis never sees them.
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "piper",
    "piper.voice",
    "kokoro_onnx",
    "sounddevice",
    "soundfile",
    "espeakng_loader",
    # Voice post-processing. The old spec EXCLUDED scipy; voice_fx needs it.
    "scipy",
    "scipy.signal",
    "scipy.special",
    "scipy.special._cdflib",
    # Spell check
    "spellchecker",
    # Camera and images
    "cv2",
    "PIL",
    "PIL._tkinter_finder",
]

a = Analysis(
    ["ai_companion/main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Genuinely unused, and each is large.
        "tkinter",
        "matplotlib",
        "pandas",
        "notebook",
        "jupyter",
        "IPython",
        "pytest",
        "sphinx",
        # NOTE: scipy is deliberately NOT excluded. It was in the old spec
        # and would break voice post-processing at runtime.
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AICompanion",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is off on purpose. It frequently corrupts Qt and ONNX DLLs, and the
    # failure appears as a crash on launch rather than a build error - an
    # expensive thing to debug for a modest size saving.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/app.ico",
)

# One-folder rather than one-file. A one-file build unpacks ~1 GB to a temp
# directory on every launch, which takes 10-20 seconds and writes gigabytes
# to the disk repeatedly. One-folder starts in about two seconds.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AICompanion",
)
