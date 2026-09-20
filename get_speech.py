"""Download the speech models: Whisper (listening) and Piper (speaking).

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe get_speech.py

Downloads are explicit and resumable. Nothing is fetched automatically by the
app itself - that would break the no-automatic-downloads rule this project was
built under.

Total: roughly 120 MB depending on the voices you pick.
"""
from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
PIPER_BASE = f"{HF}/rhasspy/piper-voices/resolve/main/en"

# Whisper: faster-whisper pulls its own weights from HF on first load, so the
# only thing to do here is warm the cache so the first press of the mic is not
# a surprise download.
WHISPER_SIZES = {
    "1": ("tiny.en", "~75 MB", "Fastest. Noticeably more mistakes."),
    "2": ("base.en", "~145 MB", "Recommended. Good accuracy, ~15x realtime."),
    "3": ("small.en", "~480 MB", "Best accuracy, ~6x realtime on CPU."),
}

# Piper voices. All British male except where noted - closest register to the
# JARVIS sound without pretending we can clone the actor.
VOICES = {
    "1": {
        "id": "en_GB-alan-medium",
        "path": "en_GB/alan/medium/en_GB-alan-medium",
        "mb": 63,
        "note": "British male, measured and neutral. Closest to JARVIS.",
    },
    "2": {
        "id": "en_GB-northern_english_male-medium",
        "path": "en_GB/northern_english_male/medium/"
                "en_GB-northern_english_male-medium",
        "mb": 63,
        "note": "British male, warmer and more characterful.",
    },
    "3": {
        "id": "en_GB-cori-high",
        "path": "en_GB/cori/high/en_GB-cori-high",
        "mb": 110,
        "note": "British female, high quality.",
    },
    "4": {
        "id": "en_US-ryan-high",
        "path": "en_US/ryan/high/en_US-ryan-high",
        "mb": 110,
        "note": "American male, high quality.",
    },
}


def human(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def download(url: str, dest: Path) -> bool:
    """Resumable download with a progress bar."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "ai-companion"})
    if existing:
        print(f"   resuming from {human(existing)}")
        req.add_header("Range", f"bytes={existing}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + existing
            mode = "ab" if existing else "wb"
            done = existing
            with open(tmp, mode) as fh:
                while True:
                    chunk = resp.read(1024 * 128)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        bar = "#" * int(pct / 2.5)
                        print(
                            f"\r   [{bar:<40}] {pct:5.1f}%  "
                            f"{human(done)} / {human(total)}",
                            end="", flush=True,
                        )
            print()
    except KeyboardInterrupt:
        print("\n   Interrupted. Re-run to resume.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"\n   Failed: {exc}")
        return False

    tmp.replace(dest)
    return True


def main() -> int:
    root = Path(__file__).resolve().parent
    voices_dir = root / "models" / "piper"
    voices_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  SPEECH MODELS")
    print("=" * 66)

    # ---- Piper voices ----
    print("\nVOICES (Piper - what the assistant sounds like)\n")
    for key, voice in VOICES.items():
        installed = (voices_dir / f"{voice['id']}.onnx").exists()
        mark = " [installed]" if installed else ""
        print(f"  [{key}] {voice['id']}{mark}")
        print(f"      {voice['mb']} MB - {voice['note']}\n")
    print("  [a] all of them   [s] skip\n")

    choice = input("Which voice(s)? ").strip().lower()
    wanted = []
    if choice == "a":
        wanted = list(VOICES.values())
    elif choice != "s":
        for ch in choice.replace(",", " ").split():
            if ch in VOICES:
                wanted.append(VOICES[ch])

    for voice in wanted:
        onnx = voices_dir / f"{voice['id']}.onnx"
        cfg = voices_dir / f"{voice['id']}.onnx.json"
        if onnx.exists() and cfg.exists():
            print(f"\n{voice['id']}: already present")
            continue
        print(f"\nDownloading {voice['id']} ...")
        free = shutil.disk_usage(voices_dir).free
        if free < voice["mb"] * 1024 * 1024 * 2:
            print("   Not enough free disk space.")
            continue
        if not onnx.exists():
            if not download(f"{PIPER_BASE}/{voice['path']}.onnx", onnx):
                continue
        if not cfg.exists():
            download(f"{PIPER_BASE}/{voice['path']}.onnx.json", cfg)

    # ---- Whisper ----
    print("\n" + "=" * 66)
    print("\nLISTENING (Whisper - what turns your speech into text)\n")
    for key, (name, size, note) in WHISPER_SIZES.items():
        print(f"  [{key}] {name:10} {size:10} {note}")
    print("  [s] skip\n")

    pick = input("Which model? [2] ").strip() or "2"
    if pick != "s" and pick in WHISPER_SIZES:
        name = WHISPER_SIZES[pick][0]
        print(f"\nFetching {name} (cached by faster-whisper)...")
        try:
            from faster_whisper import WhisperModel

            WhisperModel(name, device="cpu", compute_type="int8")
            print("   done.")
        except Exception as exc:  # noqa: BLE001
            print(f"   Failed: {exc}")
            print("   Install with: pip install faster-whisper")

    # ---- Kokoro ----
    print("\n" + "=" * 66)
    print("\nKOKORO (slower engine, generally judged more natural)\n")
    print("  ~340 MB. Runs at about 1.2x realtime on CPU, versus Piper's 10x.")
    print("  Worth it if Piper sounds too flat to you.\n")
    if input("Download Kokoro? [y/N] ").strip().lower() in ("y", "yes"):
        kokoro_dir = root / "models" / "kokoro"
        kokoro_dir.mkdir(parents=True, exist_ok=True)
        base = ("https://github.com/thewh1teagle/kokoro-onnx/releases/"
                "download/model-files-v1.0")
        for filename in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            dest = kokoro_dir / filename
            if dest.exists():
                print(f"\n{filename}: already present")
                continue
            print(f"\nDownloading {filename} ...")
            download(f"{base}/{filename}", dest)

    # ---- record the choice ----
    installed = sorted(p.stem for p in voices_dir.glob("*.onnx"))
    print("\n" + "=" * 66)
    print("Voices installed:", ", ".join(installed) or "none")
    if installed:
        cfg_path = root / "config.json"
        if cfg_path.exists():
            answer = input(
                f"\nUse '{installed[0]}' as the assistant voice? [Y/n] "
            ).strip().lower()
            if answer in ("", "y", "yes"):
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                speech = data.setdefault("speech", {})
                speech["piper_voice"] = installed[0]
                speech["voices_dir"] = str(voices_dir).replace("\\", "/")
                if pick in WHISPER_SIZES:
                    speech["whisper_model"] = WHISPER_SIZES[pick][0]
                cfg_path.write_text(json.dumps(data, indent=2), "utf-8")
                print("   config.json updated.")

    print("\nDone. Start the app and press the mic button in the chat header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
