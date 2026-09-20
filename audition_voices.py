"""Render every voice x character combination to WAV files so you can compare.

I cannot hear audio, so I cannot tell you which of these sounds most like
JARVIS to you. This writes them all to disk and you decide.

Run from the project root with the venv Python:

    .venv\\Scripts\\python.exe audition_voices.py

Files land in voice_samples\\ - play them, pick a favourite, then set the
voice and character in CONFIG -> Speech.
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

LINE = (
    "Good evening, sir. All systems are online and operating within normal "
    "parameters. Shall I begin the diagnostic?"
)


def main() -> int:
    root = Path(__file__).resolve().parent
    voices_dir = root / "models" / "piper"
    out_dir = root / "voice_samples"

    print("=" * 66)
    print("  VOICE AUDITION")
    print("=" * 66)

    if not voices_dir.is_dir():
        print(f"\nNo voices found at {voices_dir}")
        print("Run get_speech.py first.")
        return 1

    voices = sorted(voices_dir.glob("*.onnx"))
    if not voices:
        print(f"\nNo .onnx voices in {voices_dir}")
        print("Run get_speech.py first.")
        return 1

    try:
        import numpy as np
        from piper import PiperVoice

        from ai_companion.services.voice_fx import (
            apply_voice_fx,
            describe_presets,
            settings_for,
        )
    except ImportError as exc:
        print(f"\nMissing dependency: {exc}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    presets = [name for name, _ in describe_presets()]

    print(f"\nVoices    : {len(voices)}")
    print(f"Characters: {', '.join(presets)}")
    print(f"Output    : {out_dir}\n")

    written = 0

    # --- Kokoro first: it is the one most likely to fix "sounds robotic" ---
    kokoro_dir = root / "models" / "kokoro"
    if (kokoro_dir / "kokoro-v1.0.onnx").exists():
        try:
            from kokoro_onnx import Kokoro

            from ai_companion.services.speech_engine import KOKORO_VOICES

            print("kokoro (slower, generally more natural)")
            engine = Kokoro(
                str(kokoro_dir / "kokoro-v1.0.onnx"),
                str(kokoro_dir / "voices-v1.0.bin"),
            )
            for name, description in KOKORO_VOICES:
                language = "en-gb" if name.startswith("b") else "en-us"
                samples, rate = engine.create(
                    LINE, voice=name, speed=1.0, lang=language
                )
                data = np.clip(np.asarray(samples, dtype=np.float32), -1, 1)
                target = out_dir / f"kokoro__{name}.wav"
                with wave.open(str(target), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(int(rate))
                    handle.writeframes((data * 32767).astype(np.int16).tobytes())
                print(f"   {name:12} {description:24} -> {target.name}")
                written += 1
        except Exception as exc:  # noqa: BLE001
            print(f"   kokoro unavailable: {exc}")
    else:
        print("kokoro not installed - run get_speech.py to add it\n")

    print()
    for voice_path in voices:
        print(f"{voice_path.stem}")
        try:
            voice = PiperVoice.load(str(voice_path))
        except Exception as exc:  # noqa: BLE001
            print(f"   could not load: {exc}")
            continue

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            voice.synthesize_wav(LINE, handle)
        buffer.seek(0)
        with wave.open(buffer, "rb") as handle:
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        dry = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        for preset in presets:
            processed = (
                dry if preset == "off"
                else apply_voice_fx(dry, rate, settings_for(preset))
            )
            target = out_dir / f"{voice_path.stem}__{preset}.wav"
            data = np.clip(processed, -1.0, 1.0)
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes((data * 32767).astype(np.int16).tobytes())
            print(f"   {preset:9} -> {target.name}")
            written += 1

    print(f"\n{written} samples written to {out_dir}")
    print("\nStart with the kokoro__ files - they are the most likely answer")
    print("to 'it sounds robotic'. The piper files are faster to generate.")
    print("\nPlay them, pick the one you like, then set both the Voice and the")
    print("Voice character in CONFIG -> Speech.")
    print("\nIf none sound right, the honest options are in the patch notes:")
    print("a different Piper voice, or a cloning model your CPU cannot run in")
    print("real time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
