"""Hands-free voice loop: wake word, listen, transcribe, speak.

THE LOOP
--------
    idle -> (wake word heard) -> capturing -> (silence) -> thinking
         -> speaking -> idle

Each transition emits a signal so the UI can show what is happening. A voice
interface with no visible state is indistinguishable from a broken one - the
user cannot tell "not listening" from "listening but did not hear you".

THE FEEDBACK PROBLEM
--------------------
The assistant's own voice comes out of the speakers and back into the
microphone. Without handling that, it transcribes itself and answers its own
replies forever. The listener is PAUSED for the whole duration of playback,
plus a short tail for room reverb.

PRIVACY
-------
Everything is local: faster-whisper and Piper both run offline against files
on disk. No audio leaves the machine. Recordings are never written to disk -
audio lives in memory for the length of one utterance and is discarded.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.services.speech_engine import (
    MicrophoneListener,
    Synthesiser,
    Transcriber,
    clean_for_speech,
    contains_wake_word,
    strip_wake_word,
)

# Extra silence after playback before listening resumes, to let room reverb
# die down rather than transcribing the tail of our own voice.
PLAYBACK_TAIL_SECONDS = 0.35


def _resolve(path_value: str) -> "Path":
    """Make a config path absolute against the app root.

    Config stores relative paths like "models/piper" so the project stays
    portable. Resolving them against the CURRENT WORKING DIRECTORY breaks the
    moment the app is launched from elsewhere - and a frozen exe is always
    launched from elsewhere, which is why the packaged build reported "no
    voices installed" while models\\piper sat right beside it.
    """
    from pathlib import Path

    from ai_companion.config import app_root

    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else app_root() / candidate


class VoiceState(str, Enum):
    OFF = "off"
    IDLE = "idle"           # listening for the wake word
    CAPTURING = "capturing"  # recording an utterance
    THINKING = "thinking"    # transcribing or generating
    SPEAKING = "speaking"


class VoiceService(BaseService):
    """Owns the microphone, the two engines, and the conversation loop."""

    service_name = "Voice"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._transcriber: Optional[Transcriber] = None
        self._synthesiser: Optional[Synthesiser] = None
        self._listener: Optional[MicrophoneListener] = None
        self._state = VoiceState.OFF
        self._awaiting_reply = False
        self._reply_buffer = ""
        self._spoke_last = False
        self._follow_up_until = 0.0
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------

    def start(self) -> None:
        super().start()
        speech = self._config.speech
        self._transcriber = Transcriber(
            getattr(speech, "whisper_model", "base.en")
        )
        self._synthesiser = self._build_synthesiser()
        # Engines load on first use, not here: 3.5s of Whisper load would be
        # 3.5s added to every app launch for a feature the user may not use.
        self.emit_status("Voice ready (engines load on first use)")

    def stop(self) -> None:
        self.disable()
        super().stop()

    def _build_synthesiser(self):
        """Pick the synthesiser named in config, falling back to Piper."""
        engine = getattr(self._config.speech, "tts_engine_name", "piper")
        if engine == "kokoro":
            from ai_companion.services.speech_engine import KokoroSynthesiser

            return KokoroSynthesiser(
                str(_resolve(
                    getattr(self._config.speech, "kokoro_dir",
                            "models/kokoro")
                )),
                getattr(self._config.speech, "kokoro_voice", "bm_george"),
            )
        return Synthesiser(self._voice_path())

    def _voice_path(self) -> str:
        speech = self._config.speech
        voice = getattr(speech, "piper_voice", "")
        folder = getattr(speech, "voices_dir", "models/piper")
        if not voice:
            return ""
        return str(_resolve(folder) / f"{voice}.onnx")

    # -- state --------------------------------------------------------

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def enabled(self) -> bool:
        return self._state != VoiceState.OFF

    def _set_state(self, state: VoiceState) -> None:
        if state == self._state:
            return
        self._state = state
        self._signal_bus.speech.state_changed.emit(state.value)

    # -- enable / disable ---------------------------------------------

    def enable(self) -> tuple[bool, str]:
        """Start hands-free listening."""
        if self.enabled:
            return True, ""

        ok, error = self._transcriber.load()
        if not ok:
            self.emit_error(error)
            return False, error

        ok, error = self._synthesiser.load()
        if not ok:
            # Speech-to-text alone is still useful, so this is a warning.
            self.emit_status(f"Voice input only - {error}")

        self._listener = MicrophoneListener(
            on_utterance=self._on_utterance,
            on_level=self._on_level,
            silence_threshold=getattr(
                self._config.speech, "silence_threshold", 0.012
            ),
            silence_seconds=getattr(
                self._config.speech, "silence_seconds", 1.0
            ),
            device_index=getattr(
                self._config.speech, "input_device_index", None
            ),
        )
        ok, error = self._listener.start()
        if not ok:
            self._listener = None
            self.emit_error(error)
            return False, error

        self._set_state(VoiceState.IDLE)
        wake = getattr(self._config.speech, "wake_word", "jarvis")
        self.emit_status(
            f"Listening for '{wake}'" if wake else "Listening"
        )
        return True, ""

    def disable(self) -> None:
        self._follow_up_until = 0.0
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._set_state(VoiceState.OFF)

    def toggle(self) -> tuple[bool, str]:
        if self.enabled:
            self.disable()
            return True, ""
        return self.enable()

    # -- microphone ---------------------------------------------------

    def _on_level(self, level: float) -> None:
        self._signal_bus.speech.level_changed.emit(float(level))

    def _on_utterance(self, audio: Any) -> None:
        """Called off-thread with a finished utterance."""
        if not self.enabled or self._state == VoiceState.SPEAKING:
            return

        self._set_state(VoiceState.THINKING)
        result = self._transcriber.transcribe(audio)
        if not result.ok:
            self.emit_error(result.error)
            self._set_state(VoiceState.IDLE)
            return

        heard = result.text.strip()
        if not heard:
            self._set_state(VoiceState.IDLE)
            return

        self._signal_bus.speech.transcribed.emit(heard)

        wake = getattr(self._config.speech, "wake_word", "jarvis")
        require_wake = bool(wake) and getattr(
            self._config.speech, "require_wake_word", True
        )

        # Follow-up window: for a short period after answering, treat speech
        # as a continuation. Demanding the wake word before every sentence
        # turns a conversation into a series of commands.
        if require_wake and time.time() < self._follow_up_until:
            require_wake = False

        if require_wake and not contains_wake_word(heard, wake):
            # Not addressed to us. Report it anyway: a user whose speech was
            # transcribed perfectly and then silently dropped has no way to
            # tell a missing wake word from a broken app. Silence here is
            # what made this look like a bug.
            self._signal_bus.speech.utterance_ignored.emit(
                heard,
                f'no wake word - start with "{wake}"',
            )
            self._set_state(VoiceState.IDLE)
            return

        # Any accepted utterance refreshes the window.
        self._follow_up_until = 0.0
        prompt = strip_wake_word(heard, wake) if wake else heard
        if not prompt.strip():
            self._set_state(VoiceState.IDLE)
            return

        self._send_to_model(prompt)

    # -- model --------------------------------------------------------

    def _send_to_model(self, prompt: str) -> None:
        manager = getattr(self, "_service_manager", None)
        llm = manager.get("LLM") if manager is not None else None
        if llm is None or not getattr(llm, "model_loaded", False):
            self.speak("No model is loaded.")
            self._set_state(VoiceState.IDLE)
            return

        conversation_id = getattr(llm, "_active_conversation_id", "")
        if not conversation_id:
            conversation_id = llm.create_conversation()

        self._awaiting_reply = True
        self._reply_buffer = ""
        self._signal_bus.speech.prompt_ready.emit(conversation_id, prompt)

    def on_reply_complete(self, text: str) -> None:
        """Called by the UI when the model finishes a spoken-origin turn."""
        if not self._awaiting_reply:
            return
        self._awaiting_reply = False
        if text.strip():
            self.speak(text)
        else:
            self._set_state(VoiceState.IDLE)

    # -- speaking -----------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesise and play, with the microphone paused throughout."""
        if self._synthesiser is None:
            return
        threading.Thread(
            target=self._speak_blocking, args=(text,), daemon=True
        ).start()

    def _speak_blocking(self, text: str) -> None:
        spoken = clean_for_speech(text)
        if not spoken:
            self._set_state(VoiceState.IDLE)
            return

        self._set_state(VoiceState.SPEAKING)
        if self._listener is not None:
            self._listener.pause()

        try:
            result = self._synthesiser.synthesise(spoken)
            if not result.ok:
                self.emit_error(result.error)
                return
            self._play(result.wav_bytes)
        finally:
            # Always resume, even if playback raised - otherwise one bad
            # synthesis permanently deafens the assistant.
            time.sleep(PLAYBACK_TAIL_SECONDS)
            if self._listener is not None:
                self._listener.resume()
            window = float(
                getattr(self._config.speech, "follow_up_seconds", 12.0)
            )
            self._follow_up_until = time.time() + window if window > 0 else 0.0
            if self.enabled:
                self._set_state(VoiceState.IDLE)

    def _play(self, wav_bytes: bytes) -> None:
        import io
        import wave

        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            self.emit_error("sounddevice not installed - cannot play audio")
            return
        except OSError as exc:
            # PortAudio missing: a real failure mode on a fresh machine.
            self.emit_error(f"Audio system unavailable: {exc}")
            return

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio /= 32768.0

            # Production pass: this is what turns a competent TTS voice into
            # something that sounds like it is speaking from the room.
            preset = getattr(self._config.speech, "voice_fx", "jarvis")
            if preset and preset != "off":
                from ai_companion.services.voice_fx import (
                    apply_voice_fx,
                    settings_for,
                )

                audio = apply_voice_fx(audio, rate, settings_for(preset))

            sd.play(audio, rate)
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            self.emit_error(f"Playback failed: {exc}")

    # -- one-shot helpers ---------------------------------------------

    def available_voices(self) -> list[str]:
        """Voice names found on disk."""
        from pathlib import Path

        folder = _resolve(
            getattr(self._config.speech, "voices_dir", "models/piper")
        )
        if not folder.exists():
            return []
        return sorted(p.stem for p in folder.glob("*.onnx"))

    def set_voice(self, name: str) -> tuple[bool, str]:
        """Switch voice at runtime."""
        self._config.speech.piper_voice = name
        self._synthesiser = self._build_synthesiser()
        return self._synthesiser.load()

    def preview_voice(self, name: str = "") -> None:
        """Speak a sample line so the user can choose by ear."""
        if name:
            ok, error = self.set_voice(name)
            if not ok:
                self.emit_error(error)
                return
        self.speak(
            "All systems are online and operating within normal parameters."
        )
