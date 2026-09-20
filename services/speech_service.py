"""Speech service — local STT (Whisper) and TTS."""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService


class SpeechService(BaseService):
    """Handles speech input (Whisper) and text-to-speech.

    All processing is local — no cloud APIs.
    """

    service_name = "Speech"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._whisper_model: Any = None
        self._tts_engine: Any = None
        self._listening = False
        self._recording_thread: Optional[threading.Thread] = None

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start(self) -> None:
        super().start()
        self._init_tts()

    def stop(self) -> None:
        self.stop_listening()
        super().stop()

    def _init_tts(self) -> None:
        """Initialize the TTS engine."""
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty(
                "rate", self._config.speech.tts_rate
            )
        except ImportError:
            self.emit_error("pyttsx3 not installed — TTS unavailable")
        except Exception as e:
            self.emit_error(f"TTS init failed: {e}")

    def _load_whisper(self) -> bool:
        """Lazy-load the Whisper model."""
        if self._whisper_model is not None:
            return True
        try:
            import whisper
            self._whisper_model = whisper.load_model(
                self._config.speech.whisper_model
            )
            return True
        except ImportError:
            self.emit_error("openai-whisper not installed — STT unavailable")
            return False
        except Exception as e:
            self.emit_error(f"Whisper load failed: {e}")
            return False

    def start_listening(self) -> bool:
        """Start recording audio for speech-to-text."""
        if self._listening:
            return True

        if not self._load_whisper():
            return False

        self._listening = True
        self._signal_bus.speech.listening_started.emit()
        self._recording_thread = threading.Thread(
            target=self._recording_worker, daemon=True
        )
        self._recording_thread.start()
        return True

    def stop_listening(self) -> None:
        """Stop recording and transcribe."""
        self._listening = False
        self._signal_bus.speech.listening_stopped.emit()

    def _recording_worker(self) -> None:
        """Record audio and transcribe when stopped."""
        try:
            import numpy as np

            # Record audio using sounddevice or pyaudio
            audio_data = self._record_audio()
            if audio_data is None:
                return

            # Transcribe with Whisper
            result = self._whisper_model.transcribe(audio_data)
            text = result.get("text", "").strip()
            if text:
                self._signal_bus.speech.transcription_complete.emit(text)

        except Exception as e:
            self.emit_error(f"Recording/transcription error: {e}")
        finally:
            self._listening = False

    def _record_audio(self) -> Optional[Any]:
        """Record audio until listening is stopped.

        Returns audio data as numpy array, or None on failure.
        """
        try:
            import numpy as np
            import sounddevice as sd

            sample_rate = 16000
            frames: list = []

            def callback(indata, frame_count, time_info, status):
                if self._listening:
                    frames.append(indata.copy())

            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=callback,
                device=self._config.speech.input_device_index,
            ):
                while self._listening:
                    time.sleep(0.1)

            if frames:
                audio = np.concatenate(frames, axis=0)
                return audio.flatten()
            return None

        except ImportError:
            self.emit_error("sounddevice not installed")
            return None

    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        if not self._tts_engine:
            self.emit_error("TTS engine not available")
            return

        def _speak_worker():
            try:
                self._signal_bus.speech.tts_started.emit()
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                self._signal_bus.speech.tts_finished.emit()
            except Exception as e:
                self._signal_bus.speech.tts_error.emit(str(e))

        threading.Thread(target=_speak_worker, daemon=True).start()

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """Transcribe an audio file."""
        if not self._load_whisper():
            return None
        try:
            result = self._whisper_model.transcribe(file_path)
            return result.get("text", "").strip()
        except Exception as e:
            self.emit_error(f"File transcription failed: {e}")
            return None
