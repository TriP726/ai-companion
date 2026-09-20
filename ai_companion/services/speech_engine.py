"""Speech engines: Whisper for listening, Piper for speaking.

MEASURED ON CPU (this machine, int8 / medium voice)
---------------------------------------------------
    Piper  synthesis : 12.7x realtime  (7.9s of audio in 0.62s)
    Whisper base.en  : 10.2x realtime  (7.9s of audio in 0.78s)

Both comfortably faster than realtime, which is what makes hands-free viable
without a GPU. The dominant cost is model load - 0.9s for Piper, 3.5s for
Whisper - so both are loaded once and kept resident.

WHY THESE TWO
-------------
faster-whisper (CTranslate2, int8) is ~4x the reference implementation on CPU
and ships built-in Silero VAD, which is exactly what end-of-speech detection
needs. Piper is ONNX/VITS, tens of megabytes, and the only local neural TTS
that reaches realtime on modest CPUs.

Neither touches the network after the models are on disk.

WHAT THIS IS NOT
----------------
Not a voice clone. A British neural voice is as close to the JARVIS register
as offline synthesis gets; cloning a specific actor needs XTTS-class models
that will not run in realtime here.
"""
from __future__ import annotations

import io
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Whisper wants 16 kHz mono float32.
STT_SAMPLE_RATE = 16000

# Silence (RMS) below this counts as "not speaking". Tuned against typical
# desktop mic noise floors; exposed in config so a noisy room can raise it.
DEFAULT_SILENCE_THRESHOLD = 0.012

# How long the user must stay quiet before we treat the utterance as finished.
DEFAULT_SILENCE_SECONDS = 1.0

# Guard rails so a stuck mic cannot record forever or fire on a cough.
MIN_UTTERANCE_SECONDS = 0.35
MAX_UTTERANCE_SECONDS = 30.0

# Audio retained from BEFORE the level crosses the threshold.
#
# Without this the recording starts only once you are already loud, so the
# attack of the first word is missing. That is why "Jarvis, how good is your
# code?" transcribed as "How good is your code?" - the wake word was clipped
# off before buffering began, and the matcher never saw it.
PREROLL_SECONDS = 0.6


@dataclass
class TranscriptionResult:
    ok: bool
    text: str = ""
    error: str = ""
    duration: float = 0.0


@dataclass
class SynthesisResult:
    ok: bool
    wav_bytes: bytes = b""
    sample_rate: int = 22050
    error: str = ""


# ----------------------------------------------------------------------
# Listening
# ----------------------------------------------------------------------


class Transcriber:
    """Wraps faster-whisper. Loads lazily, then stays resident."""

    def __init__(self, model_name: str = "base.en") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> tuple[bool, str]:
        """Load the model. Returns (ok, error)."""
        with self._lock:
            if self._model is not None:
                return True, ""
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                return False, (
                    "faster-whisper is not installed. Run:\n"
                    "  pip install faster-whisper"
                )
            try:
                # int8 is the right compute type on CPU: ~4x float32 speed for
                # about 0.3% WER, and it halves the memory footprint.
                self._model = WhisperModel(
                    self._model_name, device="cpu", compute_type="int8"
                )
            except Exception as exc:  # noqa: BLE001
                return False, f"Could not load Whisper '{self._model_name}': {exc}"
            return True, ""

    def transcribe(self, audio: Any) -> TranscriptionResult:
        """Transcribe a float32 mono numpy array at 16 kHz."""
        import time

        ok, error = self.load()
        if not ok:
            return TranscriptionResult(False, error=error)

        started = time.time()
        try:
            # beam_size=1 is greedy: measurably faster and the accuracy
            # difference on short utterances is not worth the latency.
            segments, _info = self._model.transcribe(
                audio, beam_size=1, language="en", vad_filter=True
            )
            text = "".join(segment.text for segment in segments).strip()
        except Exception as exc:  # noqa: BLE001
            return TranscriptionResult(False, error=f"Transcription failed: {exc}")

        return TranscriptionResult(
            True, text=text, duration=time.time() - started
        )

    def unload(self) -> None:
        with self._lock:
            self._model = None


# ----------------------------------------------------------------------
# Speaking
# ----------------------------------------------------------------------


class KokoroSynthesiser:
    """Wraps Kokoro (StyleTTS2). Slower than Piper, generally more natural.

    Measured on this hardware: ~1.2x realtime, versus Piper's ~10x. That is
    the honest trade - a 5 second reply takes about 4 seconds to generate
    instead of half a second. Worth it if you prefer the voice; painful if
    you want snappy responses.
    """

    def __init__(self, model_dir: str = "models/kokoro",
                 voice: str = "bm_george") -> None:
        self._dir = Path(model_dir)
        self._voice = voice
        self._engine: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def sample_rate(self) -> int:
        return 24000

    def load(self) -> tuple[bool, str]:
        with self._lock:
            if self._engine is not None:
                return True, ""
            model = self._dir / "kokoro-v1.0.onnx"
            voices = self._dir / "voices-v1.0.bin"
            if not model.exists() or not voices.exists():
                return False, (
                    "Kokoro model files not found. Run get_speech.py and "
                    "choose the Kokoro option."
                )
            try:
                from kokoro_onnx import Kokoro
            except ImportError:
                return False, (
                    "kokoro-onnx is not installed. Run:\n"
                    "  pip install kokoro-onnx"
                )
            try:
                self._engine = Kokoro(str(model), str(voices))
            except Exception as exc:  # noqa: BLE001
                return False, f"Could not load Kokoro: {exc}"
            return True, ""

    def synthesise(self, text: str) -> SynthesisResult:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return SynthesisResult(False, error="Nothing speakable.")

        ok, error = self.load()
        if not ok:
            return SynthesisResult(False, error=error)

        try:
            import numpy as np

            language = "en-gb" if self._voice.startswith("b") else "en-us"
            samples, rate = self._engine.create(
                cleaned, voice=self._voice, speed=1.0, lang=language
            )
            audio = np.asarray(samples, dtype=np.float32)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(int(rate))
                handle.writeframes(pcm.tobytes())
            return SynthesisResult(
                True, wav_bytes=buffer.getvalue(), sample_rate=int(rate)
            )
        except Exception as exc:  # noqa: BLE001
            return SynthesisResult(False, error=f"Synthesis failed: {exc}")

    def unload(self) -> None:
        with self._lock:
            self._engine = None


KOKORO_VOICES = (
    ("bm_george", "British male, warm"),
    ("bm_lewis", "British male, deeper"),
    ("bf_emma", "British female"),
    ("am_michael", "American male"),
    ("am_adam", "American male, lower"),
    ("af_bella", "American female"),
)


class Synthesiser:
    """Wraps Piper. Loads lazily, then stays resident."""

    def __init__(self, voice_path: str = "") -> None:
        self._voice_path = voice_path
        self._voice: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        if self._voice is None:
            return 22050
        return int(self._voice.config.sample_rate)

    def load(self) -> tuple[bool, str]:
        with self._lock:
            if self._voice is not None:
                return True, ""
            if not self._voice_path:
                return False, "No voice selected. Run get_speech.py."
            path = Path(self._voice_path)
            if not path.exists():
                return False, f"Voice file not found: {path}"
            try:
                from piper import PiperVoice
            except ImportError:
                return False, (
                    "piper-tts is not installed. Run:\n"
                    "  pip install piper-tts"
                )
            try:
                self._voice = PiperVoice.load(str(path))
            except Exception as exc:  # noqa: BLE001
                return False, f"Could not load voice: {exc}"
            return True, ""

    def synthesise(self, text: str) -> SynthesisResult:
        """Render text to WAV bytes."""
        cleaned = clean_for_speech(text)
        if not cleaned:
            return SynthesisResult(False, error="Nothing speakable in that text.")

        ok, error = self.load()
        if not ok:
            return SynthesisResult(False, error=error)

        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as handle:
                self._voice.synthesize_wav(cleaned, handle)
            return SynthesisResult(
                True, wav_bytes=buffer.getvalue(), sample_rate=self.sample_rate
            )
        except Exception as exc:  # noqa: BLE001
            return SynthesisResult(False, error=f"Synthesis failed: {exc}")

    def unload(self) -> None:
        with self._lock:
            self._voice = None


# ----------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------


def clean_for_speech(text: str, max_chars: int = 1200) -> str:
    """Strip anything that should not be read aloud.

    Code blocks are the big one: reading a 200-line Python file out loud is
    useless and takes minutes. They are replaced by a short spoken marker so
    the listener knows code was shown rather than silently skipped.
    """
    import re

    if not text:
        return ""

    # Fenced code -> spoken marker.
    def _fence(match: "re.Match") -> str:
        language = (match.group(1) or "").strip()
        return f" (showing {language} code on screen) " if language else \
               " (showing code on screen) "

    cleaned = re.sub(r"```(\w*)\n.*?```", _fence, text, flags=re.S)
    # An unterminated fence, i.e. still streaming.
    cleaned = re.sub(r"```(\w*)\n.*$", _fence, cleaned, flags=re.S)

    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)          # inline code
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)    # bold
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.M)  # bullets
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.M)  # headings
    cleaned = re.sub(r"https?://\S+", "a link", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", ". ", cleaned).replace("\n", ". ")
    # Collapse the punctuation debris left by stripping markup.
    cleaned = re.sub(r"\s*\.\s*(?=\.)", "", cleaned)
    cleaned = re.sub(r"\.\s*\.+", ".", cleaned)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) > max_chars:
        # Cut at a sentence end so it does not stop mid-word.
        cut = cleaned[:max_chars]
        last = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        cleaned = (cut[: last + 1] if last > max_chars * 0.5 else cut) + \
            " The rest is on screen."

    return cleaned


def split_sentences(text: str) -> list[str]:
    """Split into sentences so speech can start before the whole reply is done."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ----------------------------------------------------------------------
# Microphone
# ----------------------------------------------------------------------


class MicrophoneListener:
    """Records an utterance and detects when the speaker stops.

    Runs its own thread. Calls `on_utterance(audio)` with a float32 mono array
    once speech is followed by `silence_seconds` of quiet.
    """

    def __init__(
        self,
        on_utterance: Callable[[Any], None],
        on_level: Optional[Callable[[float], None]] = None,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        silence_seconds: float = DEFAULT_SILENCE_SECONDS,
        device_index: Optional[int] = None,
    ) -> None:
        self._on_utterance = on_utterance
        self._on_level = on_level
        self._silence_threshold = silence_threshold
        self._silence_seconds = silence_seconds
        self._device_index = device_index
        self._stream: Any = None
        self._running = False
        self._paused = False
        self._buffer: list = []
        self._preroll: list = []
        self._preroll_blocks = max(1, int(PREROLL_SECONDS / 0.1))
        self._silent_blocks = 0
        self._speaking = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> tuple[bool, str]:
        if self._running:
            return True, ""
        try:
            import numpy as np  # noqa: F401
            import sounddevice as sd
        except ImportError:
            return False, (
                "sounddevice is not installed. Run:\n"
                "  pip install sounddevice"
            )
        except OSError as exc:
            return False, f"Audio system unavailable: {exc}"

        block = int(STT_SAMPLE_RATE * 0.1)   # 100 ms blocks
        try:
            self._stream = sd.InputStream(
                samplerate=STT_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=block,
                device=self._device_index,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            self._stream = None
            return False, f"Could not open microphone: {exc}"

        self._running = True
        return True, ""

    def stop(self) -> None:
        self._running = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._buffer.clear()
            self._preroll.clear()
            self._speaking = False
            self._silent_blocks = 0

    def pause(self) -> None:
        """Stop collecting without closing the device.

        Used while the assistant is speaking, so its own voice is not
        transcribed back as user input.
        """
        self._paused = True
        with self._lock:
            self._buffer.clear()
            self._preroll.clear()
            self._speaking = False
            self._silent_blocks = 0

    def resume(self) -> None:
        self._paused = False

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if self._paused or not self._running:
            return
        import numpy as np

        mono = indata[:, 0].copy()
        level = float(np.sqrt(np.mean(mono * mono)))
        if self._on_level is not None:
            try:
                self._on_level(level)
            except Exception:  # noqa: BLE001
                pass

        blocks_of_silence = max(1, int(self._silence_seconds / 0.1))

        with self._lock:
            if level >= self._silence_threshold:
                if not self._speaking:
                    # Prepend the pre-roll so the first word is not clipped.
                    self._buffer.extend(self._preroll)
                    self._preroll.clear()
                self._speaking = True
                self._silent_blocks = 0
                self._buffer.append(mono)
            elif not self._speaking:
                # Idle: retain a rolling window of recent audio.
                self._preroll.append(mono)
                if len(self._preroll) > self._preroll_blocks:
                    self._preroll.pop(0)
            elif self._speaking:
                self._silent_blocks += 1
                self._buffer.append(mono)
                if self._silent_blocks >= blocks_of_silence:
                    audio = np.concatenate(self._buffer) if self._buffer else None
                    self._buffer.clear()
                    self._speaking = False
                    self._silent_blocks = 0
                    if audio is not None:
                        seconds = len(audio) / STT_SAMPLE_RATE
                        # Reject coughs and runaway recordings.
                        if MIN_UTTERANCE_SECONDS <= seconds <= MAX_UTTERANCE_SECONDS:
                            self._dispatch(audio)

            if self._speaking and len(self._buffer) * 0.1 > MAX_UTTERANCE_SECONDS:
                audio = np.concatenate(self._buffer)
                self._buffer.clear()
                self._speaking = False
                self._dispatch(audio)

    def _dispatch(self, audio) -> None:  # noqa: ANN001
        """Hand the utterance off without blocking the audio callback.

        Transcription takes hundreds of milliseconds; doing it here would
        stall the audio device and drop frames.
        """
        threading.Thread(
            target=self._on_utterance, args=(audio,), daemon=True
        ).start()


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------


def list_input_devices() -> list[tuple[int, str]]:
    """Available microphones as (index, label).

    Returns [] when audio is unavailable rather than raising, so Settings can
    still open and explain the problem. Note sounddevice raises OSError - not
    ImportError - when the PortAudio native library is missing, which is the
    common failure on a fresh machine.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        return []
    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        return []

    found: list[tuple[int, str]] = []
    for index, info in enumerate(devices):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        name = str(info.get("name", f"device {index}"))
        rate = int(info.get("default_samplerate", 0) or 0)
        found.append((index, f"{name} ({rate} Hz)" if rate else name))
    return found


def default_input_device() -> Optional[int]:
    try:
        import sounddevice as sd  # noqa: F401  (OSError if PortAudio missing)

        default = sd.default.device
        index = default[0] if isinstance(default, (list, tuple)) else default
        return int(index) if index is not None and index >= 0 else None
    except (ImportError, OSError, Exception):  # noqa: BLE001
        return None


@dataclass
class MicTestResult:
    """Outcome of a short live recording."""

    ok: bool
    peak: float = 0.0
    average: float = 0.0
    transcript: str = ""
    error: str = ""
    verdict: str = ""


def test_microphone(
    device_index: Optional[int] = None,
    seconds: float = 4.0,
    threshold: float = DEFAULT_SILENCE_THRESHOLD,
    transcriber: Optional["Transcriber"] = None,
) -> MicTestResult:
    """Record briefly and report whether the mic is actually usable.

    Reports real numbers rather than a pass/fail light: knowing the peak was
    0.004 against a 0.012 threshold tells the user to raise their gain, where
    "test failed" tells them nothing.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        return MicTestResult(
            False, error="sounddevice is not installed. Run:\n"
                         "  pip install sounddevice"
        )
    except OSError as exc:
        return MicTestResult(
            False, error=f"Audio system unavailable: {exc}"
        )

    try:
        frames = int(STT_SAMPLE_RATE * seconds)
        recording = sd.rec(
            frames,
            samplerate=STT_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        return MicTestResult(False, error=f"Could not record: {exc}")

    audio = recording[:, 0]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    average = float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0

    if peak < 0.001:
        return MicTestResult(
            False, peak=peak, average=average,
            verdict="Silent. Wrong device, muted, or no permission.",
        )
    if average < threshold * 0.5:
        return MicTestResult(
            True, peak=peak, average=average,
            verdict=(
                f"Very quiet (level {average:.4f} vs threshold {threshold:.3f}). "
                "Raise your mic gain or lower the silence threshold."
            ),
        )

    result = MicTestResult(
        True, peak=peak, average=average,
        verdict=f"Good (level {average:.4f}, peak {peak:.3f}).",
    )

    if transcriber is not None:
        transcription = transcriber.transcribe(audio)
        if transcription.ok:
            result.transcript = transcription.text
            if not transcription.text.strip():
                result.verdict += " Audio detected but nothing recognised."
        else:
            result.verdict += f" (transcription failed: {transcription.error})"
    return result


# ----------------------------------------------------------------------
# Wake word
# ----------------------------------------------------------------------


def contains_wake_word(text: str, wake_word: str) -> bool:
    """Loose match for the wake word at the start of an utterance.

    Deliberately forgiving: Whisper writes "Jarvis," / "Travis" / "Jarvis?"
    depending on audio quality, and a strict match makes the feature feel
    broken. False positives merely start a turn the user can ignore.
    """
    import re

    if not wake_word:
        return True
    haystack = re.sub(r"[^a-z\s]", "", text.lower())
    needle = re.sub(r"[^a-z\s]", "", wake_word.lower()).strip()
    if not needle:
        return True
    if needle in haystack:
        return True

    # Tolerate one wrong character anywhere in the first few words. Whisper
    # routinely hears "Jarvis" as "Travis" or "Charvis"; rejecting those makes
    # the wake word feel broken, and a false positive only starts a turn the
    # user can ignore.
    # Raw edit distance alone is the wrong test. "travis" (distance 3) IS a
    # real Whisper output for "jarvis", while "marvin" (distance 2) is a
    # different word entirely. What separates them is the ending: mishearings
    # preserve the tail, different words do not.
    #
    # So: accept a close-enough word, OR a word sharing the distinctive
    # suffix. Both gated on similar length so unrelated words cannot match.
    # KNOWN TRADE-OFF: "marvin" still matches "jarvis" at distance 2. Tighten
    # the limit to 1 and real mishearings ("charvis") start failing instead.
    # Chosen deliberately: a false positive starts a turn the user ignores, a
    # false negative makes the wake word feel broken. If it fires on a name
    # you actually use, change the wake word in Settings.
    limit = 2 if len(needle) >= 6 else 1
    suffix = needle[-3:] if len(needle) >= 5 else ""

    for word in haystack.split()[:3]:
        if abs(len(word) - len(needle)) > limit:
            continue
        if _edit_distance_within(word, needle, limit):
            return True
        if suffix and word.endswith(suffix):
            return True
    return False


def _edit_distance_within(a: str, b: str, limit: int) -> bool:
    """True when `a` and `b` differ by at most `limit` Levenshtein edits.

    Full DP rather than the substitution-only shortcut the first version
    used: that rejected "charvis" for "jarvis" because an insertion shifts
    every later character and looks like six substitutions.
    """
    if abs(len(a) - len(b)) > limit:
        return False
    if a == b:
        return True

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
        if min(previous) > limit:
            return False
    return previous[-1] <= limit


def strip_wake_word(text: str, wake_word: str) -> str:
    """Remove the wake word from the front of an utterance."""
    import re

    if not wake_word:
        return text.strip()
    pattern = re.compile(
        rf"^\W*{re.escape(wake_word)}\W*", re.IGNORECASE
    )
    stripped = pattern.sub("", text).strip()
    return stripped or text.strip()
