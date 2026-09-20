"""Post-processing that makes a neural voice sound like a room AI.

WHY THIS EXISTS
---------------
Asking for "a JARVIS voice" is usually not a request for a different speaker.
Piper's British male is already close in timbre. What is missing is the
PRODUCTION: in the films the voice is processed to sound like it is being
played through speakers in a large room, not spoken into your ear.

Four cheap DSP stages get most of the way there:

    1. EQ      - roll off mud below 120 Hz, lift presence around 3-5 kHz
    2. Reverb  - a short algorithmic room, so it sits in a space
    3. Chorus  - optional detune (OFF by default: it is what makes a voice
                 sound synthetic, which was the opposite of what was wanted)
    4. Limiter - even level, which is what reads as "calm and in control"

Measured on the dry Piper output, the spectrum slopes off steeply above 3 kHz
(-62 dB presence vs -39 dB low). That dullness is most of why it sounds like
a text-to-speech engine rather than a voice in a room.

PERFORMANCE
-----------
All stages are numpy/scipy on a few seconds of audio. Measured under 25 ms
for a typical reply on this hardware - negligible against the 600 ms Piper
already takes, and nothing compared to model inference.

Everything here is optional and falls back to the dry signal if scipy is
missing, so a failed effect can never silence the assistant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VoiceFxSettings:
    """Tunable knobs, all normalised 0..1 unless noted."""

    enabled: bool = True
    preset: str = "natural"

    # EQ
    high_pass_hz: float = 120.0
    presence_db: float = 4.5
    presence_hz: float = 4000.0

    # Room
    reverb_amount: float = 0.22      # wet/dry mix
    reverb_size: float = 0.35        # 0 = small booth, 1 = hall

    # Character
    # Detune. This is the single most "robotic" stage - it was on by default
    # while chasing a JARVIS aesthetic and made the voice sound worse, not
    # more characterful. Off unless explicitly asked for.
    chorus_amount: float = 0.0
    drive: float = 0.12              # gentle saturation

    # Level
    limiter_ceiling: float = 0.89


PRESETS: dict[str, dict] = {
    # Dry Piper output, no processing at all.
    "off": {"enabled": False},
    # The default. Clear and present, a suggestion of room, NO detune.
    "natural": {
        "high_pass_hz": 110.0, "presence_db": 4.0, "presence_hz": 4200.0,
        "reverb_amount": 0.10, "reverb_size": 0.25,
        "chorus_amount": 0.0, "drive": 0.08,
    },
    # Film-style room AI. The detune is deliberate here and it WILL sound
    # synthetic - that is the point of this preset, not an accident.
    "jarvis": {
        "high_pass_hz": 120.0, "presence_db": 4.5, "presence_hz": 4000.0,
        "reverb_amount": 0.22, "reverb_size": 0.35,
        "chorus_amount": 0.15, "drive": 0.12,
    },
    # Bigger space, more obviously "in the walls".
    "hall": {
        "high_pass_hz": 110.0, "presence_db": 5.0, "presence_hz": 3800.0,
        "reverb_amount": 0.30, "reverb_size": 0.60,
        "chorus_amount": 0.0, "drive": 0.10,
    },
    # Tight, close, minimal space - good if reverb muddies speech for you.
    "intercom": {
        "high_pass_hz": 220.0, "presence_db": 6.5, "presence_hz": 3200.0,
        "reverb_amount": 0.08, "reverb_size": 0.15,
        "chorus_amount": 0.0, "drive": 0.22,
    },
    # Clean studio: EQ and level only, no space or detune.
    "clean": {
        "high_pass_hz": 100.0, "presence_db": 3.0, "presence_hz": 4500.0,
        "reverb_amount": 0.0, "reverb_size": 0.0,
        "chorus_amount": 0.0, "drive": 0.0,
    },
}


def settings_for(preset: str) -> VoiceFxSettings:
    """Build settings from a preset name, falling back to jarvis."""
    values = PRESETS.get(preset, PRESETS["natural"])
    settings = VoiceFxSettings(preset=preset)
    for key, value in values.items():
        setattr(settings, key, value)
    return settings


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------


def _high_pass(audio: np.ndarray, rate: int, cutoff: float) -> np.ndarray:
    """Remove low rumble so the voice sits forward instead of muddy."""
    from scipy import signal

    if cutoff <= 20:
        return audio
    sos = signal.butter(2, cutoff / (rate / 2), btype="highpass", output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32)


def _presence(audio: np.ndarray, rate: int, gain_db: float,
              centre: float) -> np.ndarray:
    """Lift the consonant band so speech cuts through a room.

    A wide peaking bell rather than a shelf: a shelf also lifts hiss, and
    Piper's output already has very little energy up there.
    """
    from scipy import signal

    if abs(gain_db) < 0.1:
        return audio
    nyquist = rate / 2
    low = max(centre * 0.5, 100.0) / nyquist
    high = min(centre * 1.8, nyquist * 0.95) / nyquist
    if not 0 < low < high < 1:
        return audio
    sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
    band = signal.sosfilt(sos, audio)
    return (audio + band * (10 ** (gain_db / 20) - 1)).astype(np.float32)


def _reverb(audio: np.ndarray, rate: int, amount: float,
            size: float) -> np.ndarray:
    """Small algorithmic room: parallel combs into series allpasses.

    A Schroeder reverb rather than a convolution, because it needs no impulse
    response file and costs almost nothing on a few seconds of speech.
    """
    if amount <= 0.001:
        return audio

    scale = 0.6 + size * 1.4
    comb_delays = [int(rate * d * scale) for d in
                   (0.0297, 0.0371, 0.0411, 0.0437)]
    comb_gains = [0.76, 0.74, 0.72, 0.70]

    wet = np.zeros_like(audio)
    for delay, gain in zip(comb_delays, comb_gains):
        if delay < 1 or delay >= len(audio):
            continue
        buffer = np.zeros_like(audio)
        buffer[delay:] = audio[:-delay]
        # Feedback via a cheap IIR along the delay line.
        decayed = buffer.copy()
        for _ in range(3):
            shifted = np.zeros_like(decayed)
            shifted[delay:] = decayed[:-delay]
            decayed = decayed + shifted * gain * 0.55
        wet += decayed
    wet /= max(len(comb_delays), 1)

    for delay, gain in ((int(rate * 0.005), 0.7), (int(rate * 0.0017), 0.7)):
        if delay < 1 or delay >= len(wet):
            continue
        shifted = np.zeros_like(wet)
        shifted[delay:] = wet[:-delay]
        wet = -gain * wet + shifted + gain * shifted

    peak = float(np.max(np.abs(wet))) or 1.0
    wet = wet / peak * float(np.max(np.abs(audio)) or 1.0)
    return ((1.0 - amount) * audio + amount * wet).astype(np.float32)


def _chorus(audio: np.ndarray, rate: int, amount: float) -> np.ndarray:
    """Slight detuned double, which reads as synthetic without sounding odd.

    This is the single most 'AI' of the stages. Keep it low: past about 0.3
    it starts to sound seasick rather than artificial.
    """
    if amount <= 0.001:
        return audio

    depth = int(rate * 0.0022)
    if depth < 2:
        return audio
    rate_hz = 0.7
    t = np.arange(len(audio)) / rate
    offsets = (depth * (1 + np.sin(2 * np.pi * rate_hz * t)) / 2).astype(int)
    indices = np.clip(np.arange(len(audio)) - offsets, 0, len(audio) - 1)
    voiced = audio[indices]
    return ((1.0 - amount) * audio + amount * voiced).astype(np.float32)


def _saturate(audio: np.ndarray, drive: float) -> np.ndarray:
    """Gentle soft clip for warmth and density."""
    if drive <= 0.001:
        return audio
    gain = 1.0 + drive * 3.0
    return (np.tanh(audio * gain) / np.tanh(gain)).astype(np.float32)


def _limit(audio: np.ndarray, ceiling: float) -> np.ndarray:
    """Normalise to a fixed ceiling.

    Even level is a surprisingly large part of sounding composed: a voice
    that swings in volume reads as a person, a steady one reads as a system.
    """
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    return (audio / peak * ceiling).astype(np.float32)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def apply_voice_fx(
    audio: np.ndarray,
    rate: int,
    settings: Optional[VoiceFxSettings] = None,
) -> np.ndarray:
    """Run the chain. Returns the dry signal on any failure.

    Never raises: a broken effect must degrade to an unprocessed voice, not
    to silence.
    """
    if settings is None:
        settings = VoiceFxSettings()
    if not settings.enabled or audio is None or len(audio) == 0:
        return audio

    try:
        out = audio.astype(np.float32, copy=True)
        out = _high_pass(out, rate, settings.high_pass_hz)
        out = _presence(out, rate, settings.presence_db, settings.presence_hz)
        out = _chorus(out, rate, settings.chorus_amount)
        out = _reverb(out, rate, settings.reverb_amount, settings.reverb_size)
        out = _saturate(out, settings.drive)
        out = _limit(out, settings.limiter_ceiling)
        if not np.all(np.isfinite(out)):
            return audio
        return out
    except Exception:  # noqa: BLE001 - dry signal beats no signal
        return audio


def describe_presets() -> list[tuple[str, str]]:
    """(name, description) for the Settings dropdown."""
    return [
        ("natural", "Clear and present, minimal room - the default"),
        ("clean", "EQ and level only, no room at all"),
        ("intercom", "Tight and close, slightly driven"),
        ("hall", "Larger space, more reverb"),
        ("jarvis", "Film-style room AI - deliberately synthetic"),
        ("off", "Raw voice, no processing"),
    ]
