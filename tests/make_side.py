"""Generate a synthetic 'side' WAV: surface noise, needle-drop pop, tracks separated by gaps, exit pop."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100


def noise(seconds: float, db: float, rng: np.random.Generator) -> np.ndarray:
    amp = 10 ** (db / 20)
    return rng.normal(0, amp, int(SR * seconds))


def tone(seconds: float, freq: float, db: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    amp = 10 ** (db / 20) * np.sqrt(2)
    env = np.minimum(1.0, t / 0.5) * np.minimum(1.0, (seconds - t) / 0.5)  # gentle in/out
    return amp * np.sin(2 * np.pi * freq * t) * env + noise(seconds, -70, rng)


def pop(rng: np.random.Generator, seconds: float = 0.08, db: float = -14) -> np.ndarray:
    amp = 10 ** (db / 20)
    n = int(SR * seconds)
    return amp * np.exp(-np.arange(n) / (n / 5)) * rng.normal(0, 1, n)


def build(track_lengths: list[float], gap: float = 3.0, lead_in: float = 2.0, seed: int = 1) -> tuple[np.ndarray, list[tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    parts = [noise(1.0, -75, rng), pop(rng), noise(lead_in, -62, rng)]
    expected = []
    pos = sum(len(p) for p in parts) / SR
    for i, length in enumerate(track_lengths):
        freq = 220 * (i + 1)
        parts.append(tone(length, freq, -18, rng))
        expected.append((pos, pos + length))
        pos += length
        parts.append(noise(gap, -62, rng))
        pos += gap
    parts += [noise(4.0, -62, rng), pop(rng), noise(0.5, -75, rng)]
    sig = np.concatenate(parts)
    return sig, expected


def write_wav(path: Path, sig: np.ndarray) -> None:
    stereo = np.stack([sig, sig], axis=1)
    data = np.clip(stereo * 2147483647, -2147483648, 2147483647).astype("<i4")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(4)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    out = Path(sys.argv[1])
    lengths = [float(x) for x in sys.argv[2:]] or [20, 12, 16]
    sig, expected = build(lengths)
    write_wav(out, sig)
    for s, e in expected:
        print(f"{s:.2f} {e:.2f}")
