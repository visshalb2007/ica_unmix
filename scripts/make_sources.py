"""Generate the three clean source recordings (license-free, fully reproducible).

  speech.wav  - espeak-ng text-to-speech (male voice)
  music.wav   - additive-synthesis melody + bass line (pentatonic, 100 bpm)
  ambient.wav - rain-like ambience: sparse random drops + low rumble

All are 16 kHz mono, 10 s. Replace them with your own recordings if you like:
any three clean mono WAVs in data/sources/ will work.
"""

import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import butter, lfilter, resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ica import audio_io

SR, DUR = 16000, 10.0
N = int(SR * DUR)
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "sources")
rng = np.random.default_rng(42)


def make_speech():
    text = (
        "Independent component analysis recovers hidden signals from mixtures. "
        "Each microphone hears a blend of every speaker, yet the sources are "
        "statistically independent, and that is the clue we exploit. "
        "Whitening removes correlations, and a rotation finds the rest."
    )
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "s.wav")
        subprocess.run(
            ["espeak-ng", "-v", "en-us+m3", "-s", "150", "-p", "40", "-w", wav, text], check=True
        )
        sr, x = audio_io.read_wav(wav)
    g = math.gcd(SR, sr)
    x = resample_poly(x, SR // g, sr // g)
    x = np.pad(x, (0, max(0, N - len(x))))[:N]
    return x


def make_music():
    bpm = 100
    beat = 60.0 / bpm
    scale = [0, 2, 4, 7, 9]  # C major pentatonic
    t_all = np.zeros(N)

    def note(freq, dur, harmonics=(1.0, 0.5, 0.33, 0.2, 0.12)):
        n = int(dur * SR)
        t = np.arange(n) / SR
        y = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
        env = np.minimum(1, t / 0.01) * np.exp(-3.0 * t / dur)  # fast attack, decay
        return y * env

    # melody
    pos = 0
    while pos < N:
        dur = rng.choice([0.5, 1, 1, 2]) * beat
        deg = rng.choice(scale) + 12 * rng.choice([1, 2])
        f = 261.63 * 2 ** (deg / 12)
        y = note(f, dur)
        end = min(N, pos + len(y))
        t_all[pos:end] += y[: end - pos]
        pos += len(y)
    # bass line: one note per 2 beats
    pos = 0
    roots = [0, 7, 9, 5]
    i = 0
    while pos < N:
        f = 65.41 * 2 ** (roots[i % 4] / 12)
        y = 1.4 * note(f, 2 * beat, harmonics=(1.0, 0.4, 0.15))
        end = min(N, pos + len(y))
        t_all[pos:end] += y[: end - pos]
        pos += len(y)
        i += 1
    return t_all


def make_ambient():
    # sparse random "drops" give an impulsive, super-Gaussian rain texture
    drops = np.zeros(N)
    idx = rng.random(N) < 0.03
    drops[idx] = rng.exponential(1.0, idx.sum()) * rng.choice([-1, 1], idx.sum())
    b, a = butter(2, [1500 / (SR / 2), 6500 / (SR / 2)], btype="band")
    drops = lfilter([1.0], [1.0, -np.exp(-1 / (0.002 * SR))], drops)  # ~2 ms splash decay
    patter = lfilter(b, a, drops)
    # slow rumble, amplitude modulated
    b2, a2 = butter(2, 120 / (SR / 2), btype="low")
    rumble = lfilter(b2, a2, rng.standard_normal(N))
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * np.arange(N) / SR)
    return 1.0 * patter / patter.std() + 0.6 * mod * rumble / rumble.std()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("speech", make_speech), ("music", make_music), ("ambient", make_ambient)]:
        x = fn()
        x = x - x.mean()
        x = x / np.abs(x).max()
        audio_io.write_wav(os.path.join(OUT, f"{name}.wav"), x, SR)
        print(f"wrote {name}.wav  ({len(x) / SR:.1f}s)")
