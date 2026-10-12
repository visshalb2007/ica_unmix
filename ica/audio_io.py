"""WAV helpers (16-bit PCM) using scipy.io.wavfile."""
import numpy as np
from scipy.io import wavfile


def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind in "iu":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:           # fold to mono
        x = x.mean(axis=1)
    return sr, x


def write_wav(path, x, sr, peak=0.9):
    x = np.asarray(x, dtype=np.float64)
    m = np.abs(x).max()
    if m > 0:
        x = x * (peak / m)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))
