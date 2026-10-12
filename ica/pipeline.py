"""End-to-end helpers: build mixtures, run centre -> whiten -> FastICA."""

import numpy as np
from numpy.typing import NDArray

from .fastica import fastica
from .preprocessing import center, pca_whiten

FloatArray = NDArray[np.floating]


def make_mixing_matrix(
    n_mix: int,
    n_src: int,
    seed: int = 1,
    max_cond: float = 10.0,
) -> FloatArray:
    """Random positive mixing matrix (each 'microphone' hears every source).

    Re-draws until the condition number is below max_cond so the problem is
    well posed (a near-singular A makes any unmixing numerically fragile).
    """
    rng = np.random.default_rng(seed)
    while True:
        A = rng.uniform(0.2, 1.0, size=(n_mix, n_src))
        if np.linalg.cond(A) < max_cond:
            return A


def add_noise(X: FloatArray, snr_db: float, seed: int = 0) -> FloatArray:
    """Add white Gaussian sensor noise at a given per-channel SNR (dB)."""
    rng = np.random.default_rng(seed)
    p_sig = np.mean(X**2, axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return X + rng.standard_normal(X.shape) * np.sqrt(p_noise)


def separate(
    X: FloatArray,
    n_sources: int,
    algorithm: str = "symmetric",
    fun: str = "logcosh",
    seed: int = 0,
    max_iter: int = 500,
    tol: float = 1e-6,
    return_history: bool = False,
) -> dict:
    """Blind separation of mixtures X (channels, T) into n_sources estimates.

    Steps: 1) centre  2) PCA whitening (also reduces M -> N)  3) FastICA.
    Returns dict containing ICA estimates, the PCA-only estimates (whitened
    data, i.e. the principal components rescaled to unit variance) and the
    total unmixing matrix W_total such that S_ica = W_total @ (X - mean).
    """
    Xc, mean = center(X)
    Z, K, _K_inv, eigvals = pca_whiten(Xc, n_components=n_sources)
    res = fastica(
        Z,
        algorithm=algorithm,
        fun=fun,
        seed=seed,
        max_iter=max_iter,
        tol=tol,
        return_history=return_history,
    )
    if return_history:
        W, S, n_iter, hist = res
    else:
        W, S, n_iter, _ = res
        hist = None
    return {
        "S_ica": S,
        "S_pca": Z,
        "W": W,
        "K": K,
        "W_total": W @ K,
        "mean": mean,
        "eigvals": eigvals,
        "n_iter": n_iter,
        "history": hist,
    }
