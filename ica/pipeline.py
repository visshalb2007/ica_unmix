"""End-to-end helpers: build mixtures, run centre -> whiten -> FastICA."""
import numpy as np
from .preprocessing import center, pca_whiten
from .fastica import fastica


def make_mixing_matrix(n_mix, n_src, seed=1, max_cond=10.0):
    """Random positive mixing matrix (each 'microphone' hears every source).

    Re-draws until the condition number is below max_cond so the problem is
    well posed (a near-singular A makes any unmixing numerically fragile).
    """
    rng = np.random.default_rng(seed)
    while True:
        A = rng.uniform(0.2, 1.0, size=(n_mix, n_src))
        if np.linalg.cond(A) < max_cond:
            return A


def add_noise(X, snr_db, seed=0):
    """Add white Gaussian sensor noise at a given per-channel SNR (dB)."""
    rng = np.random.default_rng(seed)
    p_sig = np.mean(X ** 2, axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return X + rng.standard_normal(X.shape) * np.sqrt(p_noise)


def separate(X, n_sources, algorithm="symmetric", fun="logcosh", seed=0,
             max_iter=500, tol=1e-6, return_history=False):
    """Blind separation of mixtures X (channels, T) into n_sources estimates.

    Steps: 1) centre  2) PCA whitening (also reduces M -> N)  3) FastICA.
    Returns dict containing ICA estimates, the PCA-only estimates (whitened
    data, i.e. the principal components rescaled to unit variance) and the
    total unmixing matrix W_total such that S_ica = W_total @ (X - mean).
    """
    Xc, mean = center(X)
    Z, K, K_inv, eigvals = pca_whiten(Xc, n_components=n_sources)
    res = fastica(Z, algorithm=algorithm, fun=fun, seed=seed,
                  max_iter=max_iter, tol=tol, return_history=return_history)
    if return_history:
        W, S, n_iter, hist = res
    else:
        W, S, n_iter = res
        hist = None
    return {"S_ica": S, "S_pca": Z, "W": W, "K": K, "W_total": W @ K,
            "mean": mean, "eigvals": eigvals, "n_iter": n_iter, "history": hist}
