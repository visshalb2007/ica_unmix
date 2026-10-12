"""Evaluation that handles ICA's scale, sign and permutation ambiguities."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # only used for source matching

FloatArray = NDArray[np.floating]


def excess_kurtosis(X: FloatArray) -> FloatArray:
    """Excess kurtosis per row (0 for a Gaussian)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    v = (Xc**2).mean(axis=1)
    return (Xc**4).mean(axis=1) / v**2 - 3.0


def correlation_matrix(S_true: FloatArray, S_est: FloatArray) -> FloatArray:
    """Pearson correlation, rows of S_true vs rows of S_est -> (N_true, N_est)."""
    a = S_true - S_true.mean(1, keepdims=True)
    b = S_est - S_est.mean(1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def match_sources(S_true: FloatArray, S_est: FloatArray) -> tuple[NDArray[np.intp], FloatArray]:
    """Resolve permutation and sign ambiguity.

    Finds the assignment of estimates to true sources that maximises total
    |correlation| (Hungarian algorithm), then flips signs so correlation > 0.
    Returns (perm, signs) with S_aligned = signs[:,None] * S_est[perm].
    """
    C = correlation_matrix(S_true, S_est)
    rows, cols = linear_sum_assignment(-np.abs(C))
    perm = cols[np.argsort(rows)]
    signs = np.sign(C[np.arange(len(perm)), perm])
    signs[signs == 0] = 1.0
    return perm, signs


def si_sdr(ref: FloatArray, est: FloatArray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019).

    est is projected onto ref, so any rescaling of est does not change the score.
    """
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / np.dot(ref, ref)
    target = alpha * ref
    noise = est - target
    return 10.0 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + 1e-20))


def amari_index(P: FloatArray) -> float:
    """Amari performance index of the global matrix P = W_total @ A.

    0 means a perfect (scaled permutation) solution. Normalised to [0, 1].
    """
    P = np.abs(P)
    n = P.shape[0]
    r = (P.sum(1) / P.max(1) - 1.0).sum()
    c = (P.sum(0) / P.max(0) - 1.0).sum()
    return (r + c) / (2.0 * n * (n - 1)) if n > 1 else 0.0


def evaluate(
    S_true: FloatArray,
    S_est: FloatArray,
    mixtures: FloatArray | None = None,
) -> dict:
    """Full evaluation of estimates against ground truth sources.

    Returns dict with the matching, per-source |correlation|, per-source SI-SDR,
    and the SI-SDR of the best raw mixture channel as a no-separation baseline.
    """
    perm, signs = match_sources(S_true, S_est)
    S_al = signs[:, None] * S_est[perm]
    corr = np.array([abs(np.corrcoef(S_true[i], S_al[i])[0, 1]) for i in range(len(perm))])
    sdr = np.array([si_sdr(S_true[i], S_al[i]) for i in range(len(perm))])
    out = {"perm": perm, "signs": signs, "aligned": S_al, "corr": corr, "si_sdr": sdr}
    if mixtures is not None:
        base = np.array([max(si_sdr(S_true[i], x) for x in mixtures) for i in range(len(perm))])
        out["si_sdr_best_mixture"] = base
    return out
