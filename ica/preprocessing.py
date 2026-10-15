"""Centering and PCA whitening, implemented with NumPy linear algebra only."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def center(X: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Subtract the per-channel mean.

    X : (channels, samples). Returns (X_centered, mean) where mean is (channels, 1).
    """
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def pca_whiten(
    Xc: FloatArray,
    n_components: int | None = None,
    eps: float = 1e-10,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Whiten centered data with PCA via eigendecomposition of the covariance.

    Covariance  C = Xc Xc^T / (T-1) = E D E^T   (E orthogonal, D diagonal)
    Whitening   K = D^{-1/2} E^T                (keeps the top n_components)
    Z = K Xc  has identity covariance.

    Returns
    -------
    Z          : (n_components, T) whitened data
    K          : (n_components, channels) whitening matrix
    K_inv      : (channels, n_components) de-whitening matrix  E D^{1/2}
    eigvals    : eigenvalues of the covariance, descending (all channels)
    """
    n_ch, T = Xc.shape
    C = (Xc @ Xc.T) / (T - 1)
    eigvals, E = np.linalg.eigh(C)  # ascending, C is symmetric
    order = np.argsort(eigvals)[::-1]  # descending
    eigvals, E = eigvals[order], E[:, order]

    k = n_ch if n_components is None else n_components
    d = np.maximum(eigvals[:k], eps)
    K = (E[:, :k] / np.sqrt(d)).T  # D^{-1/2} E^T
    K_inv = E[:, :k] * np.sqrt(d)  # E D^{1/2}
    Z = K @ Xc
    return Z, K, K_inv, eigvals
