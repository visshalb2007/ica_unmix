"""FastICA (Hyvarinen & Oja) implemented from scratch with NumPy.

Fixed-point update for one unit w on whitened data Z:
    w+ = E[ Z g(w^T Z) ] - E[ g'(w^T Z) ] w
followed by renormalisation. Deflation extracts units one at a time (Gram-Schmidt),
symmetric estimates all units in parallel and decorrelates with W <- (W W^T)^{-1/2} W.
"""
import numpy as np


# Non-linearities g and their derivatives g'. Each approximates negentropy.
#   logcosh : robust general purpose choice (good for most super-Gaussian sources)
#   exp     : robust for highly super-Gaussian / outlier-heavy sources
#   cube    : kurtosis-based, fastest but sensitive to outliers
def _logcosh(y, a=1.0):
    t = np.tanh(a * y)
    return t, a * (1.0 - t ** 2)


def _exp(y):
    e = np.exp(-0.5 * y ** 2)
    return y * e, (1.0 - y ** 2) * e


def _cube(y):
    return y ** 3, 3.0 * y ** 2


NONLINEARITIES = {"logcosh": _logcosh, "exp": _exp, "cube": _cube}


def _sym_decorrelate(W):
    """W <- (W W^T)^{-1/2} W using an eigendecomposition."""
    s, U = np.linalg.eigh(W @ W.T)
    s = np.maximum(s, 1e-12)
    return (U * (1.0 / np.sqrt(s))) @ U.T @ W


def fastica(Z, n_components=None, algorithm="symmetric", fun="logcosh",
            max_iter=500, tol=1e-6, seed=0, return_history=False):
    """Run FastICA on whitened data.

    Parameters
    ----------
    Z : (n, T) whitened data (zero mean, identity covariance).
    algorithm : "symmetric" or "deflation".
    fun : "logcosh", "exp" or "cube".

    Returns
    -------
    W : (n_components, n) orthogonal unmixing matrix in the whitened space
    S : (n_components, T) estimated sources, S = W Z
    n_iter : iterations used (int for symmetric, list for deflation)
    history : (only if return_history) per-iteration convergence trace
    """
    n, T = Z.shape
    m = n if n_components is None else n_components
    g = NONLINEARITIES[fun]
    rng = np.random.default_rng(seed)
    history = []

    if algorithm == "symmetric":
        W = _sym_decorrelate(rng.standard_normal((m, n)))
        n_iter = max_iter
        for it in range(1, max_iter + 1):
            Y = W @ Z
            gY, gpY = g(Y)
            W_new = (gY @ Z.T) / T - np.diag(gpY.mean(axis=1)) @ W
            W_new = _sym_decorrelate(W_new)
            # converged when each new row is parallel (up to sign) to the old row
            delta = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0))
            history.append(delta)
            W = W_new
            if delta < tol:
                n_iter = it
                break
    elif algorithm == "deflation":
        W = np.zeros((m, n))
        n_iter = []
        for p in range(m):
            w = rng.standard_normal(n)
            w /= np.linalg.norm(w)
            used = max_iter
            for it in range(1, max_iter + 1):
                y = w @ Z
                gy, gpy = g(y)
                w_new = (Z * gy).mean(axis=1) - gpy.mean() * w
                # Gram-Schmidt: remove projections on already found components
                w_new -= W[:p].T @ (W[:p] @ w_new)
                w_new /= np.linalg.norm(w_new)
                delta = abs(abs(w_new @ w) - 1.0)
                w = w_new
                if delta < tol:
                    used = it
                    break
            W[p] = w
            n_iter.append(used)
    else:
        raise ValueError("algorithm must be 'symmetric' or 'deflation'")

    S = W @ Z
    if return_history:
        return W, S, n_iter, history
    return W, S, n_iter
