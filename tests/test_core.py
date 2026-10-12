"""Sanity tests: python -m pytest tests  (or: python tests/test_core.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ica import center, pca_whiten, fastica, evaluate, si_sdr  # noqa: E402
from ica.pipeline import make_mixing_matrix, separate  # noqa: E402


def _toy_sources(T=20000, seed=0):
    r = np.random.default_rng(seed)
    t = np.arange(T)
    return np.vstack([np.sign(np.sin(2 * np.pi * 7e-4 * t)),            # square (sub-Gaussian)
                      r.laplace(size=T),                                 # Laplace (super-Gaussian)
                      (t * 3e-4) % 1.0 - 0.5])                           # sawtooth (sub-Gaussian)


def test_centering():
    X = np.random.default_rng(0).standard_normal((3, 1000)) + 5
    Xc, m = center(X)
    assert np.allclose(Xc.mean(axis=1), 0, atol=1e-12) and np.allclose(m, 5, atol=0.2)


def test_whitening_identity_covariance():
    S = _toy_sources()
    X = make_mixing_matrix(3, 3) @ S
    Z, K, K_inv, _ = pca_whiten(center(X)[0])
    assert np.allclose(np.cov(Z), np.eye(3), atol=1e-8)
    assert np.allclose(K @ K_inv, np.eye(3), atol=1e-8)


def test_fastica_recovers_sources_all_variants():
    S = _toy_sources()
    X = make_mixing_matrix(3, 3, seed=3) @ S
    for alg in ["symmetric", "deflation"]:
        for fun in ["logcosh", "exp", "cube"]:
            r = separate(X, 3, algorithm=alg, fun=fun)
            ev = evaluate(S, r["S_ica"])
            assert ev["corr"].min() > 0.99, (alg, fun, ev["corr"])


def test_unmixing_orthogonal_in_whitened_space():
    S = _toy_sources()
    r = separate(make_mixing_matrix(3, 3) @ S, 3)
    assert np.allclose(r["W"] @ r["W"].T, np.eye(3), atol=1e-8)


def test_si_sdr_scale_invariant():
    r = np.random.default_rng(1)
    x = r.standard_normal(5000)
    y = x + 0.1 * r.standard_normal(5000)
    assert abs(si_sdr(x, y) - si_sdr(x, 7.3 * y)) < 1e-9


def test_overdetermined_mixtures():
    S = _toy_sources()
    X = make_mixing_matrix(5, 3) @ S          # 5 mixtures, 3 sources
    ev = evaluate(S, separate(X, 3)["S_ica"])
    assert ev["corr"].min() > 0.99


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
