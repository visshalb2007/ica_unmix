"""Experiments for the write-up.

  A. PCA vs ICA over many random mixing matrices
  B. Separation quality vs sensor-noise SNR (with oracle reference)
  C. Non-linearity x algorithm comparison (quality, iterations, time)
  D. Why non-Gaussianity matters (Gaussian sources cannot be separated) + CLT check
  E. Number of mixtures M >= N sources

Run:  python scripts/experiments.py        (about 1-2 minutes)
"""
import json
import os
import sys
import time
import warnings

import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from ica import audio_io, evaluate, excess_kurtosis  # noqa: E402
from ica.pipeline import make_mixing_matrix, add_noise, separate  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)
NAMES = ["speech", "music", "ambient"]

S = np.array([audio_io.read_wav(os.path.join(ROOT, "data", "sources", f"{n}.wav"))[1] for n in NAMES])
S = S / S.std(axis=1, keepdims=True)
N_SRC = S.shape[0]
results = {}


def run(Sx, A, snr=None, seed=0, **kw):
    X = A @ Sx
    if snr is not None:
        X = add_noise(X, snr, seed=seed)
    r = separate(X, Sx.shape[0], seed=seed, **kw)
    return X, r, evaluate(Sx, r["S_ica"]), evaluate(Sx, r["S_pca"])


# ------------------------------------------------------------------ A: PCA vs ICA
n_trials = 20
ica_sdr, pca_sdr, conds = [], [], []
for t in range(n_trials):
    A = make_mixing_matrix(3, N_SRC, seed=100 + t)
    _, _, ei, ep = run(S, A, seed=t)
    ica_sdr.append(ei["si_sdr"].mean()); pca_sdr.append(ep["si_sdr"].mean()); conds.append(np.linalg.cond(A))
results["A_pca_vs_ica"] = {"trials": n_trials,
                           "ica_mean_si_sdr_db": round(float(np.mean(ica_sdr)), 2),
                           "ica_min_si_sdr_db": round(float(np.min(ica_sdr)), 2),
                           "pca_mean_si_sdr_db": round(float(np.mean(pca_sdr)), 2),
                           "pca_max_si_sdr_db": round(float(np.max(pca_sdr)), 2)}
print("A", results["A_pca_vs_ica"])

# ------------------------------------------------------------------ B: SNR sweep
snrs = [60, 40, 30, 20, 10, 5, 0, -5]
trials = 10
B = {"snr": snrs, "ica": [], "ica_std": [], "pca": [], "pca_std": [], "oracle": [], "oracle_std": [], "mixture": []}
for snr in snrs:
    a, p, o, m = [], [], [], []
    for t in range(trials):
        A = make_mixing_matrix(3, N_SRC, seed=200 + t)
        X, r, ei, ep = run(S, A, snr=snr, seed=t)
        eo = evaluate(S, np.linalg.pinv(A) @ X)          # oracle: knows A exactly
        a.append(ei["si_sdr"].mean()); p.append(ep["si_sdr"].mean()); o.append(eo["si_sdr"].mean())
        m.append(ei["si_sdr_best_mixture"].mean() if "si_sdr_best_mixture" in ei else 0)
    B["ica"].append(np.mean(a)); B["ica_std"].append(np.std(a))
    B["pca"].append(np.mean(p)); B["pca_std"].append(np.std(p))
    B["oracle"].append(np.mean(o)); B["oracle_std"].append(np.std(o))
    print(f"B snr={snr:>3} dB  ICA {np.mean(a):6.2f}  oracle {np.mean(o):6.2f}  PCA {np.mean(p):5.2f}")
results["B_snr_sweep"] = {k: [round(float(v), 2) for v in vals] if k != "snr" else vals for k, vals in B.items() if k != "mixture"}

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.errorbar(snrs, B["oracle"], B["oracle_std"], marker="s", ls="--", color="gray", label="oracle (true A inverted)")
ax.errorbar(snrs, B["ica"], B["ica_std"], marker="o", color="#c0392b", label="ICA (ours)")
ax.errorbar(snrs, B["pca"], B["pca_std"], marker="^", color="#1f4e79", label="PCA only")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("sensor SNR of mixtures (dB)"); ax.set_ylabel("mean SI-SDR over 3 sources (dB)")
ax.set_title(f"Separation quality vs noise ({trials} random mixing matrices)")
ax.invert_xaxis(); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "snr_sweep.png"), dpi=130); plt.close(fig)

# ------------------------------------------------------------------ C: nonlinearity x algorithm
C = {}
n_seeds = 10
for alg in ["symmetric", "deflation"]:
    for fun in ["logcosh", "exp", "cube"]:
        sd, it, tm, fails = [], [], [], 0
        for t in range(n_seeds):
            A = make_mixing_matrix(3, N_SRC, seed=300 + t)
            t0 = time.time()
            _, r, ei, _ = run(S, A, seed=t, algorithm=alg, fun=fun)
            tm.append(time.time() - t0)
            sd.append(ei["si_sdr"].mean())
            it.append(r["n_iter"] if isinstance(r["n_iter"], int) else sum(r["n_iter"]))
            fails += ei["si_sdr"].min() < 15
        C[f"{alg}/{fun}"] = {"mean_si_sdr_db": round(float(np.mean(sd)), 2),
                             "mean_iterations": round(float(np.mean(it)), 1),
                             "mean_time_s": round(float(np.mean(tm)), 3),
                             "runs_with_a_source_below_15dB": int(fails), "runs": n_seeds}
        print("C", alg, fun, C[f"{alg}/{fun}"])
results["C_nonlinearity_algorithm"] = C

# ------------------------------------------------------------------ D: Gaussianity
# Averaged over 10 draws of the Gaussian sources / ICA initialisations, because
# for Gaussian sources the outcome is arbitrary (any rotation is equally valid).
A = make_mixing_matrix(3, N_SRC, seed=1)
D = {}
for label, n_gauss in [("speech+music+ambient (all non-Gaussian)", 0),
                       ("speech+music+1 Gaussian", 1),
                       ("speech+2 Gaussian", 2)]:
    per_src, kurt = [], None
    for t in range(10):
        rng = np.random.default_rng(1000 + t)
        if n_gauss == 0:
            Sx = S
        elif n_gauss == 1:
            Sx = np.vstack([S[:2], rng.standard_normal((1, S.shape[1]))])
        else:
            Sx = np.vstack([S[:1], rng.standard_normal((2, S.shape[1]))])
        _, _, ei, _ = run(Sx, A, seed=t)
        per_src.append(ei["si_sdr"])
        kurt = excess_kurtosis(Sx).round(2).tolist()
    per_src = np.array(per_src)
    D[label] = {"source_excess_kurtosis_last_draw": kurt,
                "mean_si_sdr_db_per_source": per_src.mean(0).round(2).tolist(),
                "min_si_sdr_db_per_source": per_src.min(0).round(2).tolist()}
    print("D", label, D[label])
results["D_gaussianity"] = D
X = A @ S
results["D_clt_kurtosis"] = {"sources": excess_kurtosis(S).round(2).tolist(),
                             "mixtures": excess_kurtosis(X).round(2).tolist()}
print("D CLT  sources", results["D_clt_kurtosis"]["sources"], " mixtures", results["D_clt_kurtosis"]["mixtures"])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
bins = np.linspace(-6, 6, 120)
for ax, (Dm, ttl, labs) in zip(axes, [(S, "sources", NAMES),
                                      (X / X.std(axis=1, keepdims=True), "mixtures", ["mix 1", "mix 2", "mix 3"])]):
    for k, n in enumerate(labs):
        ax.hist(Dm[k], bins=bins, density=True, histtype="step", label=f"{n} (kurt {excess_kurtosis(Dm)[k]:.1f})")
    ax.plot(bins, np.exp(-bins**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8, label="Gaussian")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 2); ax.set_title(f"Amplitude distribution of {ttl}")
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "clt_histograms.png"), dpi=130); plt.close(fig)

# ------------------------------------------------------------------ E: number of mixtures
E = {}
for M in [3, 4, 6]:
    sd = []
    for t in range(10):
        A = make_mixing_matrix(M, N_SRC, seed=400 + t)
        _, _, ei, _ = run(S, A, snr=20, seed=t)
        sd.append(ei["si_sdr"].mean())
    E[f"M={M}"] = round(float(np.mean(sd)), 2)
    print("E", f"M={M}", E[f"M={M}"])
results["E_num_mixtures_at_20dB"] = E

with open(os.path.join(OUT, "experiments.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved outputs/experiments.json")
