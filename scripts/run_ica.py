"""Main experiment: mix -> centre -> PCA whiten -> FastICA -> export -> evaluate.

Usage
  python scripts/run_ica.py                       # synthetic mix of data/sources/*.wav
  python scripts/run_ica.py --n-mix 4             # more mixtures than sources
  python scripts/run_ica.py --mixtures-dir rec/   # your own real multi-device recordings
                                                  # (no ground truth -> no metrics)
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from ica import amari_index, audio_io, evaluate, excess_kurtosis
from ica.metrics import correlation_matrix
from ica.pipeline import make_mixing_matrix, separate

NAMES = ["speech", "music", "ambient"]


def waveform_grid(rows, titles, sr, path):
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.6 * len(rows)), sharex=True)
    t = np.arange(rows[0].shape[0]) / sr
    for ax, y, ttl in zip(axes, rows, titles):
        ax.plot(t[::10], y[::10], lw=0.4, color="#1f4e79")
        ax.set_ylabel(ttl, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-dir", default=os.path.join(ROOT, "data", "sources"))
    ap.add_argument("--mixtures-dir", default=None)
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--algorithm", default="symmetric", choices=["symmetric", "deflation"])
    ap.add_argument("--fun", default="logcosh", choices=["logcosh", "exp", "cube"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    # Input validation
    if args.mixtures_dir and not os.path.isdir(args.mixtures_dir):
        ap.error(f"--mixtures-dir not found: {args.mixtures_dir}")
    if not args.mixtures_dir and not os.path.isdir(args.sources_dir):
        ap.error(f"--sources-dir not found: {args.sources_dir}")

    for sub in ["mixtures", "separated", "pca_only", "plots"]:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    # ------------------------------------------------------------------ load
    if args.mixtures_dir:  # real recordings, no ground truth
        files = sorted(glob.glob(os.path.join(args.mixtures_dir, "*.wav")))
        data = [audio_io.read_wav(f) for f in files]
        sr = data[0][0]
        T = min(len(x) for _, x in data)
        X = np.array([x[:T] for _, x in data])
        n_src = X.shape[0]
        S_true = None
    else:
        data = [audio_io.read_wav(os.path.join(args.sources_dir, f"{n}.wav")) for n in NAMES]
        sr = data[0][0]
        S_true = np.array([x for _, x in data])
        S_true = S_true / S_true.std(axis=1, keepdims=True)  # unit variance sources
        n_src = S_true.shape[0]
        A = make_mixing_matrix(args.n_mix, n_src)
        X = A @ S_true
        print(
            f"Mixing matrix A ({args.n_mix}x{n_src}), cond = {np.linalg.cond(A):.2f}\n{np.round(A, 3)}"
        )
        for k in range(X.shape[0]):
            audio_io.write_wav(os.path.join(args.out, "mixtures", f"mixture_{k+1}.wav"), X[k], sr)

    # -------------------------------------------------------------- separate
    res = separate(
        X, n_src, algorithm=args.algorithm, fun=args.fun, seed=args.seed, return_history=True
    )
    S_ica, S_pca = res["S_ica"], res["S_pca"]
    print(f"FastICA ({args.algorithm}, {args.fun}) iterations: {res['n_iter']}")

    summary = {
        "algorithm": args.algorithm,
        "nonlinearity": args.fun,
        "n_mix": int(X.shape[0]),
        "n_sources": int(n_src),
        "iterations": res["n_iter"],
    }

    if S_true is None:
        for i in range(n_src):
            audio_io.write_wav(
                os.path.join(args.out, "separated", f"component_{i+1}.wav"), S_ica[i], sr
            )
        waveform_grid(
            list(X) + list(S_ica),
            [f"mix {i+1}" for i in range(len(X))] + [f"ICA {i+1}" for i in range(n_src)],
            sr,
            os.path.join(args.out, "plots", "waveforms.png"),
        )
        print("Exported components (no ground truth available -> no metrics).")
        return

    # -------------------------------------------------------------- evaluate
    ev_ica = evaluate(S_true, S_ica, mixtures=X)
    ev_pca = evaluate(S_true, S_pca)
    P = res["W_total"] @ A
    summary.update(
        {
            "mixing_cond": float(np.linalg.cond(A)),
            "source_excess_kurtosis": dict(zip(NAMES, excess_kurtosis(S_true).round(2).tolist())),
            "ica": {
                n: {"abs_corr": round(float(c), 4), "si_sdr_db": round(float(s), 2)}
                for n, c, s in zip(NAMES, ev_ica["corr"], ev_ica["si_sdr"])
            },
            "pca_only": {
                n: {"abs_corr": round(float(c), 4), "si_sdr_db": round(float(s), 2)}
                for n, c, s in zip(NAMES, ev_pca["corr"], ev_pca["si_sdr"])
            },
            "best_raw_mixture_si_sdr_db": dict(
                zip(NAMES, ev_ica["si_sdr_best_mixture"].round(2).tolist())
            ),
            "amari_index_ica": round(float(amari_index(P)), 5),
            "mean_si_sdr_ica_db": round(float(ev_ica["si_sdr"].mean()), 2),
            "mean_si_sdr_pca_db": round(float(ev_pca["si_sdr"].mean()), 2),
        }
    )
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ----------------------------------------------------------- export audio
    for i, n in enumerate(NAMES):
        audio_io.write_wav(
            os.path.join(args.out, "separated", f"ica_{n}.wav"), ev_ica["aligned"][i], sr
        )
        audio_io.write_wav(
            os.path.join(args.out, "pca_only", f"pca_{n}.wav"), ev_pca["aligned"][i], sr
        )

    # ------------------------------------------------------------------ plots
    p = os.path.join(args.out, "plots")
    waveform_grid(
        list(S_true) + list(X) + list(ev_pca["aligned"]) + list(ev_ica["aligned"]),
        [f"src: {n}" for n in NAMES]
        + [f"mix {i+1}" for i in range(len(X))]
        + [f"PCA: {n}" for n in NAMES]
        + [f"ICA: {n}" for n in NAMES],
        sr,
        os.path.join(p, "waveforms.png"),
    )

    # joint distributions (speech vs ambient) at each stage -> shows the rotation ICA finds
    rng = np.random.default_rng(0)
    sub = rng.choice(S_true.shape[1], 6000, replace=False)
    i, j = 0, 2
    stages = [
        ("Sources (independent)", S_true),
        ("Mixtures (correlated)", X),
        ("After PCA whitening\n(uncorrelated, still mixed)", S_pca),
        ("After ICA rotation\n(independent again)", ev_ica["aligned"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7))
    for ax, (ttl, D) in zip(axes, stages):
        a, b = D[i, sub], D[j, sub]
        ax.scatter(a, b, s=1.2, alpha=0.35, color="#1f4e79")
        lim = np.percentile(np.abs(np.r_[a, b]), 99.7) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Geometry of ICA: components 1 vs 3 (speech vs ambient)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(p, "scatter_stages.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # correlation heatmaps: sources vs mixtures / PCA / ICA
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (ttl, D, labels) in zip(
        axes,
        [
            ("sources vs mixtures", X, [f"mix {k+1}" for k in range(len(X))]),
            ("sources vs PCA comps", S_pca, [f"PC{k+1}" for k in range(n_src)]),
            ("sources vs ICA comps", S_ica, [f"IC{k+1}" for k in range(n_src)]),
        ],
    ):
        C = np.abs(correlation_matrix(S_true, D))
        ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(C.shape[1]))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(n_src))
        ax.set_yticklabels(NAMES)
        for a in range(C.shape[0]):
            for b in range(C.shape[1]):
                ax.text(
                    b,
                    a,
                    f"{C[a, b]:.2f}",
                    ha="center",
                    va="center",
                    color="w" if C[a, b] < 0.6 else "k",
                    fontsize=9,
                )
        ax.set_title("|corr| " + ttl, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(p, "correlation_heatmaps.png"), dpi=130)
    plt.close(fig)

    # convergence
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.semilogy(res["history"], color="#c0392b")
    ax.axhline(1e-6, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("1 - |<w_new, w_old>| (max over rows)")
    ax.set_title("FastICA convergence (symmetric)")
    fig.tight_layout()
    fig.savefig(os.path.join(p, "convergence.png"), dpi=130)
    plt.close(fig)

    # spectrograms: truth vs ICA output
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
    for k, n in enumerate(NAMES):
        for r, (D, lab) in enumerate(
            [(S_true[k], "original"), (ev_ica["aligned"][k], "ICA output")]
        ):
            axes[r, k].specgram(D, NFFT=512, Fs=sr, noverlap=384, cmap="magma", vmin=-110)
            axes[r, k].set_title(f"{n} - {lab}", fontsize=10)
            axes[r, k].set_xlabel("time (s)" if r else "")
            axes[r, k].set_ylabel("Hz" if k == 0 else "")
    fig.tight_layout()
    fig.savefig(os.path.join(p, "spectrograms.png"), dpi=130)
    plt.close(fig)
    print("Saved outputs to", args.out)


if __name__ == "__main__":
    main()
