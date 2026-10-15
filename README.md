<div align="center">

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/8975f18e-2100-4bae-b7f4-0280dfd2ee2d" />


# Unmix the Signal

**Three voices go in. One tangled mess comes out. ICA pulls them apart again.**

*Blind audio source separation with PCA whitening and FastICA, hand-built in pure NumPy.*

[![CI](https://github.com/visshalb2007/ica_unmix/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/visshalb2007/ica_unmix/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
</div>

---

## The Cocktail Party Problem

You're at a party. Three people are talking at once. You place three microphones around the room, and each one records a *different blend* of all three voices.

**Can you recover each speaker using only those blends, with no idea how they were mixed?**

Yes. And that's what this repo does.

We take three clean sources (**speech**, **music**, and **ambient sound**), scramble them with a secret mixing matrix, and then un-scramble them with a **FastICA** algorithm written from scratch. No black-box estimators. Just covariance matrices, eigenvectors, and a fixed-point iteration that hunts for non-Gaussianity.

---

## The Scoreboard

| Metric | FastICA | PCA-only |
|---|:---:|:---:|
| **Mean SI-SDR** | **44.8 dB** | 1.4 dB |
| **Best-match correlation** | **≥ 0.9999** | ~0.71 – 0.80 |

PCA decorrelates the signals but leaves them mixed. ICA goes further, using higher-order statistics to find the *independent* sources.

> **Note:** These numbers come from the included **synthetic, noiseless** run. Real-world recordings will behave differently, so see [Know the Limits](#know-the-limits).

---

## The Pipeline

```text
  Clean Sources
  Speech | Music | Ambient
            |
            v
  Mixing Matrix (A)
            |
            v
  Mixed Recordings (X = AS)
            |
            v
  Centering
            |
            v
  PCA Whitening
  covariance + eigendecomposition
            |
            v
  FastICA from Scratch
  symmetric | deflation | logcosh, exp, cube
            |
            v
  Recovered Sources (S_hat = WX)
            |
            v
  Evaluation + Visualization
  Correlation | SI-SDR | Amari index | Plots
```

---

## What's Inside

- **FastICA, built by hand**: symmetric *and* deflationary modes
- **Manual PCA whitening**: covariance and eigendecomposition, no `sklearn`
- **Three contrast functions**: `logcosh`, `exp`, `cube`
- **Playable WAVs**: listen to the mixtures and the separated sources
- **Real metrics**: SI-SDR, absolute correlation, Amari index (with automatic source matching)
- **A PCA baseline** to show exactly what ICA adds
- **Noise torture tests**: separation quality from 60 dB down to −5 dB SNR
- **Gaussianity experiments**: see *why* ICA fails on Gaussian sources
- **Plots galore**: waveforms, spectrograms, heatmaps, convergence curves, scatter stages, histograms
- **Automated tests and CI**

### No Shortcuts

NumPy and SciPy provide the raw numerical muscle. Centering, whitening, and ICA itself are all implemented in this repo. **There is no call to a ready-made ICA or PCA estimator anywhere.**

---

## Run It

```bash
# grab the code
git clone https://github.com/visshalb2007/ica_unmix.git
cd ica_unmix

# set up an environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# install and unmix
python -m pip install -r requirements.txt
python scripts/run_ica.py
```

That one command mixes the sources, separates them, writes the WAV files, scores the result, and draws every plot into `outputs/`. Open the WAVs in any audio player and hear it work.

> **Windows tip:** if PowerShell blocks activation, call `.venv\Scripts\python.exe` directly.

Want to go deeper?

```bash
python scripts/experiments.py        # the full experiment suite
python tests/test_core.py            # run the tests
```

---

## Knobs and Dials

```bash
python scripts/run_ica.py [options]
```

| Flag | Default | What it does |
|---|---|---|
| `--sources-dir` | `data/sources` | Folder with the clean source WAVs |
| `--mixtures-dir` | *none* | Folder with your own pre-recorded mixtures |
| `--n-mix` | `3` | Number of synthetic mixture channels |
| `--algorithm` | `symmetric` | `symmetric` or `deflation` |
| `--fun` | `logcosh` | `logcosh`, `exp`, or `cube` |
| `--seed` | `0` | Random seed |
| `--out` | `outputs` | Where results go |

```bash
python scripts/run_ica.py --n-mix 4              # extra microphones
python scripts/run_ica.py --algorithm deflation  # one component at a time
python scripts/run_ica.py --fun exp              # swap the nonlinearity
python scripts/run_ica.py --out outputs/custom   # tidy output folder
```

---

## How the Magic Works

<details>
<summary><b>1. The mixing model</b></summary>

<br>

ICA assumes every recording is an instantaneous linear blend of independent sources:

$$X = AS$$

$S$ is the sources, $A$ is the unknown mixing matrix, and $X$ is what the microphones hear. The mission is to find $W$ so that

$$\hat{S} = WX$$

</details>

<details>
<summary><b>2. Centering</b></summary>

<br>

Subtract the mean from every channel to remove offsets:

$$X_c = X - \mathbb{E}[X]$$

</details>

<details>
<summary><b>3. PCA whitening</b></summary>

<br>

Compute the covariance and eigendecompose it:

$$C = \frac{1}{T}X_cX_c^\top = EDE^\top$$

Then build the whitening transform and apply it:

$$K = D^{-1/2}E^\top, \qquad Z = KX_c$$

After this, the signals have (approximately) identity covariance. All that's left to find is a **rotation**, and that's exactly what FastICA searches for.

</details>

<details>
<summary><b>4. FastICA</b></summary>

<br>

FastICA looks for directions in whitened space that are **as non-Gaussian as possible**, using fixed-point updates.

| Mode | Strategy |
|---|---|
| **Symmetric** | Estimate all components together, decorrelating iteratively |
| **Deflationary** | Peel off one component at a time, keeping each orthogonal to the last |

Why does this work? By the Central Limit Theorem, a *mixture* of signals looks more Gaussian than any single source. So hunting for maximum non-Gaussianity hunts down the originals.

Uncorrelated does **not** mean independent, and that gap is where ICA beats PCA.

</details>

<details>
<summary><b>5. Scoring the result</b></summary>

<br>

ICA recovers sources only up to **scale, sign, and permutation**, so components are matched to references before scoring:

- **Absolute Pearson correlation**: how closely the matched waveforms agree
- **SI-SDR**: target energy versus residual error, independent of global scale
- **Amari index**: how close the overall matrix $WA$ is to a scaled permutation matrix

</details>

---

## Project Map

```text
ica_unmix/
├── ica/
│   ├── audio_io.py          # read / write WAV
│   ├── preprocessing.py     # centering + PCA whitening
│   ├── fastica.py           # the algorithm, nonlinearities, orthogonalization
│   ├── metrics.py           # correlation, matching, SI-SDR, Amari
│   └── pipeline.py          # mix -> preprocess -> separate
├── data/sources/            # speech.wav, music.wav, ambient.wav
├── scripts/
│   ├── make_sources.py      # synthesize the sources
│   ├── run_ica.py           # main pipeline + plots
│   └── experiments.py       # extended experiments
├── tests/test_core.py       # numerical + recovery tests
├── outputs/                 # everything generated
├── .github/workflows/ci.yml
├── REPORT.md                # deep dive: math, methodology, findings
└── README.md
```

---

## What You Get Back

| Path | What's there |
|---|---|
| `outputs/mixtures/` | The scrambled recordings |
| `outputs/separated/` | Sources recovered by ICA |
| `outputs/pca_only/` | The PCA-only baseline |
| `outputs/plots/waveforms.png` | Original vs. mixed vs. recovered |
| `outputs/plots/spectrograms.png` | Time-frequency view |
| `outputs/plots/correlation_heatmaps.png` | ICA vs. PCA correlations |
| `outputs/plots/convergence.png` | FastICA settling in |
| `outputs/plots/scatter_stages.png` | Data shape through each stage |
| `outputs/plots/clt_histograms.png` | The Central Limit Theorem in action |
| `outputs/plots/snr_sweep.png` | Quality vs. noise level |
| `outputs/metrics.json` | Main scores |
| `outputs/experiments.json` | Experiment results |

Full tables, derivations, and findings live in **[REPORT.md](REPORT.md)**.

---

## Bring Your Own Recordings

Got real mixtures? Feed them in.

1. Record the same sources at the same time on separate devices
2. Sync them so samples line up in time
3. Convert to mono WAV at a shared sample rate
4. Drop them in a folder and run:

```bash
python scripts/run_ica.py \
  --mixtures-dir path/to/recordings \
  --out outputs_real
```

Without clean reference sources you'll get the recovered components and waveform plots, but no ground-truth scores.

> **Heads up:** real rooms add echoes and delays that break the instantaneous-mixing assumption, so expect rougher results than the synthetic demo.

---

## Know the Limits

- Sources must be statistically independent and mixed instantaneously and linearly
- Recoverable sources are capped by the number of independent mixture channels
- Output is identifiable only up to scale, sign, and permutation
- At most **one** source can be Gaussian
- Experiments use synthetic sources, not natural speech, music, or field recordings
- Room reverberation (convolutive mixing) is not modeled
- Ground-truth metrics need clean reference sources
- The optional mini-Shazam extension is not implemented

---

## References

1. A. Hyvärinen & E. Oja, *Independent Component Analysis: Algorithms and Applications*, Neural Networks, 2000.
2. A. Hyvärinen, *Fast and Robust Fixed-Point Algorithms for Independent Component Analysis*, IEEE Trans. Neural Networks, 1999.
3. P. Comon, *Independent Component Analysis, A New Concept?*, Signal Processing, 1994.
4. J. Le Roux et al., *SDR: Half-baked or Well Done?*, IEEE ICASSP, 2019.

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

<div align="center">

**∿∿∿ mix it. scramble it. unmix it. ∿∿∿**

*If this made you hear signal processing differently, drop a star.*

</div>
