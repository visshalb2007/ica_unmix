# Unmixing audio with FastICA

Three clean sources (speech, music, rain-like ambience, 10 s each at 16 kHz) are mixed with a random 3x3 matrix, then recovered with a centering, PCA-whitening and FastICA pipeline written in NumPy. No library ICA or PCA is used. `scipy` appears in only two places: reading and writing WAV files, and the Hungarian algorithm that pairs each estimate with its true source during evaluation.

Everything below can be regenerated with `python scripts/make_sources.py`, `python scripts/run_ica.py` and `python scripts/experiments.py`.

## Main result

Mixing matrix condition number 8.09, symmetric FastICA with the logcosh non-linearity, converged in 6 iterations.

| source | excess kurtosis | ICA |corr| | ICA SI-SDR | PCA-only SI-SDR | best raw mixture SI-SDR |
|---|---|---|---|---|---|
| speech | 12.5 | 1.0000 | 41.7 dB | 1.5 dB | -0.6 dB |
| music | 1.0 | 0.9999 | 39.7 dB | 0.1 dB | 2.2 dB |
| ambient | 34.6 | 1.0000 | 53.1 dB | 2.6 dB | -1.8 dB |

Mean SI-SDR is 44.8 dB for ICA and 1.4 dB for PCA. The Amari index of the total unmixing matrix times A is 0.004 (0 is perfect).

Audio is in `outputs/separated/` (ICA), `outputs/pca_only/` (the PCA components, for comparison) and `outputs/mixtures/` (the three mixtures the algorithm saw).

![waveforms](outputs/plots/waveforms.png)

![spectrograms](outputs/plots/spectrograms.png)

![correlation heatmaps](outputs/plots/correlation_heatmaps.png)

The heatmap on the right shows the permutation ambiguity directly: IC1 is the ambient source, IC2 is speech and IC3 is music. The algorithm has no way to know the order, so evaluation has to match them up first.

## 1. Why ICA and not PCA

PCA finds orthogonal directions of maximum variance, so it gives components that are uncorrelated. Uncorrelated is a second-order property (it only involves the covariance matrix). Independent is much stronger: the joint distribution has to factor into the product of the marginals, which constrains every higher-order moment too. For Gaussian data the two coincide, and for everything else they don't.

The practical consequence is that second-order statistics cannot recover the sources. If Z is whitened, then RZ for any rotation R is also white. PCA has no preference among those rotations, so it stops at an arbitrary one. ICA picks the rotation by looking at higher-order statistics (here, approximate negentropy built from tanh, which responds to kurtosis and beyond).

The scatter plot below shows this with speech against ambient. The sources form a cross because each one is mostly near zero independently of the other. The mixtures are a tilted blob. After PCA whitening the blob is round-ish but still tilted, which is what "uncorrelated but still mixed" looks like. The ICA rotation restores the cross.

![scatter](outputs/plots/scatter_stages.png)

Numerically, PCA components correlate 0.71 to 0.80 with their best-matching source and score about 1.4 dB. Over 20 different mixing matrices PCA never exceeded 3.6 dB.

## 2. What whitening does and why it makes ICA simpler

Whitening applies K = D^(-1/2) E^T, where C = E D E^T is the eigendecomposition of the covariance of the centered mixtures. After it, Z = KAS has identity covariance. The sources also have identity covariance (we scaled them to unit variance), so (KA)(KA)^T = I, which means KA is an orthogonal matrix.

That reduces the search. An unconstrained 3x3 unmixing matrix has 9 free parameters. After whitening only a rotation is left, which for 3 dimensions has 3 parameters. In the code this shows up as W being kept orthogonal at every step (symmetric decorrelation, or Gram-Schmidt in deflation mode), and `tests/test_core.py` checks that W W^T = I after fitting.

Whitening is also where we reduce dimension when there are more mixtures than sources: keeping only the top N eigenvectors discards the directions that hold only noise. Experiment E below uses this.

Centering comes first because the covariance and the fixed-point updates both assume zero mean, and the mean of the mixtures is just the mixing of the means, which carries no separation information.

## 3. Why non-Gaussianity enables separation, and the CLT link

A whitened Gaussian vector is spherically symmetric, so every rotation of it looks the same and there is nothing to identify. This is the content of the identifiability result for ICA (Comon 1994): the model is identifiable up to scale, sign and order provided at most one source is Gaussian.

The Central Limit Theorem gives the algorithm its objective. A sum of independent random variables is closer to Gaussian than the variables themselves. A mixture channel is a weighted sum of sources, so a direction w that makes w^T z as non-Gaussian as possible is a direction that undoes as much mixing as possible. FastICA maximizes an approximation of negentropy, J(y) = [E G(y) - E G(v)]^2 with v a standard Gaussian, using a Newton-like fixed-point step, w+ = E[z g(w^T z)] - E[g'(w^T z)] w.

Two checks in this project:

**Kurtosis of sources vs mixtures** (`outputs/plots/clt_histograms.png`). Source excess kurtosis is 12.5, 1.0 and 34.6. The three mixtures have 5.0, 2.0 and 8.2. The mixtures are clearly less heavy-tailed than speech and ambient. The music mixture channel is above the music source's 1.0, which is allowed: a mixture of one near-Gaussian source with two heavy-tailed ones can sit between them. The CLT says the trend is toward Gaussian, not that every mixture is below every source.

**Deliberately Gaussian sources** (experiment D). With all three real sources, each is recovered at 40 dB or better. Replacing one with Gaussian noise leaves the other two intact (42.0 and 38.6 dB), which matches the "at most one Gaussian" condition. Replacing two makes the pair unrecoverable.

## 4. How the experiments and metrics were chosen

**Metrics.** ICA returns each source with arbitrary scale, sign and position, so any metric has to be invariant to those or fix them first.

- Absolute Pearson correlation after Hungarian matching is the first-look number. It is scale and sign invariant. Its weakness is saturation: 0.999 and 0.99999 are both "1.0" at two decimals, and they sound different.
- SI-SDR (Le Roux et al., 2019) fixes that. It projects the estimate onto the reference, so rescaling the estimate changes nothing, and reports the ratio of the matched part to the leftover error in dB. I used it as the headline metric. Plain SDR would need the scale resolved separately.
- The Amari index on W_total A needs no ground-truth audio alignment and tests how close the global matrix is to a scaled permutation.
- A baseline is reported next to each result (best raw mixture channel, PCA-only), because a number like "41 dB" only means something against what you get with no separation.

Exported audio is peak-normalized to 0.9 and sign-flipped to match the original. These two steps use ground truth, but only for presentation of the output, and they do not change any metric.

**Experiments.**

- A checks that the main result is not an artifact of one lucky mixing matrix.
- B is the "separation quality across SNR levels" bonus, with an oracle line added so the loss from noise can be told apart from the loss from ICA.
- C compares non-linearities and the two FastICA variants, because those are the design choices in the algorithm.
- D ties directly to question 3.
- E covers more mixtures than sources.

### A. PCA vs ICA over 20 mixing matrices

ICA scored 44.82 dB on every one of the 20 matrices (min equals mean). PCA averaged 1.4 dB with a best case of 3.6 dB. The identical ICA number is expected: FastICA is equivariant, so with noiseless mixtures the estimated sources do not depend on A. The ~45 dB ceiling comes from the finite sample (the three signals are not exactly independent over 10 seconds; their sample correlations are about 0.001 to 0.006) rather than from the algorithm failing.

### B. Noise

White Gaussian noise added to the mixtures, 10 mixing matrices per level.

| mixture SNR | 60 | 40 | 30 | 20 | 10 | 5 | 0 | -5 |
|---|---|---|---|---|---|---|---|---|
| ICA (dB) | 43.1 | 30.3 | 20.6 | 10.7 | 1.6 | -2.9 | -7.4 | -12.8 |
| oracle, true A inverted (dB) | 50.6 | 30.6 | 20.6 | 10.6 | 0.6 | -4.4 | -9.4 | -14.4 |
| PCA only (dB) | 0.4 | 0.4 | 0.3 | -0.4 | -3.4 | -6.2 | -9.3 | -12.7 |

![snr](outputs/plots/snr_sweep.png)

From 40 dB down to 20 dB ICA is indistinguishable from the oracle, so the drop is the noise itself being amplified by inverting a matrix with condition number around 8, not an ICA weakness. At 60 dB ICA is 7 dB under the oracle, which is the finite-sample ceiling from experiment A. Below 10 dB ICA is slightly better than the oracle; my guess is that whitening estimated from noisy data shrinks the weak directions, which acts like a mild regularization that the plain inverse lacks. I did not test that explanation.

PCA is flat near 0 dB until the noise dominates, and the three curves meet at about -5 dB.

### C. Non-linearity and algorithm (10 seeds each)

| algorithm / g | mean SI-SDR | iterations | time |
|---|---|---|---|
| symmetric / logcosh | 44.8 dB | 6.0 | 0.13 s |
| symmetric / exp | 45.8 dB | 6.7 | 0.13 s |
| symmetric / cube | 39.6 dB | 4.5 | 0.28 s |
| deflation / logcosh | 47.3 dB | 10.5 (total) | 0.08 s |
| deflation / exp | 50.1 dB | 10.8 (total) | 0.10 s |
| deflation / cube | 38.9 dB | 8.8 (total) | 0.22 s |

No variant failed (no run had a source below 15 dB). The kurtosis-based cube non-linearity is 5 to 11 dB worse than the other two, which fits its known sensitivity to outliers: the ambient source has kurtosis 34.6, and y^3 is dominated by the few largest samples. The deflation variants scored a few dB higher here; I did not run enough different source sets to say whether that holds in general. The cube timing is higher in absolute terms probably because of how NumPy computes y**3 and y**2 on 160,000 samples, not because of the algorithm.

I kept symmetric/logcosh as the default for the main run because it is the standard choice and does not accumulate error across components the way deflation can.

### D. Gaussian sources (10 draws each, per-source SI-SDR)

| sources | mean SI-SDR per source | worst case per source |
|---|---|---|
| speech, music, ambient | 41.7, 39.7, 53.1 | 41.7, 39.7, 53.1 |
| speech, music, 1 Gaussian | 42.0, 38.6, 43.7 | 41.6, 36.3, 39.9 |
| speech, 2 Gaussians | 53.3, 6.4, 6.4 | 44.7, 0.6, 0.6 |

Speech is still recovered when the other two are Gaussian (it is the only non-Gaussian one, and the Gaussian pair is confined to a 2D subspace). The two Gaussians inside that subspace come out at a random rotation, so the result is erratic: 0.6 dB in the worst draw, 6.4 dB on average. My first single-draw run of this happened to give 13 dB for both, which is why this version averages over 10 draws.

### E. More mixtures than sources (20 dB SNR, 10 matrices)

| mixtures M | 3 | 4 | 6 |
|---|---|---|---|
| mean SI-SDR | 12.2 dB | 13.2 dB | 14.9 dB |

More microphones help in noise, because PCA whitening keeps the 3 strongest directions and averages the noise out over the rest. (These matrices differ from experiment B's, so the M=3 value is not the same as B's 10.7 dB at 20 dB.)

## Limitations

- The sources are synthetic: espeak-ng text-to-speech, additive-synthesis music and rain made from filtered random impulses. I generated them so the repo is self-contained and the license is clean. They are not natural recordings, and the speech in particular is more regular than a human voice.
- Mixing is instantaneous (a matrix multiply). Real rooms add delay and reverb, which makes the mixing convolutive and breaks the model that this FastICA assumes. The bonus item on real multi-device recordings is therefore not done. `run_ica.py --mixtures-dir` will run the pipeline on your own multi-channel WAVs, but without ground truth it only exports components and cannot report metrics.
- The ~45 dB scores are for noiseless instantaneous mixing and say little about what to expect on real recordings. Experiment B is the more realistic indicator.
- Mini-Shazam stretch goal is not attempted.
- Sample-rate and length are the same for every source; the code does not resample.
