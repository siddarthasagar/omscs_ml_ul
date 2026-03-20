# Phase 3 — Design

**Status:** [ ] NOT STARTED

## Algorithm Guidance

### PCA
- Fit full n_components (12 for Wine, 104 for Adult) to capture complete spectrum.
- Collect: explained variance ratio per component, cumulative explained variance.
- Selection rule: smallest n where cumulative variance ≥ 90%.
- Risk: may over-compress Adult's sparse categorical dimensions.

### ICA
- FastICA. Fit with n_components from PCA selection (or sweep).
- Collect: absolute kurtosis per component.
- Selection rule: elbow in sorted kurtosis — components below the elbow have near-Gaussian distributions and contribute little.
- Risk: convergence may be slow or unstable on Adult OHE space. Log convergence warnings.

### Random Projection
- SparseRandomProjection. Sweep seeds 42–51, fixed n_components (from PCA selection).
- Collect: reconstruction error per seed (via pseudo-inverse).
- Selection rule: seed-median reconstruction error plateau — use the n_components at the plateau onset.
- Risk: high run-to-run variance is expected and acceptable; report variance as evidence.

## Figures

| Figure | Content | Function |
|--------|---------|---------|
| `{dataset}_pca_variance.png` | Explained variance + cumulative curve vs component | `plot_pca_variance` |
| `{dataset}_ica_kurtosis.png` | Kurtosis bar chart vs component, sorted descending | `plot_ica_kurtosis` |
| `{dataset}_rp_stability.png` | Reconstruction error vs seed, line or box | `plot_rp_stability` |

## Frozen n_components (locked 2026-03-21)

| Dataset | PCA | ICA | RP |
|---------|-----|-----|----|
| Wine    | 8   | 4   | 8  |
| Adult   | 22  | 11  | 22 |

- **Wine PCA=8:** cumvar=0.937 at component 8 (≥90% threshold).
- **Wine ICA=4:** 4 of 8 components above-median absolute kurtosis.
- **Wine RP=8:** same target dim as PCA; mean recon error=0.356 ± 0.069.
- **Adult PCA=22:** cumvar=0.907 at component 22 out of 104 OHE features.
- **Adult ICA=11:** 11 of 22 components above-median kurtosis.
- **Adult RP=22:** mean recon error=0.107 ± 0.003 (very stable).

## New src/ Functions to Implement

- `src/unsupervised/reduction.py`: `fit_pca`, `fit_ica`, `fit_rp`, `rp_reconstruction_error`
- `src/utils/plotting.py`: `plot_pca_variance`, `plot_ica_kurtosis`, `plot_rp_stability`
