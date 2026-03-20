# Phase 3 — Tasks

**Status:** ✅ ALL COMPLETE (2026-03-21)

## Implementation

- [x] Implement `fit_pca(X_train, n_components=None)` in `src/unsupervised/reduction.py`
- [x] Implement `fit_ica(X_train, n_components, seed)` in `src/unsupervised/reduction.py`
- [x] Implement `fit_rp(X_train, n_components, seed)` in `src/unsupervised/reduction.py`
- [x] Implement `rp_reconstruction_error(rp, X)` in `src/unsupervised/reduction.py`
- [x] Implement `plot_pca_variance(df, dataset, out_dir)` in `src/utils/plotting.py`
- [x] Implement `plot_ica_kurtosis(df, dataset, out_dir)` in `src/utils/plotting.py`
- [x] Implement `plot_rp_stability(df, dataset, out_dir)` in `src/utils/plotting.py`
- [x] Write `scripts/run_phase_3_raw_reduction.py` with logger wired

## Execution

- [x] Run PCA on Wine and Adult — `wine_pca.csv`, `adult_pca.csv`
- [x] Run ICA on Wine and Adult — `wine_ica.csv`, `adult_ica.csv`
- [x] Run RP stability sweep (seeds 42–51) — `wine_rp_stability.csv`, `adult_rp_stability.csv`
- [x] Produce 6 figures in `artifacts/figures/phase3_reduction/`

## Selection (label-free)

- [x] Wine: PCA=8, ICA=4, RP=8 — recorded in design.md
- [x] Adult: PCA=22, ICA=11, RP=22 — recorded in design.md
