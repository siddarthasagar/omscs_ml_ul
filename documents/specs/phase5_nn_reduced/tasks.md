# Phase 5 — Tasks

**Status:** [x] COMPLETE — 2026-03-21

- [x] Implement `WineNN(input_dim)` in `src/supervised/nn_baseline.py` (carried from Phase 0)
- [x] Implement `train_wine_nn(X_train, y_train, X_val, y_val, seed)` in `src/supervised/training.py`
- [x] Implement `plot_f1_comparison(df, out_dir)` in `src/utils/plotting.py`
- [x] Implement `plot_learning_curves(history_df, variant, out_dir)` in `src/utils/plotting.py`
- [x] Write `scripts/run_phase_5_nn_reduced.py` with logger wired
- [x] input_dims locked in design.md: raw=12, pca=8, ica=4, rp=8
- [x] Ran 40 training runs (4 variants × 10 seeds) — comparison_table.csv produced
- [x] Produced 40 history CSVs in artifacts/metrics/phase5_nn_reduced/{variant}/
- [x] Produced phase5_f1_boxplot.png and phase5_learning_curves.png (all variants overlaid, 1 figure)
- [x] Gate 3 passed: identical lr/betas/wd/batch/epochs across all variants; only input_dim differs
- [ ] Draft narrative: does DR help or hurt Wine NN? (for report)
