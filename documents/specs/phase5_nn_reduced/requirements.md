# Phase 5 — Requirements

**Status:** [ ] NOT STARTED — blocked on Phase 3

## What This Phase Must Produce

1. WineNN trained on 4 input variants: raw (12 features), PCA-reduced, ICA-reduced, RP-reduced.
2. 10 seeds (42–51) per variant → 40 training runs total.
3. Comparison table and per-run history CSVs.
4. Figures: F1 boxplot and learning curves.

## Acceptance Criteria

- SHALL produce `comparison_table.csv` with cols: variant, seed, val_macro_f1, test_macro_f1, epochs_to_converge (40 rows).
- SHALL produce per-run history CSVs: `{variant}_seed{seed}_history.csv` with cols: epoch, train_loss, val_loss, train_f1, val_f1.
- SHALL use identical optimizer, lr, batch_size, max_epochs across all variants — only input_dim changes (Gate 3 check).
- SHALL NOT use early stopping in baseline runs.
- SHALL produce `phase5_f1_boxplot.png` (4 boxes, one per variant).
- SHALL produce `{variant}_learning_curves.png` per variant (mean ± std across seeds).
- Wine ONLY — Adult is excluded from Steps 4 and 5 (see steering/product.md).

## Output Locations

| Artifact | Path |
|----------|------|
| CSVs | `artifacts/metrics/phase5_nn_reduced/` |
| Figures | `artifacts/figures/phase5_nn_reduced/` |
| Log | `artifacts/logs/phase5_<ts>.log` |

## Pre-requisites

- Phase 3 complete with frozen n_components.
- `src/supervised/nn_baseline.py` and `src/supervised/training.py` implemented.
