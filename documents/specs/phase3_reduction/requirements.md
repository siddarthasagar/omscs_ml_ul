# Phase 3 — Requirements

**Status:** [ ] NOT STARTED — unblocked (Phase 2 complete)

## What This Phase Must Produce

1. PCA, ICA, and Random Projection fit on `X_train` for Wine and Adult.
2. Per-method diagnostic CSVs for component selection (label-free).
3. One frozen `n_components` per method per dataset.
4. Diagnostic figures for each method.

## Acceptance Criteria

- SHALL produce `{dataset}_pca.csv` with cols: component, explained_variance, cumulative_variance.
- SHALL produce `{dataset}_ica.csv` with cols: component, kurtosis.
- SHALL produce `{dataset}_rp_stability.csv` with cols: seed, n_components, reconstruction_error (seed sweep 42–51).
- SHALL produce 6 figures (one per method per dataset).
- SHALL NOT use labels during n_components selection.
- SHALL log all artifact paths and frozen selections to `artifacts/logs/phase3_<ts>.log`.
- Frozen n_components SHALL be recorded in `tasks.md` of this spec once selected.

## Output Locations

| Artifact | Path |
|----------|------|
| CSVs | `artifacts/metrics/phase3_reduction/` |
| Figures | `artifacts/figures/phase3_reduction/` |
| Log | `artifacts/logs/phase3_<ts>.log` |

## Pre-requisites

- Phase 2 complete (data loaders and preprocessing validated).
- `src/utils/logger.py` wired (done — ADR-003).
