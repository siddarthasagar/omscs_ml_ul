# Phase 4 — Requirements

**Status:** [ ] NOT STARTED — blocked on Phase 3

## What This Phase Must Produce

1. Clustering (KMeans + GMM) re-run on all 6 reduced spaces (2 datasets × 3 DR methods).
2. Summary comparison table: raw vs reduced clustering metrics.
3. Figures comparing clustering quality across representations.

## Acceptance Criteria

- SHALL produce `summary_table.csv` with exactly 12 rows (2 datasets × 3 DR methods × 2 clusterers).
- Summary table SHALL have cols: dataset, dr_method, clusterer, silhouette, calinski_harabasz, davies_bouldin, bic.
- SHALL re-select K in each reduced space using the same label-free criteria as Phase 2:
  KMeans by joint silhouette/CH/DB (highest silhouette, CH tiebreaker, DB tiebreaker);
  GMM by BIC minimum. Raw-space frozen K values are NOT reused here.
- SHALL use frozen n_components from Phase 3 design.md — no re-selection.
- SHALL produce at least 2 figures comparing raw vs reduced clustering.
- SHALL log all 12 combinations to `artifacts/logs/phase4_<ts>.log`.

## Output Locations

| Artifact | Path |
|----------|------|
| CSVs | `artifacts/metrics/phase4_reduced_clustering/` |
| Figures | `artifacts/figures/phase4_clustering/` |
| Log | `artifacts/logs/phase4_<ts>.log` |

## Pre-requisites

- Phase 3 complete with frozen n_components recorded in phase3 design.md.
