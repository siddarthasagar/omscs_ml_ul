# Phase 2 — Requirements

**Status:** ✅ COMPLETE (2026-03-20)

## What This Phase Must Produce

1. K-Means sweep over k=2..20 on `X_train` for Wine and Adult, using `SEED_EXPLORE=42`.
2. GMM sweep over n=2..20 on `X_train` for Wine and Adult, using `SEED_EXPLORE=42`.
3. One frozen K per dataset per algorithm, selected **without labels**.
4. Metric CSVs and diagnostic figures persisted to artifacts.

## Acceptance Criteria

- SHALL produce `wine_kmeans.csv` with cols: k, inertia, silhouette, calinski_harabasz, davies_bouldin.
- SHALL produce `wine_gmm.csv` with cols: n_components, bic, aic, silhouette.
- SHALL produce `adult_kmeans.csv` and `adult_gmm.csv` with the same schemas.
- SHALL produce 4 PNG figures (2×2 KMeans plot, 1×2 GMM plot per dataset).
- SHALL NOT use labels during K selection.
- SHALL log all artifact paths to `artifacts/logs/phase2_<ts>.log`.
- Frozen K values SHALL be documented in ADR-002.

## Output Locations

| Artifact | Path |
|----------|------|
| CSVs | `artifacts/metrics/phase2_clustering/` |
| Figures | `artifacts/figures/phase2_clustering/` |
| Log | `artifacts/logs/phase2_<ts>.log` |
