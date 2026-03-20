# Phase 4 — Design

**Status:** [ ] NOT STARTED

## Combinations

12 total: 2 datasets × 3 DR methods × 2 clusterers.

| # | Dataset | DR Method | Clusterer |
|---|---------|-----------|-----------|
| 1 | Wine | PCA | KMeans (k=2) |
| 2 | Wine | PCA | GMM (n=7) |
| 3 | Wine | ICA | KMeans (k=2) |
| 4 | Wine | ICA | GMM (n=7) |
| 5 | Wine | RP | KMeans (k=2) |
| 6 | Wine | RP | GMM (n=7) |
| 7 | Adult | PCA | KMeans (k=8) |
| 8 | Adult | PCA | GMM (n=7) |
| 9 | Adult | ICA | KMeans (k=8) |
| 10 | Adult | ICA | GMM (n=7) |
| 11 | Adult | RP | KMeans (k=8) |
| 12 | Adult | RP | GMM (n=7) |

K values from ADR-002. n_components from Phase 3 design.md (locked 2026-03-21):
- Wine: PCA=8, ICA=4, RP=8
- Adult: PCA=22, ICA=11, RP=22

## Narrative Goal

For each combination: did the reduced representation improve, maintain, or degrade clustering quality relative to the raw-space baseline from Phase 2? Answer must cite specific metric differences, not just direction.

## Figures

| Figure | Content | Function |
|--------|---------|---------|
| `phase4_clustering_heatmap.png` | Heatmap of silhouette across 12 combinations | `plot_phase4_heatmap` |
| `{dataset}_phase4_bar.png` | Grouped bar: raw vs PCA vs ICA vs RP per clusterer | `plot_phase4_comparison` |
