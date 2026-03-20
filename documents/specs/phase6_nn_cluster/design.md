# Phase 6 — Design

**Status:** [ ] NOT STARTED

## Feature Engineering

All three variants append cluster-derived features to the original 12 Wine features. Cluster models are fit on `X_train` using frozen K from ADR-002. Features are appended before the train/val/test split is used for NN training — the cluster model itself is fit on `X_train` only.

| Variant | Derived feature | Shape | Note |
|---------|----------------|-------|------|
| `kmeans_onehot` | One-hot KMeans assignment | (n, 2) | Hard assignment |
| `kmeans_dist` | Distance to each centroid | (n, 2) | Soft distance measure |
| `gmm_posterior` | GMM posterior P(component \| x) | (n, 7) | Soft probabilistic |

## Architecture

Same `WineNN(input_dim)` from Phase 5 with varying input_dim. Training config identical to Phase 5 (see steering/tech.md).

## Stretch Variant (optional, post-baseline)

Cluster-only features (no raw 12) — test if cluster features alone are sufficient. Only implement if appended results are ambiguous.

## Figures

| Figure | Content | Function |
|--------|---------|---------|
| `phase6_f1_boxplot.png` | Boxplot per variant (30 runs), Phase 5 raw baseline overlaid | `plot_f1_comparison(df, out_dir, baseline_df)` |

## Comparison Goal

Do cluster-derived features (hard assignment, distance, or soft posterior) add predictive value on Wine beyond the raw feature baseline? Which variant performs best?
