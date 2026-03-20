# Phase 6 — Requirements

**Status:** [ ] NOT STARTED — blocked on Phases 2 and 5

## What This Phase Must Produce

1. WineNN trained on 3 cluster-augmented input variants (raw 12 features + cluster-derived features).
2. 10 seeds per variant → 30 training runs total.
3. Comparison table and boxplot vs Phase 5 raw baseline.

## Acceptance Criteria

- SHALL produce `comparison_table.csv` with cols: variant, input_dim, seed, val_macro_f1, test_macro_f1 (30 rows).
- SHALL produce `phase6_f1_boxplot.png` with Phase 5 raw baseline overlaid.
- SHALL use frozen K values from ADR-002 to derive cluster features.
- SHALL use appended features as primary approach (raw 12 + cluster features), not cluster-only.
- SHALL use identical training config as Phase 5 (same Gate 3 check applies).
- Wine ONLY.

## Feature Engineering Variants

| Variant | Features | input_dim |
|---------|----------|-----------|
| `kmeans_onehot` | raw 12 + one-hot KMeans assignments | 12 + K_wine = 14 |
| `kmeans_dist` | raw 12 + KMeans centroid distances | 12 + K_wine = 14 |
| `gmm_posterior` | raw 12 + GMM posterior probabilities | 12 + n_wine = 19 |

## Output Locations

| Artifact | Path |
|----------|------|
| CSVs | `artifacts/metrics/phase6_nn_cluster/` |
| Figures | `artifacts/figures/phase6_nn_cluster/` |
| Log | `artifacts/logs/phase6_<ts>.log` |

## Pre-requisites

- Phase 2 complete (frozen K values).
- Phase 5 complete (raw baseline established for comparison).

## Blocker Questions (resolve after Phase 5 results)

1. Did any DR variant in Phase 5 beat raw? If yes: should Phase 6 append cluster features to the best DR input or always to raw?
2. Assignment says "by themselves or by appending" — current plan: appended as primary. Confirm after seeing Phase 5 numbers.
