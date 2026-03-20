# ADR-002 — Frozen K Values From Phase 2 Label-Free Selection

**Status:** Accepted — 2026-03-20 (Phase 2 complete)
**Do not edit. This decision is frozen.**

## Context

Phase 2 swept K-Means and GMM over k/n_components in range(2, 21) on `X_train` for both datasets using `SEED_EXPLORE=42`. Selection was performed without labels — only silhouette, Calinski-Harabasz, Davies-Bouldin, inertia (KMeans), and BIC (GMM) were used.

## Decision

| Dataset | Algorithm | Frozen K |
|---------|-----------|----------|
| Wine    | KMeans    | **2**    |
| Wine    | GMM       | **7**    |
| Adult   | KMeans    | **8**    |
| Adult   | GMM       | **7**    |

## Rationale

**Wine KMeans = 2:** All four metrics agree at k=2. Silhouette peaks (0.340), CH peaks (1497), DB troughs (1.335), inertia drops sharply from k=2→3 then flattens. Parsimony favors the lowest k where metrics converge.

**Wine GMM = 7:** BIC reaches a local minimum at n=7 (60841), with a slight uptick at n=8 (61446). AIC excluded as primary criterion — it monotonically decreases and favors more components by design.

**Adult KMeans = 8:** Best silhouette score (0.114) in the sweep. CH has a local peak at k=8 (2583). DB continues improving but no clear elbow — k=8 balances metric convergence with parsimony. Adult silhouette scores are uniformly low (0.09–0.11) due to curse of dimensionality in 104-feature OHE space.

**Adult GMM = 7:** BIC elbow around n=6-7 (-8,290,938 at n=7). Reasonable silhouette (0.058). Consistent with Wine GMM choice.

## Consequences

- These K values are used as-is in Phase 4 (clustering in reduced spaces).
- These K values are used to derive cluster features in Phase 6 (NN with cluster-derived features).
- Do not re-select K from Phase 2 metrics without explicit user instruction.
- Artifact files: `artifacts/metrics/phase2_clustering/{wine,adult}_{kmeans,gmm}.csv`
