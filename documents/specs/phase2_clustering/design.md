# Phase 2 — Design

**Status:** ✅ COMPLETE (2026-03-20)

## Algorithm Configuration

**KMeans:** sklearn defaults — `max_iter=300`, `n_init="auto"` (resolves to 10 re-inits with k-means++). C-optimized BLAS backend. No HPO beyond the k sweep — intentional per assignment methodology.

**GMM:** `max_iter=100`, `reg_covar=1e-3` (explicit regularization to prevent singular covariance in Adult's 104-feature OHE space). float64 cast for numerical stability.

## K Selection Logic (label-free)

- **KMeans:** prefer k where silhouette peaks, CH peaks, DB troughs, and inertia elbow bends. When metrics agree, take the lowest k (parsimony).
- **GMM:** prefer n where BIC reaches its minimum. AIC excluded as primary — it monotonically decreases by design and favors more components. Shown for reference only.

## Frozen Selections (see ADR-002)

| Dataset | KMeans K | GMM n |
|---------|----------|-------|
| Wine    | 2        | 7     |
| Adult   | 8        | 7     |

## Figures

| Figure | Content |
|--------|---------|
| `{dataset}_kmeans.png` | 2×2: Elbow (inertia), Silhouette, Calinski-Harabasz, Davies-Bouldin vs k |
| `{dataset}_gmm.png` | 1×2: BIC+AIC (same axes), Silhouette vs n_components |

## Key Observations from Results

**Wine KMeans:** All metrics agree at k=2. Clustering captures red/white wine type partition (dominant geometric split), not quality label structure. Expected — physicochemical features do not cleanly encode the 8-class quality label.

**Adult KMeans:** All silhouettes low (0.09–0.11) — curse of dimensionality from 104 OHE features flattens Euclidean distances. k=8 is the best available choice, not a strong signal. Motivates DR in Phase 3.

**Adult GMM (negative BIC):** BIC values are negative because Adult's log-likelihood is very large in absolute value after OHE expansion. More negative = better. Normal behavior for this data.
