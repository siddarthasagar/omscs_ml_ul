# Project FAQ — CS7641 UL Spring 2026

Captures explanations for recurring methodological questions raised during implementation.
**Rule:** Any time a "why does X work this way?" question is asked and answered, it goes here.

---

## Phase 2 — Raw Clustering

### Q: The phase 2 script finished very quickly. Is the speed correct?

**A: Yes — expected and correct.**

Wine is 3,897 rows × 12 features (after 60/20/20 split). Adult is 27,132 rows × 104 OHE features.
The sweep runs 38 total fits: 19 k values × 2 datasets × 2 algorithms (KMeans + GMM).
Both use sklearn's C-optimized implementations (KMeans uses BLAS, GMM uses scipy linalg).
Wine finishes in ~6 seconds; Adult in ~2.7 minutes — dominated by Adult GMM on 104 features.

---

### Q: Does KMeans use the right number of iterations? Is there any hyperparameter optimization?

**A: Yes, defaults are correct. No HPO — intentional.**

- **KMeans:** `max_iter=300` (sklearn default), `n_init="auto"` (resolves to 10 random re-inits using k-means++ initialization). Each restart selects the best result by inertia. This is more than sufficient for datasets of this size.
- **GMM:** `max_iter=100` (sklearn default), `reg_covar=1e-3` (added explicitly to prevent singular covariance matrices in high-dimensional Adult OHE space).
- **No HPO beyond the k/n_components sweep.** The assignment requires selecting K *without* labels — pure sweep over metrics (silhouette, BIC, AIC, CH, DB) is the correct methodology. Grid-searching max_iter or other internal params is out of scope and would not change the label-free selection.

---

### Q: How do you select K from the Phase 2 sweep? What are the actual chosen values?

**A: Label-free selection using metric convergence. Chosen values below.**

Selection logic (no labels used at any point):
- **KMeans:** prefer k where silhouette peaks, CH peaks, DB troughs, and the inertia elbow bends. When metrics agree, take the lowest k (parsimony).
- **GMM:** prefer n where BIC reaches its minimum (or elbow before overfitting). AIC is excluded as a primary criterion because it monotonically decreases — it favors more components by design and is only shown for reference.

| Dataset | Algorithm | Chosen K | Primary signal |
|---------|-----------|----------|----------------|
| Wine    | KMeans    | **2**    | All four metrics agree: silhouette=0.340 (peak), CH=1497 (peak), DB=1.335 (trough), inertia drops sharply 2→3 then flattens |
| Wine    | GMM       | **7**    | BIC local minimum at n=7 (60841); silhouette still positive (0.040) |
| Adult   | KMeans    | **8**    | Best silhouette (0.114), CH local peak (2583), DB improving; elbow is gradual due to high dimensionality |
| Adult   | GMM       | **7**    | BIC elbow around n=6-7 (-8290938); silhouette reasonable (0.058) |

These values are **frozen** for use in Phase 4 (clustering in reduced spaces) and Phase 6 (cluster-derived NN features).

---

### Q: Why does Wine KMeans converge to K=2 when there are 8 wine quality classes?

**A: Clustering captures structure in feature space, not label space.**

Wine has 8 quality classes (3–9), but the physicochemical features (alcohol, acidity, sulfates, etc.) cluster naturally around wine type (`type` = red/white, encoded 0/1). Red and white wines have structurally different profiles, so K=2 reflects the dominant geometric partition. Quality labels are a finer-grained subjective rating that is not fully recoverable from raw features — this mismatch is expected and is part of what Phase 2 is meant to reveal. We use labels only post-hoc (Phase 4 analysis, Phase 5 NN training).

---

### Q: Why does Adult have low silhouette scores (0.09–0.11) across all k values?

**A: High-dimensional OHE spaces are notoriously hard to cluster with Euclidean metrics.**

Adult after one-hot encoding has 104 binary/continuous features. Euclidean distance in high-dimensional binary spaces suffers from the curse of dimensionality — distances between points become nearly uniform, which flattens the silhouette score. This is expected and is exactly why Phase 3 (dimensionality reduction) exists: PCA/ICA/RP compress Adult to a lower-dimensional space before re-clustering in Phase 4.

---
