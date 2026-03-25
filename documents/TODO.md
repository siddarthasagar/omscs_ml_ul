# TODO — CS7641 UL Spring 2026

Tracks cross-cutting issues, blocked items, and decisions that do not belong in phase scripts.
Update this file whenever an item is resolved or a new issue surfaces.

---

## Report Revision — Open Items

Items identified by post-submission assessment (`assement.txt`), ordered by priority.

### 🔴 High Priority (likely point loss)

- [x] **R1 — PCA/ICA component loadings figure** ✓ 2026-03-24
  scripts/run_analysis_loadings.py → 3 figures (wine_pca_loadings, wine_ica_loadings, adult_pca_loadings).
  Key finding: Wine PC1 dominated by `type` (red/white) — explains ARI=0.998 KMeans stability.
  Adult PC1–2 load on education-num/age/relationship — confirms OHE redundancy compression.
  Added 2 figures + interpretive prose to Section V.

- [x] **R2 — Cluster-label alignment (ARI/purity)** ✓ 2026-03-24
  scripts/run_analysis_ari.py → artifacts/metrics/analysis/ari_results.csv
  Key results: Wine KMeans K=2 ARI=0.981 vs type, 0.182 vs quality; Adult KMeans K=8 ARI=0.059.
  Added paragraph with all ARI values to Section IV prose.

- [x] **R3 — NN training time per variant** ✓ 2026-03-24
  scripts/run_analysis_timing.py → artifacts/metrics/analysis/timing_results.csv
  Key result: no meaningful speed difference on Wine (~0.27s all variants); explained by
  small dataset size where matrix multiply cost is negligible. Added Speed paragraph to Section VII.

### 🟡 Medium Priority (partial credit risk)

- [x] **R4 — GMM BIC sweep figures** ✓ 2026-03-24
  Expanded Fig. 1 to 2×4 panel (KMeans + GMM for Wine and Adult). Caption updated.

- [x] **R5 — Same/different clusters after DR** ✓ 2026-03-24
  scripts/run_analysis_cluster_stability.py → artifacts/metrics/analysis/cluster_stability.csv
  Key results: Wine KMeans+PCA ARI=0.998 (same), Wine KMeans+ICA ARI=-0.010 (completely different),
  Adult KMeans+PCA ARI=0.804 (mostly same), all GMM 0.24–0.41 (partially different).
  Added paragraph to Section VI explaining membership changes and their causes.

- [x] **R6 — Algorithm improvement suggestions** ✓ 2026-03-24
  Added paragraph to Conclusion: cosine distance for Adult KMeans, reduce Wine GMM K toward 2–3.

- [ ] **T1 — Review NN training hyperparameters (NEW)**
  Review `src/supervised/training.py` and confirm canonical training defaults (Adam lr, betas,
  weight_decay, train/val batch sizes, max epochs, hidden dim). If `training.py` is absent or
  incomplete, document canonical defaults from `src/config.py` and add a short entry to
  `documents/faq.md` describing these defaults for the report methods section.

- [ ] **T2 — Phase 4 heatmap LaTeX layout (NEW)**
  Change Phase 4 heatmap layout used in the report to be wide-and-short (landscape-style)
  and arrange heatmap figures stacked vertically (one below the other). Update the LaTeX
  figure include or wrapper in the report template so the stacked, wide heatmaps render
  correctly in Overleaf.

### 🟢 Minor (polish)

- [x] **R7 — ICA median-kurtosis threshold justification** ✓ 2026-03-24
  Added sentence connecting above-median rule to Hyvärinen & Oja non-Gaussianity criterion.

- [x] **R8 — RP dimension design choice statement** ✓ 2026-03-24
  Added sentence noting RP dim = PCA dim is deliberate to hold d constant for fair comparison.

- [x] **R9 — CH/DB discussion in Step 3** ✓ 2026-03-24
  Added 2 sentences: CH/DB consistent with silhouette; noted Adult GMM under RP divergence.

---

## Resolved

| Date       | Item |
|------------|------|
| 2026-03-24 | Phase 8 complete — 5 LaTeX table bodies in artifacts/tables/ |
| 2026-03-24 | UL_Report_schinne3.tex drafted — 5 pages, all 5 steps covered, compiles clean |
| 2026-03-24 | copy_report_to_overleaf.sh created and validated |
| 2026-03-24 | copy_to_submission.sh and copy_report_assets.sh relative-path bugs fixed |
| 2026-03-21 | Phase 7 complete — 4 t-SNE PNGs (labels + clusters for Wine and Adult) |
| 2026-03-21 | Phase 6 complete — 30 runs (3 variants × 10 seeds), 1 figure, baseline overlaid |
| 2026-03-21 | Phase 5 complete — 40 runs (4 variants × 10 seeds), 2 figures, Gate 3 passed |
| 2026-03-21 | Phase 4 complete — 12 combinations run, summary_table.csv + 3 PNGs produced |
| 2026-03-21 | Phase 3 complete — 6 CSVs + 6 PNGs, n_components frozen |
| 2026-03-21 | Specs reorganized — Kiro-style steering/specs/adr structure, old monolithics archived |
| 2026-03-20 | Phase 2 K values frozen — documented in ADR-002 |
| 2026-03-20 | FAQ process established — documents/faq.md created |
| 2026-03-20 | Implemented src/utils/logger.py; wired into Phase 2 script |
| 2026-03-20 | Added inertia to run_kmeans_sweep output |
| 2026-03-20 | Created src/utils/plotting.py with plot_kmeans_sweep and plot_gmm_sweep |
| 2026-03-20 | Phase 2 script generates 4 PNGs in artifacts/figures/phase2_clustering/ |
| 2026-03-19 | Phase 0 audit complete — all feature/architecture constants locked |
