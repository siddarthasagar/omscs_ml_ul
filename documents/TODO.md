# TODO — CS7641 UL Spring 2026

Tracks cross-cutting issues, blocked items, and decisions that do not belong in phase scripts.
Update this file whenever an item is resolved or a new issue surfaces.

---

## Open — Action Required

- [ ] **T1 — Review NN training hyperparameters**
  Review `src/supervised/training.py` and confirm canonical training defaults (Adam lr, betas,
  weight_decay, train/val batch sizes, max epochs, hidden dim). If `training.py` is absent or
  incomplete, document canonical defaults from `src/config.py` and add a short entry to
  `documents/faq.md` describing these defaults for the report methods section.

- [ ] **T2 — Phase 4 heatmap LaTeX layout**
  Change Phase 4 heatmap layout used in the report to be wide-and-short (landscape-style)
  and arrange heatmap figures stacked vertically (one below the other). Update the LaTeX
  figure include or wrapper in the report template so the stacked, wide heatmaps render
  correctly in Overleaf.

---

## Resolved

| Date       | Item |
|------------|------|
| 2026-03-24 | R9 — CH/DB discussion in Step 3: added 2 sentences; noted Adult GMM under RP divergence |
| 2026-03-24 | R8 — RP dimension design choice: noted RP dim = PCA dim is deliberate for fair comparison |
| 2026-03-24 | R7 — ICA median-kurtosis threshold justification: connected to Hyvärinen & Oja criterion |
| 2026-03-24 | R6 — Algorithm improvement suggestions added to Conclusion |
| 2026-03-24 | R5 — Same/different clusters after DR: cluster_stability.csv + Section VI paragraph |
| 2026-03-24 | R4 — GMM BIC sweep figures: expanded Fig. 1 to 2×4 panel |
| 2026-03-24 | R3 — NN training time per variant: timing_results.csv + Speed paragraph in Section VII |
| 2026-03-24 | R2 — Cluster-label alignment (ARI/purity): ari_results.csv + Section IV paragraph |
| 2026-03-24 | R1 — PCA/ICA component loadings figure: 3 figures + Section V interpretive prose |
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
