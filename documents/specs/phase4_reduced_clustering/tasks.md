# Phase 4 — Tasks

**Status:** [x] IMPLEMENTATION COMPLETE — run `make phase4` to produce artifacts

- [x] Implement `plot_phase4_heatmap(df, out_dir)` in `src/utils/plotting.py`
- [x] Implement `plot_phase4_comparison(df_reduced, df_raw, dataset, out_dir)` in `src/utils/plotting.py`
- [x] Write `scripts/run_phase_4_reduced_cluster.py` with logger wired
- [x] n_components locked in design.md (from Phase 3: Wine PCA=8 ICA=4 RP=8; Adult PCA=22 ICA=11 RP=22)
- [ ] Run all 12 combinations — produce `summary_table.csv`  ← needs `make phase4`
- [ ] Produce 3 figures in `artifacts/figures/phase4_clustering/`  ← needs `make phase4`
- [ ] Verify summary_table.csv has exactly 12 rows  ← assertion in script
- [ ] Draft narrative: which reduced spaces improve vs degrade clustering (for report)
