# Phase I/O Design — Logging, Figures, Results

Defines the standard pattern for **every** phase script: what logging to emit,
what figures to generate, and what result files to produce.

---

## Standard Script Template

Every `scripts/run_phase_N_*.py` follows this skeleton:

```python
run_id = f"phaseN_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
log = configure_logger(run_id)   # → artifacts/logs/phaseN_<timestamp>.log
log.info("Phase N start — run_id=%s", run_id)

# ... work ...

log.info("Phase N complete.")
```

**Rules:**
- No bare `print()` calls. All output through `log.info()` / `log.warning()`.
- tqdm progress bars are exempt (they write to stderr, not stdout).
- `run_id` convention: `phase{N}_{timestamp}` (e.g. `phase3_20260320T143000`).
- Logger creates `artifacts/logs/` if missing. Scripts must not re-create it.

---

## Phase 2 — Raw Clustering ✅ DONE

**Script:** `scripts/run_phase_2_raw_cluster.py`

| Layer | What | Destination |
|-------|------|-------------|
| Log | phase start, dataset shapes, sweep start/end, artifact paths | `artifacts/logs/phase2_<ts>.log` |
| Results | `{wine,adult}_kmeans.csv` — cols: k, inertia, silhouette, CH, DB | `artifacts/metrics/phase2_clustering/` |
| Results | `{wine,adult}_gmm.csv` — cols: n_components, bic, aic, silhouette | `artifacts/metrics/phase2_clustering/` |
| Figures | `{wine,adult}_kmeans.png` — 2×2: Elbow, Silhouette, CH, DB vs k | `artifacts/figures/phase2_clustering/` |
| Figures | `{wine,adult}_gmm.png` — 1×2: BIC+AIC, Silhouette vs n_components | `artifacts/figures/phase2_clustering/` |

**Frozen K selections (label-free, not to be re-run):**

| Dataset | KMeans K | GMM n |
|---------|----------|-------|
| Wine    | 2        | 7     |
| Adult   | 8        | 7     |

---

## Phase 3 — Raw Dimensionality Reduction

**Script:** `scripts/run_phase_3_raw_reduction.py`

### Logging

```
Phase 3 start — run_id=phase3_<ts>
── WINE | shape=(3897, 12) ──
  PCA: fitting, n_components=12 (full), explained_variance_ratio logged per component
  ICA: fitting, n_components=<chosen>, kurtosis per component logged
  RP:  stability sweep seeds=42..51, reconstruction_error per seed logged
── ADULT | shape=(27132, 104) ──
  (same structure)
Phase 3 complete. Frozen n_components: wine_pca=<N>, wine_ica=<N>, wine_rp=<N>, ...
```

### Results

| File | Columns | Note |
|------|---------|------|
| `wine_pca.csv` | component, explained_variance, cumulative_variance | Full 12-component sweep |
| `wine_ica.csv` | component, kurtosis | Per-component kurtosis for component selection |
| `wine_rp_stability.csv` | seed, n_components, reconstruction_error | seed sweep 42–51 |
| `adult_pca.csv` | component, explained_variance, cumulative_variance | Full sweep |
| `adult_ica.csv` | component, kurtosis | |
| `adult_rp_stability.csv` | seed, n_components, reconstruction_error | |

→ Destination: `artifacts/metrics/phase3_reduction/`

### Figures

| File | Content | Function |
|------|---------|---------|
| `{dataset}_pca_variance.png` | Explained variance + cumulative curve vs component | `plot_pca_variance(df, dataset, out_dir)` |
| `{dataset}_ica_kurtosis.png` | Kurtosis bar chart vs component | `plot_ica_kurtosis(df, dataset, out_dir)` |
| `{dataset}_rp_stability.png` | Reconstruction error vs seed (box or line) | `plot_rp_stability(df, dataset, out_dir)` |

→ Destination: `artifacts/figures/phase3_reduction/`
→ Add all three functions to `src/utils/plotting.py`.

### Frozen n_components (to be decided after running)

| Dataset | PCA | ICA | RP |
|---------|-----|-----|----|
| Wine    | TBD | TBD | TBD |
| Adult   | TBD | TBD | TBD |

Selection rule: PCA = cumulative variance ≥ 90%; ICA = elbow in sorted kurtosis; RP = seed-median reconstruction error plateau. All label-free.

---

## Phase 4 — Clustering in Reduced Spaces

**Script:** `scripts/run_phase_4_reduced_cluster.py`

### Logging

```
Phase 4 start — run_id=phase4_<ts>
── WINE × PCA (n=<N>) ──
  KMeans k=2: silhouette=0.XX, CH=XXXX, DB=X.XX
  GMM    n=7: bic=XXXXX, silhouette=0.XX
── WINE × ICA (n=<N>) ──
  ...
(12 combinations total: 2 datasets × 3 DR methods × 2 clusterers)
Phase 4 complete. Summary → artifacts/metrics/phase4_reduced_clustering/summary_table.csv
```

### Results

| File | Columns | Rows |
|------|---------|------|
| `summary_table.csv` | dataset, dr_method, clusterer, silhouette, calinski_harabasz, davies_bouldin, bic | 12 |

→ Destination: `artifacts/metrics/phase4_reduced_clustering/`

### Figures

| File | Content | Function |
|------|---------|---------|
| `phase4_clustering_heatmap.png` | Heatmap of silhouette scores across 12 DR×clusterer combinations, per dataset | `plot_phase4_heatmap(df, out_dir)` |
| `{dataset}_phase4_bar.png` | Grouped bar: raw vs PCA vs ICA vs RP for each clusterer metric | `plot_phase4_comparison(df, dataset, out_dir)` |

→ Destination: `artifacts/figures/phase4_clustering/`

---

## Phase 5 — Wine NN on Reduced Inputs

**Script:** `scripts/run_phase_5_nn_reduced.py`

### Logging

```
Phase 5 start — run_id=phase5_<ts>
── Variant: raw (input_dim=12) ──
  seed=42: epoch 1/20 train_loss=X.XX val_loss=X.XX val_f1=0.XX
  ...
  seed=42: DONE val_macro_f1=0.XX test_macro_f1=0.XX
  ...
── Variant: pca (input_dim=<N>) ──
  ...
Phase 5 complete. comparison_table.csv written (40 rows: 4 variants × 10 seeds).
```

### Results

| File | Columns | Rows |
|------|---------|------|
| `comparison_table.csv` | variant, seed, val_macro_f1, test_macro_f1, epochs_to_converge | 40 |
| `{variant}_seed{seed}_history.csv` | epoch, train_loss, val_loss, train_f1, val_f1 | 20 × 40 = 800 |

→ Destination: `artifacts/metrics/phase5_nn_reduced/`

### Figures

| File | Content | Function |
|------|---------|---------|
| `phase5_f1_boxplot.png` | Boxplot of val_macro_f1 across 10 seeds, one box per variant | `plot_f1_comparison(df, out_dir)` |
| `{variant}_learning_curves.png` | Mean ± std train/val loss + val_f1 across seeds vs epoch | `plot_learning_curves(history_df, variant, out_dir)` |

→ Destination: `artifacts/figures/phase5_nn_reduced/`
→ `plot_learning_curves` goes into `src/utils/plotting.py`.

---

## Phase 6 — Wine NN With Cluster-Derived Features

**Script:** `scripts/run_phase_6_nn_cluster_features.py`

### Logging

```
Phase 6 start — run_id=phase6_<ts>
── Variant: kmeans_onehot (input_dim=12+2=14) ──
  seed=42: val_macro_f1=0.XX test_macro_f1=0.XX
  ...
── Variant: kmeans_dist (input_dim=12+2=14) ──
  ...
── Variant: gmm_posterior (input_dim=12+7=19) ──
  ...
Phase 6 complete. comparison_table.csv written (30 rows: 3 variants × 10 seeds).
```

### Results

| File | Columns | Rows |
|------|---------|------|
| `comparison_table.csv` | variant, input_dim, seed, val_macro_f1, test_macro_f1 | 30 |

→ Destination: `artifacts/metrics/phase6_nn_cluster/`

### Figures

| File | Content | Function |
|------|---------|---------|
| `phase6_f1_boxplot.png` | Boxplot per variant, with Phase 5 raw baseline overlaid for reference | `plot_f1_comparison(df, baseline_df, out_dir)` |

→ Destination: `artifacts/figures/phase6_nn_cluster/`

---

## Phase 7 — t-SNE (Extra Credit)

**Script:** `scripts/run_phase_7_tsne.py`

### Logging

```
Phase 7 start — run_id=phase7_<ts>
── WINE t-SNE (n=3897, perplexity=30) — fitting ...
  Done. Embedding shape=(3897, 2)
  Figure (labels)   → artifacts/figures/phase7_tsne/wine_tsne_labels.png
  Figure (clusters) → artifacts/figures/phase7_tsne/wine_tsne_clusters.png
── ADULT t-SNE ...
Phase 7 complete.
```

### Results

None — t-SNE is visualization only, no CSV output.

### Figures

| File | Content | Function |
|------|---------|---------|
| `{dataset}_tsne_labels.png` | 2D scatter, coloured by ground-truth label | `plot_tsne(embedding, labels, title, out_path)` |
| `{dataset}_tsne_clusters.png` | 2D scatter, coloured by frozen KMeans cluster assignment | `plot_tsne(embedding, cluster_labels, title, out_path)` |

→ Destination: `artifacts/figures/phase7_tsne/`
→ `plot_tsne` goes into `src/utils/plotting.py`.

**Constraint:** t-SNE result is never saved as a CSV or used as NN input.

---

## Summary: plotting.py functions to add

| Function | Phase | Status |
|----------|-------|--------|
| `plot_kmeans_sweep(df, dataset, out_dir)` | 2 | ✅ done |
| `plot_gmm_sweep(df, dataset, out_dir)` | 2 | ✅ done |
| `plot_pca_variance(df, dataset, out_dir)` | 3 | TODO |
| `plot_ica_kurtosis(df, dataset, out_dir)` | 3 | TODO |
| `plot_rp_stability(df, dataset, out_dir)` | 3 | TODO |
| `plot_phase4_heatmap(df, out_dir)` | 4 | TODO |
| `plot_phase4_comparison(df, dataset, out_dir)` | 4 | TODO |
| `plot_f1_comparison(df, out_dir, baseline_df=None)` | 5, 6 | TODO |
| `plot_learning_curves(history_df, variant, out_dir)` | 5 | TODO |
| `plot_tsne(embedding, labels, title, out_path)` | 7 | TODO |

---

## Summary: artifacts/logs/ entries per phase

| run_id pattern | Script |
|----------------|--------|
| `phase2_<ts>` | `run_phase_2_raw_cluster.py` |
| `phase3_<ts>` | `run_phase_3_raw_reduction.py` |
| `phase4_<ts>` | `run_phase_4_reduced_cluster.py` |
| `phase5_<ts>` | `run_phase_5_nn_reduced.py` |
| `phase6_<ts>` | `run_phase_6_nn_cluster_features.py` |
| `phase7_<ts>` | `run_phase_7_tsne.py` |
