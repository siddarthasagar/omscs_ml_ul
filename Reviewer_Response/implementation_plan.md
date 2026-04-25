# Reviewer Response — Implementation Plan (v3)

Date: 2026-04-26

## Overview

Three graded gaps to fix (16 points lost, max 8 recoverable):

| Gap | Points Lost | Fix |
|---|---|---|
| A3. Figure legibility | 2 | rcParams report-wide + re-run Phase 2 and 3 |
| E. Step 3 missing K re-selection + visualization | 8 | New sweep logic + `figure*` sweep figures + Phase 8 macros |
| G. Step 5 missing wall-clock analysis | 6 | Timing instrumentation + corrected prose |

---

## Change 1 — Phase 4: K Re-Selection in Reduced Spaces

### Root cause

`run_phase_4_reduced_cluster.py` reused raw-space frozen K values in all reduced spaces.
The requirements spec explicitly said "SHALL use frozen K values from ADR-002 — no re-selection."
That spec was wrong relative to the assignment intent. **Update the spec first.**

`documents/specs/phase4_reduced_clustering/requirements.md` line 15 — change to:
> SHALL re-select K in each reduced space using the same label-free criteria as Phase 2:
> KMeans by joint silhouette/CH/DB; GMM by BIC minimum.

### K selection rule — must match Step 1 methodology exactly

The report already states at UL_Report_schinne3.tex:148 that Step 1 selected K by:
> "joint behavior of silhouette score, Calinski-Harabasz (CH), and Davies-Bouldin (DB)"

and GMM by "minimizing BIC". The reduced-space sweep must use the **same rules**:
- **KMeans**: pick K with highest silhouette; ties broken by highest CH then lowest DB
- **GMM**: pick n_components with lowest BIC

### Changes to `scripts/run_phase_4_reduced_cluster.py`

**Add sweep function:**
```python
def sweep_reduced_space(
    X_r: np.ndarray, k_range: range, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    km_rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(X_r)
        labels = km.labels_
        km_rows.append({
            "k": k,
            "inertia":           km.inertia_,
            "silhouette":        silhouette_score(X_r, labels),
            "calinski_harabasz": calinski_harabasz_score(X_r, labels),
            "davies_bouldin":    davies_bouldin_score(X_r, labels),
        })
    km_df = pd.DataFrame(km_rows)

    gmm_rows = []
    for n in k_range:
        gmm = GaussianMixture(
            n_components=n, random_state=seed, reg_covar=1e-3
        ).fit(X_r.astype(np.float64))
        labels = gmm.predict(X_r.astype(np.float64))
        gmm_rows.append({
            "n_components": n,
            "bic":          gmm.bic(X_r.astype(np.float64)),
            "aic":          gmm.aic(X_r.astype(np.float64)),
            "silhouette":   silhouette_score(X_r, labels),
        })
    gmm_df = pd.DataFrame(gmm_rows)
    return km_df, gmm_df
```

**Add K selection function (joint criteria):**
```python
def select_k_reduced(km_df: pd.DataFrame, gmm_df: pd.DataFrame) -> tuple[int, int]:
    best_sil = km_df["silhouette"].max()
    candidates = km_df[km_df["silhouette"] >= best_sil - 1e-6]
    if len(candidates) > 1:
        candidates = candidates.sort_values(
            ["calinski_harabasz", "davies_bouldin"],
            ascending=[False, True],
        )
    km_k = int(candidates.iloc[0]["k"])
    gmm_k = int(gmm_df.loc[gmm_df["bic"].idxmin(), "n_components"])
    return km_k, gmm_k
```

**Update `run_dataset()`**: call sweep + select_k before clustering; use returned K values
instead of `f["kmeans_k"]` / `f["gmm_n"]`. Save sweep CSVs:
```
artifacts/metrics/phase4_clustering/{dataset}_{dr}_kmeans_sweep.csv
artifacts/metrics/phase4_clustering/{dataset}_{dr}_gmm_sweep.csv
```
(12 CSVs total — 6 DR spaces × 2 clusterers)

**Update `phase4.json`** — add `reduced_k` block:
```json
{
  "reduced_k": {
    "wine_pca":  {"kmeans": 2, "gmm": 5},
    "wine_ica":  {"kmeans": 2, "gmm": 4},
    "wine_rp":   {"kmeans": 2, "gmm": 5},
    "adult_pca": {"kmeans": 6, "gmm": 7},
    "adult_ica": {"kmeans": 5, "gmm": 6},
    "adult_rp":  {"kmeans": 7, "gmm": 7}
  },
  "silhouette":         { ... },
  "ari_raw_vs_reduced": { ... }
}
```
(Actual values filled in after `make phase4`.)

### New Step 3 sweep figures

**Figure layout (critical — must avoid legibility problem):**

Two figures, one per dataset. Each figure is **`figure*` spanning both columns at `\textwidth`**.
Layout: **3 rows × 2 panels** = 6 panels per figure:
- Row 1: PCA reduced space — left: KMeans silhouette sweep; right: GMM BIC+AIC sweep
- Row 2: ICA reduced space — same
- Row 3: RP reduced space — same

This is 6 panels at `\textwidth` = approximately 85mm per panel. Readable without zoom.
Each panel title states the selected K, e.g. "Wine PCA — KMeans (selected K=2, silhouette max)".

**Do NOT use 3×3 panels.** AIC is shown in the GMM right panel alongside BIC on the same
axes (identical to Phase 2's `plot_gmm_sweep` which already overlays BIC+AIC). KMeans CH
and DB do not need their own panel — they are the tiebreaker criteria, not the primary
selection metric. The caption explains the full selection rule.

Add `plot_phase4_reduced_sweeps()` to `src/utils/plotting.py`:
```python
def plot_phase4_reduced_sweeps(
    sweep_data: dict,   # {"pca": (km_df, gmm_df), "ica": ..., "rp": ...}
    dataset_name: str,
    out_dir: Path,
) -> Path:
    """
    3×2 figure: one row per DR method, left=KMeans silhouette, right=GMM BIC+AIC.
    Selected K is marked with a vertical dashed line on each panel.
    figsize=(14, 10), dpi=150.
    """
    ...
    out_path = out_dir / f"{dataset_name}_phase4_reduced_sweeps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    ...
```

**No new "combined" composition plots needed for Steps 1 or 5.** The LaTeX subfigure
markup for Fig. 1 (clustering sweeps) and the F1 boxplot figure is kept as-is.
The rcParams font-size increase + Phase 2/3/4/6 re-runs are sufficient for those figures.
Do not add new composition functions — that would add scope with no clear benefit.

### Update Phase 8 (`scripts/run_phase_8_report_tables.py`)

The Phase 4 silhouette table reads from `phase4.json["silhouette"]` — values update
automatically after re-run, no structural change needed.

**Add 12 new macros** in `emit_report_numbers()`:
```python
meta4 = _load_metadata(4)
rk = meta4["reduced_k"]
for space, vals in rk.items():
    ds, dr = space.rsplit("_", 1)
    prefix = ds.title() + dr.title()          # e.g. "WinePca"
    macros[f"{prefix}ReducedKmK"]  = str(vals["kmeans"])
    macros[f"{prefix}ReducedGmmN"] = str(vals["gmm"])
```
Macro names: `\WinePcaReducedKmK`, `\WinePcaReducedGmmN`, `\WineIcaReducedKmK`, ...
(12 macros total). These allow Step 3 prose to reference reduced-space K values without
hardcoding.

---

## Change 2 — Phase 6: Wall Clock Instrumentation

### Changes to `scripts/run_phase_6_nn_cluster_features.py`

```python
import time

# inside the per-seed loop:
t0 = time.perf_counter()
hist = train_wine_nn(Xtr, y_train, Xv, y_val, seed=seed)
train_time_s = time.perf_counter() - t0

comparison_rows.append({
    "variant":      variant,
    "seed":         seed,
    "input_dim":    Xtr.shape[1],
    "val_f1_final": final_f1,
    "val_f1_best":  best_f1,
    "train_time_s": round(train_time_s, 3),
})
```

Update `phase6.json` to include `mean_timing_s`:
```json
{
  "mean_f1":       { "kmeans_onehot": 0.xxx, ... },
  "mean_timing_s": { "kmeans_onehot": x.x,  ... },
  "input_dim":     { "kmeans_onehot": 14,   ... }
}
```

### Update Phase 8

Add `Time (s)` column to `emit_phase6_table()` and add macros:
`\KMeansOnehotTimingS`, `\KMeansDistTimingS`, `\GmmPostTimingS`, plus
`\RawBaselineTimingS` (loaded from phase5.json for comparison).

### Step 5 timing prose (correct framing)

Updates per epoch are fixed: ~30 batches/epoch × 20 epochs regardless of input width.
Do NOT claim "fewer gradient updates." Correct paragraph:

> **Speed.** Because cluster features are *appended* to the raw 12-d input, input
> dimension grows (\KMeansOnehotDim-d for KMeans one-hot; \GmmPostDim-d for GMM
> posterior), so no reduction in first-layer computation is expected.
> Measured wall-clock times confirm this: mean training time is
> \KMeansOnehotTimingS\,s (KMeans one-hot), \KMeansDistTimingS\,s (KMeans dist),
> and \GmmPostTimingS\,s (GMM posterior), compared to \RawBaselineTimingS\,s for
> the raw baseline --- differences are within measurement noise and reflect the
> marginally wider input matrix.  The performance gain from cluster augmentation
> therefore comes entirely from richer feature representation, not from reduced
> computational cost.

---

## Change 3 — Figure Legibility

### Root cause

The reviewer's A3 deduction was report-wide. Fixes apply in two tiers.

### Tier 1 — global rcParams (zero re-run cost, affects all regenerated figures)

Add at the top of `src/utils/plotting.py`:
```python
import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
```

### Tier 2 — re-run Phase 2 and Phase 3 to regenerate their figures with new fonts

Phase 2 (K-means/GMM sweep plots, Step 1 figures) and Phase 3 (PCA/ICA/RP plots,
Step 2 figures) are fast and risk-free to re-run. Adding them to the execution plan
ensures the largest set of report figures gets the font update.

**Phase 5 (NN training, 40 runs):** skip re-run — already full credit, slow, and the
F1 boxplot figure already has reasonable text sizes. Not worth the time risk given
today's deadline. Note this in the response document.

**Phase 7 (t-SNE):** skip re-run — already full credit (extra credit), and t-SNE
scatter point sizes drive legibility more than font size. Not worth re-running.

### What stays as-is in LaTeX layout

The existing `0.48\linewidth` subfigure markup for Figs. 1–6 and the t-SNE figure
is kept. The rcParams update improves font sizes within those panels without requiring
any LaTeX edits to already-full-credit sections. The new Step 3 sweep figures are
`figure*` at `\textwidth` and do not need subfigure markup.

---

## Execution Order

```
# A. Spec update (no code)
# documents/specs/phase4_reduced_clustering/requirements.md — update line 15

# B. plotting.py
# 1. Add global rcParams block at top
# 2. Add plot_phase4_reduced_sweeps() — 3×2 panels, figure* layout

# C. Phase scripts
# 3. Edit run_phase_4_reduced_cluster.py (sweep + joint K selection + sweep CSVs +
#    call plot_phase4_reduced_sweeps + reduced_k in phase4.json)
# 4. Edit run_phase_6_nn_cluster_features.py (add train_time_s)

# D. Phase 8
# 5. Edit run_phase_8_report_tables.py (12 reduced_k macros + phase6 timing column)

# E. Re-runs (in order)
make phase2          # ~2 min — regenerates K-sweep figures with new fonts
make phase3          # ~2 min — regenerates DR figures with new fonts
make phase4          # ~5 min — new K-selection logic + sweep figures
make phase6          # ~10 min — timing instrumentation
make phase8          # ~1 min — all table bodies + macros

# F. Report edits
# 6. UL_Report_schinne3.tex:
#    - Add two figure* sweep figures to Step 3 section
#    - Update Step 3 prose: opening paragraph states K is re-selected per reduced
#      space, references sweep figures, cites new K values via macros
#    - Add timing column to Phase 6 table (table column, not caption)
#    - Add corrected timing paragraph to Step 5
# 7. pdflatex UL_Report_schinne3.tex (twice for cross-refs)
#    Inspect at 100% zoom — every axis label and legend must be readable

# G. Reviewer response
# 8. Write Reviewer_Response/UL_Report_Reviewer_Response_schinne3.tex
```

---

## Files Modified

| File | Change |
|---|---|
| `documents/specs/phase4_reduced_clustering/requirements.md` | Update K selection rule |
| `src/utils/plotting.py` | Global rcParams; add `plot_phase4_reduced_sweeps()` |
| `scripts/run_phase_4_reduced_cluster.py` | Sweep + joint K selection; sweep CSVs; `reduced_k` in phase4.json |
| `scripts/run_phase_6_nn_cluster_features.py` | Add `train_time_s` |
| `scripts/run_phase_8_report_tables.py` | 12 reduced_k macros; phase6 timing column + macros |
| `REPORT_UL/UL_Report_schinne3.tex` | New `figure*` sweep figs in Step 3; K re-selection prose; Step 5 timing |
| `Reviewer_Response/UL_Report_Reviewer_Response_schinne3.tex` | New: 2-page response |

## Files NOT Modified

- `scripts/run_phase_2_raw_cluster.py` — re-run but no code change
- `scripts/run_phase_3_raw_reduction.py` — re-run but no code change
- `scripts/run_phase_5_nn_reduced.py` — not re-run, full credit
- `scripts/run_phase_7_tsne.py` — not re-run, full credit
- ADR-002 — records raw-space frozen K; still correct for Steps 1 and 5

---

## Reviewer Response Document Structure (2-page max)

```
§1  A3 Legibility
    - Global font-size rcParams applied; Phases 2, 3, 4, 6 figures regenerated
    - New Step 3 sweep figures use figure* full-textwidth layout
    - Phases 5 and 7 not regenerated (full credit; noted explicitly)

§2  E — Step 3 K Re-Selection
    - Spec corrected: K is now re-selected per reduced space
    - Same joint silhouette/CH/DB (KMeans) and BIC (GMM) criteria as Step 1
    - Two new figure* figures added (Wine and Adult reduced-space sweeps, 3×2 panels)
    - Reduced-space K values vs raw-space K: [table after re-run]
    - Prose updated: where K differs from raw-space, explains why

§3  G — Step 5 Wall Clock
    - Added train_time_s; re-ran 30 seeds
    - Timing numbers: [from re-run]
    - No speedup expected (features appended); differences are noise-level
    - Performance gain is representational, not computational
```
