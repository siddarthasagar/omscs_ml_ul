# UL Execution Guide

## Section 1: Module Architecture

### Directory Tree (target state)

```text
omscs_ml_ul/
├── data/                              # Raw CSVs
├── documents/                         # Specs and reports
├── src/
│   ├── config.py                      # Centralized constants
│   ├── data/
│   │   ├── adult.py                   # Adult loader & preprocessor
│   │   └── wine.py                    # Wine loader & preprocessor (11-feature contract)
│   ├── unsupervised/
│   │   ├── clustering.py              # K-Means, GMM wrappers & metrics
│   │   └── reduction.py              # PCA, ICA, RP, t-SNE wrappers & diagnostics
│   ├── supervised/
│   │   ├── nn_baseline.py             # PyTorch model definition
│   │   └── training.py                # Training loop, early stopping, evaluation
│   └── utils/
│       ├── metrics.py                 # Silhouette, BIC/AIC, Macro-F1
│       ├── plotting.py                # Report-ready figures
│       └── logger.py                  # Standardized per-run logging
├── tests/
│   ├── test_data.py                   # Gate 1: leakage + shape checks
│   └── test_unsupervised.py           # Gate 2: label-free selection checks
├── scripts/
│   ├── run_phase_2_raw_cluster.py
│   ├── run_phase_3_raw_reduction.py
│   ├── run_phase_4_reduced_cluster.py
│   ├── run_phase_5_nn_reduced.py
│   ├── run_phase_6_nn_cluster_features.py
│   ├── run_phase_7_tsne.py
│   └── generate_report_tables.py      # Aggregates flat metrics into LaTeX/MD tables
├── artifacts/
│   ├── logs/                          # Execution logs ({phase_id}_{run_id}.log)
│   ├── metrics/                       # Flat CSV/JSON outputs per phase
│   ├── figures/                       # Plots for the report
│   └── tables/                        # Final LaTeX/MD tables ready for inclusion
├── Makefile
└── pyproject.toml
```

### Module Contracts

#### `src/config.py`
```python
SEED_EXPLORE: int = 42
SEEDS_REPORT: list[int] = list(range(42, 52))   # 42–51 inclusive
WINE_N_FEATURES: int = 11
DATA_DIR: Path = Path("data")
ARTIFACTS_DIR: Path = Path("artifacts")
```

#### `src/data/wine.py`
```python
def load_wine(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    Drops `type` and duplicate target column. Asserts X_train.shape[1] == 11.
    Fits StandardScaler on X_train only.
    """
```

#### `src/data/adult.py`
```python
def load_adult(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    ColumnTransformer (StandardScaler + OneHotEncoder) fit on X_train only.
    """
```

#### `src/unsupervised/clustering.py`
```python
def run_kmeans_sweep(X: np.ndarray, k_range: range, seed: int) -> pd.DataFrame:
    """Cols: k, silhouette, calinski_harabasz, davies_bouldin"""

def run_gmm_sweep(X: np.ndarray, n_range: range, seed: int) -> pd.DataFrame:
    """Cols: n_components, bic, aic"""
```

#### `src/unsupervised/reduction.py`
```python
def fit_pca(X_train: np.ndarray, n_components: int | None = None) -> tuple[PCA, np.ndarray]:
    """Returns (fitted_pca, X_transformed)"""

def fit_ica(X_train: np.ndarray, n_components: int, seed: int) -> tuple[FastICA, np.ndarray]:
    """Returns (fitted_ica, kurtosis_array)"""

def fit_rp(X_train: np.ndarray, n_components: int, seed: int) -> tuple[SparseRandomProjection, np.ndarray]:
    """Returns (fitted_rp, X_transformed)"""

def rp_reconstruction_error(rp: SparseRandomProjection, X: np.ndarray) -> float:
    """Reconstruction error via pseudo-inverse"""

def fit_tsne(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Returns 2D embedding array. Visualization only — never used as NN input."""
```

#### `src/supervised/nn_baseline.py`
```python
def WineNN(input_dim: int) -> nn.Module:
    """
    Architecture locked from Phase 0 audit (hidden dims, activation, dropout).
    Only input_dim varies across raw/PCA/ICA/RP variants.
    """
```

#### `src/supervised/training.py`
```python
def train_wine_nn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: dict
) -> tuple[nn.Module, dict]:
    """
    Optimizer, lr, batch_size, patience locked from Phase 0 audit.
    Returns (model, history) where history contains train/val loss and macro_f1 per epoch.
    """
```

#### `src/utils/metrics.py`
```python
def macro_f1(y_true, y_pred) -> float: ...
def silhouette(X, labels) -> float: ...
def bic_aic(gmm, X) -> tuple[float, float]: ...
```

#### `src/utils/plotting.py`
```python
def plot_elbow(metrics_df: pd.DataFrame, save_path: Path) -> None: ...
def plot_tsne(embedding: np.ndarray, labels: np.ndarray, save_path: Path) -> None: ...
def plot_learning_curves(history: dict, save_path: Path) -> None: ...
```

#### `src/utils/logger.py`
```python
def configure_logger(run_id: str) -> None:
    """Routes logs to artifacts/logs/{run_id}.log and console."""
```

#### `scripts/generate_report_tables.py`
```python
def main() -> None:
    """Reads artifacts/metrics/*.csv and generates LaTeX/MD fragments in artifacts/tables/."""
```

---

## Section 2: Phase-by-Phase Execution

---

### Phase 0: Baseline Audit

**Status:** [ ] not started

**Objective:** Resolve the OL Wine feature-count ambiguity before any code is written.

**No files to create.** All findings are recorded in CLAUDE.md Open Blockers.

**Concrete tasks:**

1. Open `/Users/siddarthasagarchinne/github/omscs_ml_ol`. Locate Wine preprocessing path. Check whether `type` is dropped. Count features passed to the model.
2. Locate saved model checkpoint. Inspect `model.state_dict()['fc1.weight'].shape[1]` — this is the authoritative input_dim.
3. Record `lr`, `batch_size`, `hidden_dims`, `patience` from the OL config. Write these into CLAUDE.md Open Blockers section.
4. If `input_dim == 12`: document decision to rebuild baseline under the 11-feature contract. Note this in CLAUDE.md constraint #7.

**Exit criteria:**
- CLAUDE.md Open Blockers updated with confirmed answers to all 3 items.
- Feature count confirmed. Baseline config locked.

---

### Phase 1: Data Loading

**Status:** [ ] blocked on Phase 0

**Objective:** Build one reliable preprocessing contract per dataset for all later experiments.

**Files to create:**
- `src/__init__.py`
- `src/config.py`
- `src/data/__init__.py`
- `src/data/wine.py`
- `src/data/adult.py`
- `tests/__init__.py`
- `tests/test_data.py`

**Concrete tasks:**

1. `src/config.py` — define all constants listed in Section 1.

2. `src/data/wine.py` — implement `load_wine(seed=42)`:
   - load `data/wine.csv`
   - drop `type` and duplicate target column
   - normalize target to the class used in prior work
   - split into train/val/test using `seed`
   - fit `StandardScaler` on `X_train` only
   - `assert X_train.shape[1] == WINE_N_FEATURES`
   - return `(X_train, X_val, X_test, y_train, y_val, y_test)`

3. `src/data/adult.py` — implement `load_adult(seed=42)`:
   - load `data/adult.csv`
   - separate `class` target
   - split into train/val/test using `seed`
   - fit `ColumnTransformer(StandardScaler + OneHotEncoder)` on `X_train` only
   - return `(X_train, X_val, X_test, y_train, y_val, y_test)`

4. `tests/test_data.py` — implement:
   - `test_wine_feature_count()` — asserts `X_train.shape[1] == 11`
   - `test_wine_no_leakage()` — fits scaler on X_train, transforms X_val, checks val stats differ from train
   - `test_adult_no_leakage()` — same pattern for adult ColumnTransformer
   - `test_split_sizes()` — asserts train/val/test fractions are in expected range

**Validation command:**
```
uv run pytest tests/test_data.py -v
```

**Exit criteria:** All 4 tests pass. Feature counts and label distributions stable.

---

### Phase 2: Step 1 Raw Clustering

**Status:** [ ] blocked on Phase 1

**Objective:** Characterize cluster structure in original feature spaces.

**Files to create:**
- `src/unsupervised/__init__.py`
- `src/unsupervised/clustering.py`
- `scripts/run_phase_2_raw_cluster.py`
- `tests/test_unsupervised.py`

**Concrete tasks:**

1. `src/unsupervised/clustering.py`:
   - `run_kmeans_sweep(X, k_range, seed)` — for each k: fit KMeans, compute silhouette, calinski_harabasz, davies_bouldin; return DataFrame with cols `[k, silhouette, calinski_harabasz, davies_bouldin]`
   - `run_gmm_sweep(X, n_range, seed)` — for each n: fit GaussianMixture, compute bic, aic; return DataFrame with cols `[n_components, bic, aic]`

2. `scripts/run_phase_2_raw_cluster.py`:
   - sweep `k_range=range(2, 21)` on `X_train` for both datasets using `SEED_EXPLORE`
   - save 4 CSVs to `artifacts/metrics/phase2_clustering/`:
     - `wine_kmeans.csv`, `wine_gmm.csv`, `adult_kmeans.csv`, `adult_gmm.csv`

3. `tests/test_unsupervised.py`:
   - `test_kmeans_label_free()` — calls `run_kmeans_sweep` and confirms it receives no label arg
   - `test_gmm_bic_aic_present()` — confirms output DataFrame has `bic` and `aic` columns

**Validation command:**
```
uv run pytest tests/test_unsupervised.py -v && uv run python scripts/run_phase_2_raw_cluster.py
```

**Exit criteria:** Tests pass. 4 CSV artifacts produced. One K and one n_components chosen per dataset (label-free selection from metrics).

---

### Phase 3: Step 2 Raw Dimensionality Reduction

**Status:** [ ] blocked on Phase 1

**Objective:** Characterize how each linear DR method changes data geometry.

**Files to create:**
- `src/unsupervised/reduction.py`
- `src/utils/__init__.py`
- `src/utils/metrics.py`
- `src/utils/plotting.py`
- `scripts/run_phase_3_raw_reduction.py`

**Concrete tasks:**

1. `src/unsupervised/reduction.py` — implement `fit_pca`, `fit_ica`, `fit_rp`, `rp_reconstruction_error` per Section 1 contracts.

2. `src/utils/metrics.py` — implement `macro_f1`, `silhouette`, `bic_aic`.

3. `src/utils/plotting.py` — implement `plot_elbow`, `plot_tsne`, `plot_learning_curves`.

4. `scripts/run_phase_3_raw_reduction.py`:
   - run PCA, ICA, RP on both datasets
   - sweep RP seeds `42–51` and record `rp_reconstruction_error` per seed
   - save diagnostic CSVs to `artifacts/metrics/phase3_reduction/`:
     - `wine_pca.csv` (explained variance per component)
     - `wine_ica.csv` (kurtosis per component)
     - `wine_rp_stability.csv` (reconstruction error per seed)
     - `adult_pca.csv`, `adult_ica.csv`, `adult_rp_stability.csv`

**Validation command:**
```
uv run python scripts/run_phase_3_raw_reduction.py
```

**Exit criteria:** 6 artifact files produced. One frozen n_components chosen per method per dataset (from diagnostic outputs — no labels used).

---

### Phase 4: Step 3 Clustering in Reduced Spaces

**Status:** [ ] blocked on Phases 2 and 3

**Objective:** Compare raw-space and reduced-space clustering quality.

**Files to create:**
- `scripts/run_phase_4_reduced_cluster.py`

**Concrete tasks:**

1. `scripts/run_phase_4_reduced_cluster.py`:
   - for each of 12 combinations (2 datasets × 3 DR methods × 2 clusterers): transform `X_train` using frozen DR from Phase 3, run clustering using frozen K/n_components from Phase 2, compute metrics
   - save `artifacts/metrics/phase4_reduced_clustering/summary_table.csv` with 12 rows and cols `[dataset, dr_method, clusterer, silhouette, calinski_harabasz, davies_bouldin_or_bic]`

**Validation command:**
```
uv run python scripts/run_phase_4_reduced_cluster.py
```

**Exit criteria:** `summary_table.csv` exists with exactly 12 rows. Narrative drafted: which reduced spaces improve vs. degrade clustering.

---

### Phase 5: Step 4 Wine NN on Reduced Inputs

**Status:** [ ] blocked on Phase 0 (baseline must be confirmed first)

**Objective:** Measure whether linear DR changes Wine NN behavior.

**Files to create:**
- `src/supervised/__init__.py`
- `src/supervised/nn_baseline.py`
- `src/supervised/training.py`
- `scripts/run_phase_5_nn_reduced.py`

**Concrete tasks:**

1. `src/supervised/nn_baseline.py` — implement `WineNN(input_dim: int)`:
   - architecture (hidden dims, activation, dropout) locked from Phase 0 audit
   - only `input_dim` varies between raw/PCA/ICA/RP variants

2. `src/supervised/training.py` — implement `train_wine_nn(X_train, y_train, X_val, y_val, config)`:
   - optimizer, lr, batch_size, patience all locked from Phase 0 audit
   - returns `(model, history)` where `history = {epoch: [train_loss, val_loss, train_f1, val_f1]}`

3. `scripts/run_phase_5_nn_reduced.py`:
   - 4 variants: raw (11 features), PCA-reduced, ICA-reduced, RP-reduced
   - for each variant × seeds 42–51: train, record val Macro-F1
   - save `artifacts/metrics/phase5_nn_reduced/comparison_table.csv` with cols `[variant, seed, val_macro_f1, test_macro_f1, epochs_to_converge]`

**Gate 3 check (manual):** Confirm all 4 variants use identical optimizer, lr, batch_size, patience. Only input layer size differs.

**Validation command:**
```
uv run python scripts/run_phase_5_nn_reduced.py
```

**Exit criteria:** `comparison_table.csv` exists. Gate 3 check passes. One defensible conclusion about whether DR helps or hurts the Wine NN.

---

### Phase 6: Step 5 Wine NN With Cluster-Derived Features

**Status:** [ ] blocked on Phases 2 and 5

**Objective:** Test whether cluster information adds predictive value on Wine.

**Files to create:**
- `scripts/run_phase_6_nn_cluster_features.py`

**Concrete tasks:**

1. `scripts/run_phase_6_nn_cluster_features.py`:
   - 3 augmented variants (appended to raw 11 Wine features):
     - `kmeans_onehot`: one-hot K-Means assignments
     - `kmeans_dist`: K-Means centroid distances
     - `gmm_posterior`: GMM posterior probabilities
   - for each variant × seeds 42–51: train `WineNN(input_dim=11+k)`, record val Macro-F1
   - save `artifacts/metrics/phase6_nn_cluster/comparison_table.csv`

**Validation command:**
```
uv run python scripts/run_phase_6_nn_cluster_features.py
```

**Exit criteria:** `comparison_table.csv` exists. One defensible conclusion about whether hard or soft cluster features add useful signal.

---

### Phase 7: t-SNE Extra Credit

**Status:** [ ] optional, blocked on Phase 1

**Objective:** Add a nonlinear visualization layer without contaminating the core methodology.

**Files to create:**
- `scripts/run_phase_7_tsne.py`

**Concrete tasks:**

1. Add `fit_tsne(X, seed=42)` to `src/unsupervised/reduction.py` — returns 2D embedding array.

2. `scripts/run_phase_7_tsne.py`:
   - run t-SNE on canonical `X_train` for both datasets
   - generate 2D scatter with label overlay → `artifacts/figures/phase7_tsne/wine_tsne_labels.png`, `adult_tsne_labels.png`
   - generate 2D scatter with cluster assignment overlay → `wine_tsne_clusters.png`, `adult_tsne_clusters.png`

**Validation command:**
```
uv run python scripts/run_phase_7_tsne.py
```

**Exit criteria:** 4 PNG files produced. t-SNE not used as NN input anywhere.

---

### Phase 8: Report Table Generation

**Status:** [ ] blocked on previous phases

**Objective:** Automate formatting of flat metric artifacts into report-ready LaTeX or Markdown tables.

**Files to create:**
- `scripts/generate_report_tables.py`

**Concrete tasks:**

1. `scripts/generate_report_tables.py`:
   - Load CSVs/JSONs from `artifacts/metrics/`
   - Pivot, group, and format DataFrames to align with the final `REPORT_UL` structure
   - Output structured `.md` and `.tex` snippets into `artifacts/tables/`
   - Ensure the tables are directly copy-pasteable (or auto-included via `\input{}`) into the final LaTeX document.

**Validation command:**
```
uv run python scripts/generate_report_tables.py
```

**Exit criteria:** Aggregated `.tex` or `.md` tables are present in `artifacts/tables/` for immediate use in the report.

---

## Section 3: Validation Gates

| Gate | Command | Passes when |
|------|---------|-------------|
| Gate 1 | `uv run pytest tests/test_data.py -v` | All 4 data tests green |
| Gate 2 | `uv run pytest tests/test_unsupervised.py -v` | Clustering tests green; label-free confirmed |
| Gate 3 | Manual review | All Phase 5/6 variants use identical optimizer config; only input_dim differs |
| Gate 4 | Manual count | ≥ 8 CSVs and ≥ 3 figures in `artifacts/` before writing report prose |
