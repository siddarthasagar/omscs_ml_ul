---
inclusion: always
---

# Structure — CS7641 UL Spring 2026

## Repository Layout

```
omscs_ml_ul/
├── data/                              # Raw CSVs (git-tracked)
├── documents/
│   ├── steering/                      # Always-loaded project context
│   ├── specs/                         # Per-phase: requirements + design + tasks
│   ├── adr/                           # Frozen architecture decision records
│   ├── TODO.md                        # Cross-phase open issues
│   ├── faq.md                         # Methodological Q&A
│   └── canvas/                        # Assignment docs (read-only reference)
├── src/
│   ├── config.py                      # Centralized constants
│   ├── data/
│   │   ├── wine.py                    # load_wine(seed) → 6-tuple
│   │   └── adult.py                   # load_adult(seed) → 6-tuple
│   ├── unsupervised/
│   │   ├── clustering.py              # run_kmeans_sweep, run_gmm_sweep
│   │   └── reduction.py              # fit_pca, fit_ica, fit_rp, fit_tsne
│   ├── supervised/
│   │   ├── nn_baseline.py             # WineNN(input_dim)
│   │   └── training.py                # train_wine_nn(...)
│   └── utils/
│       ├── logger.py                  # configure_logger(run_id)
│       ├── metrics.py                 # macro_f1, silhouette, bic_aic
│       └── plotting.py                # all plot_* functions
├── scripts/
│   ├── run_phase_2_raw_cluster.py
│   ├── run_phase_3_raw_reduction.py
│   ├── run_phase_4_reduced_cluster.py
│   ├── run_phase_5_nn_reduced.py
│   ├── run_phase_6_nn_cluster_features.py
│   ├── run_phase_7_tsne.py
│   └── generate_report_tables.py
├── tests/
│   ├── test_data.py                   # Gate 1: shape + leakage
│   └── test_unsupervised.py           # Gate 2: label-free selection
└── artifacts/                         # git-ignored runtime outputs
    ├── logs/                          # phase{N}_<ts>.log per run
    ├── metrics/phase{N}_*/            # CSVs per phase
    ├── figures/phase{N}_*/            # PNGs per phase
    └── tables/                        # LaTeX/MD for report
```

## Module Contracts

### `src/config.py`
```python
SEED_EXPLORE: int = 42
SEEDS_REPORT: list[int] = list(range(42, 52))
WINE_N_FEATURES: int = 12
WINE_N_CLASSES: int = 8
NN_LR: float = 1e-3
NN_BETAS: tuple = (0.9, 0.999)
NN_WEIGHT_DECAY: float = 0.0
NN_TRAIN_BATCH_SIZE: int = 128
NN_VAL_BATCH_SIZE: int = 256
NN_MAX_EPOCHS: int = 20
NN_HIDDEN_DIM: int = 100
DATA_DIR: Path = Path("data")
ARTIFACTS_DIR: Path = Path("artifacts")
```

### `src/data/wine.py`
```python
def load_wine(seed: int = 42) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    Drops quality (leakage) and class (target). Keeps type (numeric 0/1).
    Asserts X_train.shape[1] == 12. Fits StandardScaler on X_train only."""
```

### `src/data/adult.py`
```python
def load_adult(seed: int = 42) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    ColumnTransformer (StandardScaler + OneHotEncoder) fit on X_train only."""
```

### `src/unsupervised/clustering.py`
```python
def run_kmeans_sweep(X, k_range, seed) -> DataFrame:
    """Cols: k, inertia, silhouette, calinski_harabasz, davies_bouldin"""

def run_gmm_sweep(X, n_range, seed, reg_covar=1e-3) -> DataFrame:
    """Cols: n_components, bic, aic, silhouette"""
```

### `src/unsupervised/reduction.py`
```python
def fit_pca(X_train, n_components=None) -> tuple[PCA, ndarray]
def fit_ica(X_train, n_components, seed) -> tuple[FastICA, ndarray]  # returns kurtosis array
def fit_rp(X_train, n_components, seed) -> tuple[SparseRandomProjection, ndarray]
def rp_reconstruction_error(rp, X) -> float
def fit_tsne(X, seed=42) -> ndarray  # 2D embedding, visualization only
```

### `src/supervised/nn_baseline.py`
```python
def WineNN(input_dim: int) -> nn.Module:
    """Linear(input_dim, 100) → ReLU → Linear(100, 8). No dropout."""
```

### `src/supervised/training.py`
```python
def train_wine_nn(X_train, y_train, X_val, y_val, config) -> tuple[nn.Module, dict]:
    """Returns (model, history). history keys: epoch, train_loss, val_loss, train_f1, val_f1."""
```

### `src/utils/logger.py`
```python
def configure_logger(run_id: str) -> logging.Logger:
    """Writes to artifacts/logs/{run_id}.log and stdout. Idempotent."""
```

### `src/utils/plotting.py`

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

## Naming Conventions

- Phase scripts: `run_phase_{N}_{slug}.py`
- Artifact dirs: `artifacts/{metrics,figures}/phase{N}_{slug}/`
- Log files: `artifacts/logs/phase{N}_{YYYYMMDDTHHMMSS}.log`
- Spec dirs: `documents/specs/phase{N}_{slug}/`
