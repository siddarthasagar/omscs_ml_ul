## Project
CS7641 Spring 2026 Unsupervised Learning. Python 3.13, uv, scikit-learn + PyTorch.
Datasets: data/adult.csv, data/wine.csv. No src/ or tests/ yet — planning only.

## Current Status
- Phase 0: [x] COMPLETE — all blockers resolved 2026-03-19
- Phase 1–7: [ ] unblocked, ready to implement
Update this block at end of every session.

## Phase 0 Audit Results (locked 2026-03-19)
- OL input_dim: **12** — 11 physicochemical features + `type` (numeric 0/1, StandardScaled, NOT dropped)
- OL architecture: `Linear(12, 100) → ReLU → Linear(100, 8)`
- OL optimizer: Adam, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0
- OL batch_size: 128 (train), 256 (val)
- OL max_epochs: 20
- OL early stopping: NOT used in baseline (regularization experiments used patience=[3,5,7])
- OL dropout: none
- OL split: 60/20/20, stratified, seed=42
- OL preprocessing: ColumnTransformer fit on X_train only; StandardScaler for all numeric cols (incl. type); no categorical OHE needed (type is already 0/1)
- Source: `/Users/siddarthasagarchinne/github/omscs_ml_ol/src/backbone.py`, `src/project_constants.py`, `src/optimizer_constants.py`

**Decision:** UL Wine uses 12 features (match OL exactly) so raw-feature UL run is numerically comparable to OL-reported Macro-F1.

## Build Commands
make dev                                       # venv + all deps
make test                                      # full pytest
make lint / make format                        # ruff
uv run pytest tests/test_data.py -v            # Gate 1 (after Phase 1)
uv run pytest tests/test_unsupervised.py -v    # Gate 2 (after Phase 2)

## Non-Negotiable Constraints
1. Wine: exactly 12 features. Drop only `quality` (leakage) and `class` (target). Keep `type` (numeric 0/1). Assert X_train.shape[1] == 12.
2. Preprocessing: fit StandardScaler/OneHotEncoder on X_train ONLY. Never on val or test.
3. Unsupervised selection: K and n_components chosen WITHOUT labels. Labels post-selection only.
4. NN follow-on (Steps 4+5): Wine ONLY. Adult excluded from supervised experiments.
5. Seeds: exploration = 42; report-grade = 42–51 inclusive.
6. t-SNE: visualization only. Never used as NN training input.
7. UL raw-feature Wine run (12 features) is numerically comparable to OL-reported Macro-F1. UL comparisons are both internal (raw vs reduced vs cluster-augmented) AND against OL baseline.

## Key File Locations
Full spec (scope + contracts):     documents/specs/ul_master_spec.md
Phase execution + module design:   documents/specs/ul_execution_guide.md
Algorithm guidance + hypotheses:   documents/specs/ul_research_notes.md
Assignment v2:                     documents/canvas/assignment/md/UL_Report_Spring_2026_v2.md
FAQ v2:                            documents/canvas/assignment/md/UL_Report_Spring_2026_FAQ_v2.md
Prior OL report:                   documents/REPORT_OL/OL_Report_schinne3.tex
Prior OL repo:                     /Users/siddarthasagarchinne/github/omscs_ml_ol

## Target Directory Layout (does not exist yet)
src/config.py                     SEED_EXPLORE, SEEDS_REPORT, WINE_N_FEATURES=12, paths
src/data/wine.py                  load_wine(seed) -> (X_train, X_val, X_test, y_train, y_val, y_test)
src/data/adult.py                 load_adult(seed) -> same signature
src/unsupervised/clustering.py    run_kmeans_sweep(), run_gmm_sweep()
src/unsupervised/reduction.py     fit_pca(), fit_ica(), fit_rp(), fit_tsne()
src/supervised/nn_baseline.py     WineNN(input_dim: int)
src/supervised/training.py        train_wine_nn(X, y, X_val, y_val, config) -> (model, history)
src/utils/metrics.py              macro_f1(), silhouette(), bic_aic()
src/utils/plotting.py             plot_elbow(), plot_tsne(), plot_learning_curves()
scripts/run_phase_N_*.py          one entrypoint per phase
tests/test_data.py                Gate 1: 12-feature assert, no-leakage checks
tests/test_unsupervised.py        Gate 2: label-free selection checks
artifacts/phase*/                 runtime outputs (git-ignored)
