## Project
CS7641 Spring 2026 Unsupervised Learning. Python 3.13, uv, scikit-learn + PyTorch.
Datasets: data/adult.csv, data/wine.csv. No src/ or tests/ yet — planning only.

## Current Status
- Phase 0: [ ] not started — MUST complete before any code is written
- Phase 1–7: [ ] blocked
Update this block at end of every session.

## Open Blockers (resolve before Phase 0 closes)
1. OL Wine feature count: audit /Users/siddarthasagarchinne/github/omscs_ml_ol — is input_dim 11 or 12?
2. OL optimizer config: lr, batch_size, hidden dims, early-stopping patience for Wine NN?
3. If OL used 12 features: document decision to rebuild baseline here before Phase 5.

## Build Commands
make dev                                       # venv + all deps
make test                                      # full pytest
make lint / make format                        # ruff
uv run pytest tests/test_data.py -v            # Gate 1 (after Phase 1)
uv run pytest tests/test_unsupervised.py -v    # Gate 2 (after Phase 2)

## Non-Negotiable Constraints
1. Wine: exactly 11 features. Drop `type` and duplicate target. Assert X_train.shape[1] == 11.
2. Preprocessing: fit StandardScaler/OneHotEncoder on X_train ONLY. Never on val or test.
3. Unsupervised selection: K and n_components chosen WITHOUT labels. Labels post-selection only.
4. NN follow-on (Steps 4+5): Wine ONLY. Adult excluded from supervised experiments.
5. Seeds: exploration = 42; report-grade = 42–51 inclusive.
6. t-SNE: visualization only. Never used as NN training input.
7. Do not compare UL NN to OL baseline until Phase 0 resolves feature count.

## Key File Locations
Full spec (scope + contracts):     documents/specs/ul_master_spec.md
Phase execution + module design:   documents/specs/ul_execution_guide.md
Algorithm guidance + hypotheses:   documents/specs/ul_research_notes.md
Assignment v2:                     documents/canvas/assignment/md/UL_Report_Spring_2026_v2.md
FAQ v2:                            documents/canvas/assignment/md/UL_Report_Spring_2026_FAQ_v2.md
Prior OL report:                   documents/REPORT_OL/OL_Report_schinne3.tex
Prior OL repo:                     /Users/siddarthasagarchinne/github/omscs_ml_ol

## Target Directory Layout (does not exist yet)
src/config.py                     SEED_EXPLORE, SEEDS_REPORT, WINE_N_FEATURES, paths
src/data/wine.py                  load_wine(seed) -> (X_train, X_val, X_test, y_train, y_val, y_test)
src/data/adult.py                 load_adult(seed) -> same signature
src/unsupervised/clustering.py    run_kmeans_sweep(), run_gmm_sweep()
src/unsupervised/reduction.py     fit_pca(), fit_ica(), fit_rp(), fit_tsne()
src/supervised/nn_baseline.py     WineNN(input_dim: int)
src/supervised/training.py        train_wine_nn(X, y, X_val, y_val, config) -> (model, history)
src/utils/metrics.py              macro_f1(), silhouette(), bic_aic()
src/utils/plotting.py             plot_elbow(), plot_tsne(), plot_learning_curves()
scripts/run_phase_N_*.py          one entrypoint per phase
tests/test_data.py                Gate 1: 11-feature assert, no-leakage checks
tests/test_unsupervised.py        Gate 2: label-free selection checks
artifacts/phase*/                 runtime outputs (git-ignored)
