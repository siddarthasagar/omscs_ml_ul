---
inclusion: always
---

# Tech — CS7641 UL Spring 2026

## Stack

- **Python:** 3.13
- **Package manager:** uv
- **ML:** scikit-learn (clustering, DR, preprocessing), PyTorch (NN)
- **Data:** pandas, numpy
- **Viz:** matplotlib
- **Testing:** pytest
- **Linting:** ruff

## Seeds

| Constant | Value | Use |
|----------|-------|-----|
| `SEED_EXPLORE` | 42 | All exploratory runs, phase sweeps |
| `SEEDS_REPORT` | 42–51 inclusive | Report-grade NN training (10 seeds) |

## Wine Feature Contract (locked — see ADR-001)

- Input dim: **12** (11 physicochemical + `type` numeric 0/1, StandardScaled)
- Drop: `quality` (leakage), `class` (target)
- Assert `X_train.shape[1] == 12` in loader

## Preprocessing Rules

1. Fit all transforms (StandardScaler, OneHotEncoder, LabelEncoder) on `X_train` only.
2. Apply fitted transforms to val and test — never re-fit.
3. Split before fitting. Split order: raw CSV → train/val/test → fit transforms on train.

## NN Config (locked from Phase 0 OL audit — never change)

| Param | Value |
|-------|-------|
| Architecture | `Linear(input_dim, 100) → ReLU → Linear(100, 8)` |
| Optimizer | Adam |
| lr | 1e-3 |
| betas | (0.9, 0.999) |
| weight_decay | 0.0 |
| train batch_size | 128 |
| val batch_size | 256 |
| max_epochs | 20 |
| early stopping | not used in baseline |
| dropout | none |
| split | 60/20/20, stratified, seed=42 |

Only `input_dim` changes across raw / PCA / ICA / RP variants.

## Frozen K Values (locked — see ADR-002)

| Dataset | KMeans K | GMM n |
|---------|----------|-------|
| Wine    | 2        | 7     |
| Adult   | 8        | 7     |

## Logging Standard (see ADR-003)

Every phase script calls `configure_logger(run_id)` from `src/utils/logger.py`.
`run_id` pattern: `phase{N}_{YYYYMMDDTHHMMSS}` (e.g. `phase3_20260320T143000`).
No bare `print()` in phase scripts. tqdm progress bars are exempt.

## Build Commands

```
make dev                          # venv + all deps
make test                         # full pytest
make lint / make format           # ruff
uv run pytest tests/test_data.py -v           # Gate 1
uv run pytest tests/test_unsupervised.py -v   # Gate 2
uv run python scripts/run_phase_N_*.py        # phase entrypoints
```
