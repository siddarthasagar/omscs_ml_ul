# ADR-001 — Wine Uses 12 Features (Matches OL input_dim)

**Status:** Accepted — 2026-03-19 (Phase 0 audit)
**Do not edit. This decision is frozen.**

## Context

The OL project (`omscs_ml_ol`) trained a Wine neural network with `input_dim=12`. The 12 features are 11 physicochemical measurements plus `type` (a binary 0/1 column indicating red vs white wine). The OL repo kept `type` as a numeric feature and StandardScaled it alongside the others. It dropped `quality` (leakage) and `class` (target).

An earlier draft of the UL project considered dropping `type` to use 11 features, matching a common SL-era preprocessing choice.

## Decision

UL Wine uses exactly **12 features**: 11 physicochemical + `type` (numeric 0/1, StandardScaled). Drop `quality` and `class` only.

## Rationale

- Matching OL's `input_dim=12` exactly makes the UL raw-feature NN run (Phase 5) numerically comparable to the OL-reported Macro-F1 baseline.
- Dropping `type` would introduce a deliberate preprocessing divergence that breaks the OL comparison — the most important external reference point for Steps 4 and 5.
- Phase 0 audit confirmed this by reading `omscs_ml_ol/src/backbone.py`, `src/project_constants.py`, and `src/optimizer_constants.py` directly.

## Consequences

- `load_wine()` must assert `X_train.shape[1] == 12`.
- Any UL/OL comparison in the report is valid only if this feature count is respected.
- If a future experiment intentionally drops `type`, it must be treated as a separate variant (not the baseline) and disclosed explicitly in the report.

## Source Files Audited

- `/Users/siddarthasagarchinne/github/omscs_ml_ol/src/backbone.py`
- `/Users/siddarthasagarchinne/github/omscs_ml_ol/src/project_constants.py`
- `/Users/siddarthasagarchinne/github/omscs_ml_ol/src/optimizer_constants.py`
