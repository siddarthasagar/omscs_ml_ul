# ADR-003 — Standard Logging and I/O Pattern for Phase Scripts

**Status:** Accepted — 2026-03-20
**Do not edit. This decision is frozen.**

## Context

Phase 2 initially used bare `print()` calls, producing no persistent log files. `artifacts/logs/` was empty. Debugging required re-running scripts.

## Decision

Every `scripts/run_phase_N_*.py` must:

1. Call `configure_logger(run_id)` from `src/utils/logger.py` as the first action in `main()`.
2. Use `log.info()` / `log.warning()` for all output. No bare `print()`.
3. Use `run_id = "phase{N}"` (e.g. `"phase2"`). One log file per phase, overwritten on each run.
4. Log: phase start, dataset shapes, experiment progress, all artifact paths on save.

tqdm progress bars are exempt (they write to stderr).

## Standard Output Destinations

| Layer | Pattern | Example |
|-------|---------|---------|
| Log | `artifacts/logs/phase{N}.log` | `phase2.log` (overwritten on rerun) |
| Metrics | `artifacts/metrics/phase{N}_{slug}/` | `phase2_clustering/wine_kmeans.csv` |
| Figures | `artifacts/figures/phase{N}_{slug}/` | `phase2_clustering/wine_kmeans.png` |
| Tables | `artifacts/tables/` | `clustering_summary.tex` |

## Consequences

- `src/utils/logger.py` is a required module for all phases.
- No new phase script may be implemented without wiring the logger first.
- Implemented and verified: Phase 2 script (`run_phase_2_raw_cluster.py`) — 2026-03-20.
