# TODO — CS7641 UL Spring 2026

Tracks cross-cutting issues, blocked items, and decisions that do not belong in phase scripts.
Update this file whenever an item is resolved or a new issue surfaces.

---

## Open Items

### [BUG] Logging is not wired — artifacts/logs/ is empty

**Priority:** Medium — does not block phase correctness, but makes debugging hard.

**Problem:** `src/utils/logger.py` is specified in the execution guide (Section 1) and `artifacts/logs/` is defined as the output directory, but:
- `src/utils/logger.py` does not exist yet.
- No script calls a logger. All output goes to stdout/tqdm only.
- `artifacts/logs/` directory does not exist (not created by any script).

**What to build:**
- `src/utils/logger.py` — implement `configure_logger(run_id: str) -> logging.Logger`:
  - Creates `artifacts/logs/{run_id}.log` (append mode).
  - Logs to both file and console (INFO level).
  - `run_id` convention: `phase{N}_{dataset}_{timestamp}` (e.g., `phase2_wine_20260320T143000`).
- Each `scripts/run_phase_*.py` calls `configure_logger(run_id)` at the top of `main()` and replaces bare `print()` calls with `logger.info()`.

**Acceptance criteria:**
- `artifacts/logs/` is populated after every phase script run.
- Each log file captures dataset shape, k sweep progress, metric values, and artifact paths.
- No bare `print()` calls in phase scripts (tqdm progress bars are exempt).

**Do before:** Phase 3 implementation (so all future phases are logged from the start).

---

### [PROCESS] FAQ document must be kept current

**Rule (by design):** Whenever a methodological question is asked during implementation — "why does X work this way?", "is the speed correct?", "how is K chosen?", "why use this default?" — the answer must be written into `documents/faq.md` before the conversation ends.

**Current FAQ location:** `documents/faq.md`

**What goes in the FAQ:** Explanations that are not obvious from reading the code — methodology rationale, expected behavior, metric interpretation, assignment constraints. Not: code patterns, file paths, or implementation details already in the execution guide.

**Do not create a new FAQ section inline in this TODO.** Add the entry directly to `documents/faq.md`.

---

### [DECISION] Freeze Phase 2 K selections before Phase 3

**Status:** Decided — pending written confirmation in execution guide.

Frozen K values (chosen label-free from Phase 2 metrics, see `documents/faq.md`):

| Dataset | Algorithm | K |
|---------|-----------|---|
| Wine    | KMeans    | 2 |
| Wine    | GMM       | 7 |
| Adult   | KMeans    | 8 |
| Adult   | GMM       | 7 |

**Action:** Update `ul_execution_guide.md` Phase 2 status block and Phase 4 inputs to reflect these frozen values before implementing Phase 3.

---

## Phase 3 Pre-requisites (must be true before starting)

- [ ] Logging wired (`src/utils/logger.py` + calls in all existing scripts)
- [ ] Phase 2 K selections documented in execution guide (table above)
- [ ] `documents/faq.md` up to date (all Phase 2 questions answered)
- [ ] `wine_kmeans.csv` verified: has `inertia` column (done 2026-03-20)
- [ ] 4 PNG figures verified in `artifacts/figures/phase2_clustering/` (done 2026-03-20)

---

## Resolved

| Date       | Item |
|------------|------|
| 2026-03-20 | Added `inertia` to `run_kmeans_sweep` output |
| 2026-03-20 | Created `src/utils/plotting.py` with `plot_kmeans_sweep` and `plot_gmm_sweep` |
| 2026-03-20 | Phase 2 script generates 4 PNGs in `artifacts/figures/phase2_clustering/` |
| 2026-03-19 | Phase 0 audit complete — all feature/architecture constants locked |
