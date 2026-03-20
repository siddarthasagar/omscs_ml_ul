# TODO — CS7641 UL Spring 2026

Tracks cross-cutting issues, blocked items, and decisions that do not belong in phase scripts.
Update this file whenever an item is resolved or a new issue surfaces.

---

## Open Items

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

- [x] Logging wired (`src/utils/logger.py` + Phase 2 script updated 2026-03-20)
- [ ] Phase 2 K selections documented in execution guide (table above)
- [x] `documents/faq.md` created, Phase 2 Q&As written, Stop hook enforces rule going forward
- [x] `wine_kmeans.csv` verified: has `inertia` column (2026-03-20)
- [x] 4 PNG figures verified in `artifacts/figures/phase2_clustering/` (2026-03-20)

---

## Resolved

| Date       | Item |
|------------|------|
| 2026-03-20 | FAQ process established — `documents/faq.md` created, CLAUDE.md rule + Stop hook wired |
| 2026-03-20 | Implemented `src/utils/logger.py`; wired into Phase 2 script |
| 2026-03-20 | Added `inertia` to `run_kmeans_sweep` output |
| 2026-03-20 | Created `src/utils/plotting.py` with `plot_kmeans_sweep` and `plot_gmm_sweep` |
| 2026-03-20 | Phase 2 script generates 4 PNGs in `artifacts/figures/phase2_clustering/` |
| 2026-03-19 | Phase 0 audit complete — all feature/architecture constants locked |
