# TODO — CS7641 UL Spring 2026

Tracks cross-cutting issues, blocked items, and decisions that do not belong in phase scripts.
Update this file whenever an item is resolved or a new issue surfaces.

---

## Open Items

_None currently. Next phase: Phase 4 — Clustering in Reduced Spaces._

---

## Resolved

| Date       | Item |
|------------|------|
| 2026-03-21 | Phase 3 complete — 6 CSVs + 6 PNGs, n_components frozen (Wine: PCA=8 ICA=4 RP=8 / Adult: PCA=22 ICA=11 RP=22) |
| 2026-03-21 | Specs reorganized — Kiro-style steering/specs/adr structure, old monolithics archived |
| 2026-03-20 | Phase 2 K values frozen — documented in ADR-002 (supersedes "pending execution guide" action) |
| 2026-03-20 | FAQ process established — `documents/faq.md` created, CLAUDE.md rule + Stop hook wired |
| 2026-03-20 | Implemented `src/utils/logger.py`; wired into Phase 2 script |
| 2026-03-20 | Added `inertia` to `run_kmeans_sweep` output |
| 2026-03-20 | Created `src/utils/plotting.py` with `plot_kmeans_sweep` and `plot_gmm_sweep` |
| 2026-03-20 | Phase 2 script generates 4 PNGs in `artifacts/figures/phase2_clustering/` |
| 2026-03-19 | Phase 0 audit complete — all feature/architecture constants locked |
