## Project
CS7641 Spring 2026 Unsupervised Learning. Python 3.13, uv, scikit-learn + PyTorch.
Datasets: data/adult.csv, data/wine.csv.

## Current Status
- Phase 0: [x] COMPLETE — all blockers resolved 2026-03-19
- Phase 1: [x] COMPLETE — data loaders, preprocessing, Gate 1 tests passing
- Phase 2: [x] COMPLETE — raw clustering sweep done, 4 CSVs + 4 PNGs, K values frozen 2026-03-20
- Phase 3: [x] COMPLETE — 6 CSVs + 6 PNGs, n_components frozen 2026-03-21
- Phase 4: [x] COMPLETE — 12 combinations run, summary_table.csv + 3 PNGs produced 2026-03-21
- Phase 5–7: [ ] ready to implement
Update this block at end of every session.

## Session Rules (enforced every conversation)
1. **FAQ rule:** Any methodological question asked and answered must be written to `documents/faq.md` before the session ends.
2. **TODO rule:** Any bug, blocker, or cross-cutting issue belongs in `documents/TODO.md`.
3. **Frozen K values (ADR-002):** Wine KMeans=2, Wine GMM=7, Adult KMeans=8, Adult GMM=7. Do not re-select without explicit user instruction.
4. **Logging:** All phase scripts must call `configure_logger(run_id)` from `src/utils/logger.py`. Do not implement a new phase script without wiring logging first.
5. **Spec-first:** Before implementing any phase, confirm requirements.md and design.md in the relevant `documents/specs/phase{N}_*/` are current.

## Key File Locations

### Always-loaded steering (tech constraints, product goals, structure)
documents/steering/product.md          — objective, hypotheses, algorithm scope, success criteria
documents/steering/tech.md             — stack, seeds, Wine contract, NN config, frozen K values
documents/steering/structure.md        — repo layout, module contracts, naming conventions

### Per-phase specs (requirements + design + tasks)
documents/specs/phase2_clustering/     — ✅ COMPLETE
documents/specs/phase3_reduction/      — [ ] next
documents/specs/phase4_reduced_clustering/
documents/specs/phase5_nn_reduced/
documents/specs/phase6_nn_cluster/
documents/specs/phase7_tsne/           — optional (extra credit)

### Frozen decisions (do not edit)
documents/adr/adr-001-wine-12-features.md
documents/adr/adr-002-frozen-k-values.md
documents/adr/adr-003-logging-pattern.md

### Cross-phase trackers
documents/TODO.md                      — open issues and blockers
documents/faq.md                       — methodological Q&A

### Reference (read-only)
documents/canvas/assignment/md/UL_Report_Spring_2026_v2.md
documents/canvas/assignment/md/UL_Report_Spring_2026_FAQ_v2.md
documents/REPORT_OL/OL_Report_schinne3.tex
/Users/siddarthasagarchinne/github/omscs_ml_ol  (prior OL repo)

### Archive (stale — do not use)
documents/_archive/                    — old monolithic spec files

## Build Commands
make dev                                       # venv + all deps
make test                                      # full pytest
make lint / make format                        # ruff
uv run pytest tests/test_data.py -v            # Gate 1
uv run pytest tests/test_unsupervised.py -v    # Gate 2
