# UL Planning Pack

This directory stores the planning documents for the OMSCS ML Unsupervised Learning assignment.

## Documents

- `ul_master_spec.md`
  The authoritative feature specification for the assignment work in this repo.
- `ul_research_notes.md`
  Distilled assignment requirements, inherited project context from SL/OL, and research-driven hypotheses.
- `ul_implementation_plan.md`
  The execution plan, phase breakdown, artifact checklist, and validation gates.

## Current Direction

- Assignment focus: CS7641 Unsupervised Learning and Dimensionality Reduction Report, Spring 2026.
- Datasets: Adult Income and Wine Quality are both mandatory for Steps 1 to 3.
- Wine is the locked default dataset for Steps 4 and 5.
- Extra credit is included in scope via a `t-SNE` analysis track.
- Default stack: `scikit-learn` for clustering and dimensionality reduction, `PyTorch` only for the neural-network follow-on experiments.

## Important Planning Note

The prior OL repo contains a Wine preprocessing inconsistency:

- the written SL/OL reports describe Wine as an 11-feature physicochemical dataset with `type` removed
- the OL code comments suggest `type` may have been retained

This planning pack treats that mismatch as a first-class prerequisite to resolve before implementation begins.
