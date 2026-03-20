---
inclusion: always
---

# Product — CS7641 UL Spring 2026

## Objective

Produce a reproducible, assignment-complete analysis pipeline and report for the CS7641 Spring 2026 Unsupervised Learning assignment using the Adult Income and Wine Quality datasets. Work must support both the analysis report and the reproducibility sheet, and must remain consistent with prior SL and OL project decisions unless a divergence is explicitly justified in an ADR.

## Assignment Contract

Derived from:
- `documents/canvas/assignment/md/UL_Report_Spring_2026_v2.md`
- `documents/canvas/assignment/md/UL_Report_Spring_2026_FAQ_v2.md`

Requirements:
1. Use the same two datasets from prior SL and OL reports.
2. Run clustering and dimensionality reduction on both datasets.
3. Re-run clustering after dimensionality reduction.
4. Run neural-network follow-on experiments on Wine only.
5. Compare transformed-feature and cluster-derived-feature NN behavior against prior OL baseline.
6. Hypothesis-driven analysis — not a result dump.
7. 8-page report + reproducibility document.
8. At least two peer-reviewed references beyond course materials.

## Dataset Scope

- **Adult Income + Wine Quality:** mandatory for Steps 1–3 (clustering + DR + re-clustering).
- **Wine Quality only:** locked for Steps 4–5 (NN follow-on).

**Rationale for Wine lock:** Wine has heavily overlapping quality classes and a sub-0.50 Macro-F1 ceiling in OL — the most analytically interesting candidate for testing whether representation learning can move the needle. Adult yielded a near-linear OL decision boundary where further representation changes are less likely to produce interesting outcomes.

## Algorithm Scope

| Category | Algorithms |
|----------|-----------|
| Clustering | K-Means, Gaussian Mixture Model (EM) |
| Linear DR | PCA, ICA, Random Projection |
| Extra credit | t-SNE (visualization only) |

## Hypotheses (to be accepted/rejected with evidence)

**Adult:** Likely to show weak, coarse cluster structure. OHE feature space mixes sparse categorical directions with scaled numeric features. PCA or RP may improve compactness modestly but strong semantic clusters are not expected.

**Wine:** Likely to benefit more from representation changes than from raw-feature classification because labels overlap heavily in the original physicochemical space. PCA may capture dominant global variance; ICA may surface more discriminative non-Gaussian directions; GMM soft posteriors may expose coarse latent regimes.

**t-SNE:** Unlikely to produce features suitable for supervised training, but may visually explain why certain class boundaries remain weak.

## Success Criteria

- Every required assignment step has an explicit implementation and analysis.
- Both datasets used where mandated; Wine-only NN scope enforced.
- No-leakage rules explicit and followed throughout.
- Wine 12-feature contract respected (see ADR-001).
- Frozen K values respected (see ADR-002).
- Final outputs: report-ready tables, figures, metrics, reproducibility notes.
- Hypotheses accepted or rejected with evidence.

## Out of Scope

- Writing final report prose during implementation.
- Implementing code inside spec documents.
- Expanding to additional datasets.
- Replacing Wine NN track with Adult.
- Using t-SNE as NN training input.
