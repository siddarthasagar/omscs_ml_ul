# Reviewer Response Analysis

Date: 2026-04-26

## Purpose

This document summarizes what actually went wrong in the original UL report,
why those problems cost points, and what needs to happen next to maximize the
 reviewer response outcome.

The goal is not to rewrite the whole project. The goal is to fix the specific
graded gaps that the reviewer identified and to avoid spending time reworking
sections that already earned full credit.

## Missed Points Summary

| Area | Points Lost | What the reviewer said | Actual underlying issue |
|---|---:|---|---|
| A3. Graph/Text Legibility | 2 | Figures and/or writing were not readable without zoom | The report packed too many multi-panel figures into a compressed IEEE layout, making labels and legends too small |
| E. Step 3: DR then Clustering | 8 | Missing orderly performance visualization; cluster selection in reduced spaces was unclear; wanted Silhouette and AIC/BIC plots | Phase 4 reused raw-space frozen K values instead of re-selecting K in each reduced space, so the report could not show reduced-space selection sweeps |
| G. Step 5: Clustering on NN | 6 | Missing wall-clock analysis and filtering-effect discussion for EM and KMeans cluster features | The Phase 6 pipeline did not record training times, so the report only had F1 evidence and no timing evidence |

Total missed report points: 16

Maximum recoverable via reviewer response: 8 points

## What Went Wrong

### 1. Step 3 was implemented as a fixed-K comparison, not as a full repeat of Step 1

This was the biggest problem.

The assignment wording for Step 3 says to re-apply clustering after dimensionality
reduction. In practice, that means the reduced-space versions need the same kind of
hyperparameter-selection justification used in Step 1. The reviewer explicitly asked
for reduced-space Silhouette and AIC/BIC plots to justify the new cluster counts.

The project drifted away from that requirement internally:

- `documents/specs/phase4_reduced_clustering/requirements.md` says Phase 4 should use frozen K values with no re-selection.
- `documents/adr/adr-002-frozen-k-values.md` carries that decision forward.
- `scripts/run_phase_4_reduced_cluster.py` applies Phase 2 frozen K values directly in reduced spaces.

That design produced valid fixed-K comparison artifacts, but it did not produce the
selection evidence the grader expected. As a result, the report could show summary
results and interpretation, but not the reduced-space cluster-selection process.

Why this mattered:

- The analysis itself was good enough to earn the pairwise-comparison points.
- The missing points came from missing experimental justification, not weak prose.
- A better figure alone would not have fixed this. The experiment design itself has to change.

### 2. Step 3 visualization was too compressed for the specific question being graded

The report only presented:

- a summary table of reduced-space silhouettes, and
- a single heatmap combining all 12 combinations.

That heatmap is useful as a compact summary, but it is not a substitute for showing how
K was chosen in each reduced space. It hides the sweep process that Step 1 showed clearly.

In other words, the report gave the answer, but not the evidence trail the grader wanted to verify.

### 3. Step 5 had performance evidence but no timing instrumentation

The reviewer wanted wall-clock analysis and some discussion of the filtering effect.

Step 4 had timing support because `scripts/run_phase_5_nn_reduced.py` records
`train_time_s` per seed. Step 5 did not: `scripts/run_phase_6_nn_cluster_features.py`
stores only F1 metrics. Because the timing data never existed in the artifact table,
the report could not make a defensible timing claim for cluster-derived features.

Why this mattered:

- The reviewer accepted the predictive-performance discussion.
- The missing points were specifically for wall-clock explanation.
- This is an instrumentation gap first, and a writing gap second.

### 4. Legibility suffered from layout pressure, not from one single bad plot

The report used aggressive float compression and many 2x2 figure grids inside an IEEE
two-column format. The individual source plots are mostly readable on their own, but the
PDF likely crossed the grader's non-zoom threshold once those plots were scaled down.

This means the fix is not just "increase caption size" or "write a clearer caption."
The revised report needs fewer panels per figure, larger rendered axis/legend text, and
more selective use of full-width figures.

## Root Causes

### Spec drift

The project's internal Phase 4 requirements diverged from the assignment's Step 3 intent.
That caused the script, artifacts, and report to all line up with the wrong procedure.

### Missing instrumentation

Phase 6 tracked accuracy but not runtime, so the report had no timing evidence to cite.

### Over-optimization for page limit

The report successfully stayed within the page limit, but some figures were compressed so
hard that readability dropped below the grading bar.

### Evidence mismatch

The report often had strong interpretation, but the grader wanted visible experimental
justification for specific decisions. Where that justification was missing, the prose did
not compensate.

## What Does Not Need Major Rework

These areas already scored well and should only need light touch-ups, if any:

- Hypothesis framing and follow-through
- Step 1 raw clustering analysis
- Step 2 dimensionality reduction analysis
- Step 4 NN-on-DR interpretation
- Extra credit t-SNE section

This matters because the reviewer response should focus effort where points were actually lost.

## What Must Happen Next

### Priority 1: Fix Step 3 correctly

1. Update the Phase 4 methodology so each reduced representation gets its own clustering sweep.
2. Re-run KMeans and GMM sweeps on each reduced space.
3. Select reduced-space K values using the same label-free logic used in Step 1.
4. Generate reduced-space selection plots for Wine and Adult.
5. Replace or supplement the current heatmap with figures that show the selection process clearly.
6. Rewrite the Step 3 report section so it distinguishes:
   - reduced-space K selection,
   - final reduced-space comparison, and
   - interpretation of why K changed or stayed stable.

Without this, the reviewer response will likely only earn partial credit.

### Priority 2: Add Step 5 timing data and analysis

1. Modify `scripts/run_phase_6_nn_cluster_features.py` to record per-seed `train_time_s`.
2. Regenerate the Phase 6 comparison table and metadata.
3. Add timing columns or summary timing statistics to the report table.
4. Add a short paragraph explaining whether cluster features changed runtime and why.
5. If no real speedup appears, state that directly rather than forcing a claim.

Important nuance:

- Because the cluster features were appended to the raw 12-dimensional input, this setup does
  not reduce input dimension.
- Any timing difference would come from optimization behavior or easier routing/filtering, not
  from a smaller first-layer matrix multiply.

### Priority 3: Fix report legibility

1. Split the densest 2x2 figures into fewer panels per figure.
2. Use larger plot text and legend text at export time.
3. Give the most important Step 3 and Step 5 figures more space in the PDF.
4. Recompile and inspect the final PDF at normal zoom, not just in Overleaf preview.

### Priority 4: Write the actual two-page reviewer response

The response should be specific and evidence-based.

It should explicitly say:

- what was changed,
- what new experiments were run,
- what figures/tables were added or replaced,
- what conclusions changed, and
- where the revised report now addresses each reviewer comment.

The response document should not argue that the old report was "basically fine."
It should show concrete good-faith correction.

## Likely Files To Touch

Methodology and experiments:

- `documents/specs/phase4_reduced_clustering/requirements.md`
- `scripts/run_phase_4_reduced_cluster.py`
- `scripts/run_phase_6_nn_cluster_features.py`
- `src/utils/plotting.py` (if plot sizes/fonts need adjustment)

Artifacts that will likely be regenerated:

- `artifacts/metrics/phase4_clustering/*`
- `artifacts/figures/phase4_clustering/*`
- `artifacts/metrics/phase6_nn_cluster/*`
- `artifacts/figures/phase6_nn_cluster/*`
- `artifacts/metadata/phase4.json`
- `artifacts/metadata/phase6.json`
- `artifacts/tables/tab_phase4_silhouette.tex`
- `artifacts/tables/tab_phase6_nn.tex`
- `artifacts/tables/report_numbers.tex`

Report files:

- `REPORT_UL/UL_Report_schinne3.tex`
- `REPORT_UL/tables/*`
- `REPORT_UL/figures/*`

Reviewer-response deliverable:

- `Reviewer_Response/UL_Report_Reviewer_Response_schinne3.pdf`

## Practical Recommendation

If time is limited, do not spend it polishing already-full-credit sections.

The highest-value path is:

1. Rebuild Step 3 correctly.
2. Add Step 5 timing.
3. Improve figure readability.
4. Then write the two-page response around those concrete changes.

That sequence addresses all three deduction buckets directly and gives the best chance of
earning the full reviewer-response recovery.