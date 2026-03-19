# UL Master Spec

## Status

Draft v1

## Objective

Produce a reproducible, assignment-complete analysis pipeline and report plan for the CS7641 Spring 2026 Unsupervised Learning assignment using the required Adult Income and Wine Quality datasets. The work must support both the analysis report and the reproducibility sheet while remaining consistent with prior SL and OL project decisions unless a divergence is explicitly justified.

## Assignment Contract

The implementation must satisfy the assignment requirements captured from:

- `documents/canvas/assignment/md/UL_Report_Spring_2026_v2.md`
- `documents/canvas/assignment/md/UL_Report_Spring_2026_FAQ_v2.md`

The contract is:

1. Use the same two datasets from the prior SL and OL reports.
2. Run clustering and dimensionality reduction on both datasets.
3. Re-run clustering after dimensionality reduction.
4. Run neural-network follow-on experiments on only one dataset.
5. Compare transformed-feature and cluster-derived-feature NN behavior against the prior best NN baseline.
6. Include a hypothesis-driven analysis, not a result dump.
7. Deliver an 8-page report plus a reproducibility document.
8. Include at least two peer-reviewed references beyond course materials.

## Locked Decisions

### Dataset Scope

- `Adult Income` and `Wine Quality` are both mandatory for Steps 1 to 3.
- `Wine Quality` is the locked dataset for Steps 4 and 5.

Dataset selection rationale for Steps 4–5:

- The v2 assignment (2026-03-11) allows either Adult or Wine for the NN follow-on experiments.
- Wine is deliberately locked here because the prior SL and OL work already established a structurally pathological case: heavily overlapping quality classes, sub-0.50 Macro-F1 ceiling, and a stated hypothesis that the raw feature space is insufficient.
- This makes Wine the more analytically interesting candidate for testing whether representation learning or cluster-derived information can move the needle.
- Adult is excluded from Steps 4–5 because the prior OL work yielded a near-linear decision boundary where further representation changes are less likely to produce analytically interesting outcomes.

### Algorithm Scope

Required algorithms:

- Clustering: `K-Means`, `Expectation Maximization / Gaussian Mixture Model`
- Linear dimensionality reduction: `PCA`, `ICA`, `Random Projection`

Extra-credit track:

- Non-linear manifold learning: `t-SNE`

## Canonical Data Contracts

### Adult Income

- Target: `class` / income label
- Features: mixed numeric and categorical tabular features from the prior SL/OL work
- Preprocessing plan:
  - split before fitting transforms
  - standardize numeric columns
  - one-hot encode categorical columns
  - fit transforms on training split only

### Wine Quality

Canonical planned representation:

- Target: one label column only, normalized to the class used in prior work
- Features: 11 physicochemical predictors
- Excluded fields:
  - duplicate target column
  - `type`

Rationale:

- the written SL and OL reports consistently describe the Wine problem as an 11-feature physicochemical task
- the user explicitly selected Wine because the earlier assignments found severe overlap in that feature space

Note: if the Phase 0 audit reveals the OL checkpoint was trained on 12 features, treat it as non-authoritative and rebuild the raw-feature Wine NN baseline under the canonical 11-feature contract before any Step 4 comparisons.

## Step-Level Specification

### Step 1: Clustering on Raw Data

Run `K-Means` and `GMM` separately on Adult and Wine.

Required outputs:

- chosen cluster/component counts
- preprocessing description
- clustering quality metrics
- stability commentary
- concise interpretation of cluster structure relative to dataset properties

Selection rule:

- cluster selection must remain label-free
- labels may be used only after selection for interpretation

### Step 2: Dimensionality Reduction on Raw Data

Run `PCA`, `ICA`, and `Random Projection` separately on Adult and Wine.

Required outputs:

- chosen dimensionalities
- selection rationale
- transformed-space observations
- method-specific diagnostics

Method-specific selection defaults:

- PCA: explained variance plus elbow review
- ICA: kurtosis, convergence behavior, and interpretability
- RP: reconstruction error and distance-preservation behavior across repeated random seeds

### Step 3: Clustering in Reduced Spaces

For each Step 2 representation, re-run Step 1 clustering.

Required outputs:

- comparison tables across raw and reduced spaces
- concise narrative about improved or degraded separation, stability, and interpretability
- explicit links back to stated hypotheses

### Step 4: Neural Networks on Reduced Data

Use Wine only (see dataset selection rationale under Locked Decisions).

Baseline definition:

- The reference model is the best Wine neural network from the OL report: a fully-connected PyTorch network trained with the Adam optimizer, using the learning rate, batch size, and early-stopping configuration that produced the best OL validation Macro-F1 on Wine.
- If the Phase 0 audit reveals that the OL checkpoint was trained on 12 features rather than 11, the baseline must be rebuilt under the canonical 11-feature contract before any Step 4 comparisons are made.

Compare this baseline on:

- raw Wine features (the rebuilt or inherited baseline)
- PCA-reduced Wine features
- ICA-reduced Wine features
- RP-reduced Wine features

Rules:

- keep train/validation/test split discipline fixed
- keep optimizer, learning rate, batch size, and stopping rule fixed across all input variants so only the input representation changes
- resize only the input layer as needed for transformed spaces

### Step 5: Neural Networks With Cluster-Derived Features

Use Wine only.

Primary feature-engineering variants:

- original Wine features + one-hot K-Means assignments
- original Wine features + K-Means centroid distances
- original Wine features + GMM posterior probabilities

Core comparison rule:

- the primary Step 5 plan uses appended features, not cluster-only inputs

Stretch-only variant:

- cluster-only features may be tested later if the appended-feature result is ambiguous

### Extra Credit: t-SNE

Use `t-SNE` as a visualization-first nonlinear manifold method on both datasets.

Rules:

- do not make `t-SNE` part of the core NN feature pipeline
- use it to support qualitative comparison and visualization
- if cluster overlays are shown, present them as exploratory evidence, not as the main selection criterion

## Success Criteria

The UL work is complete only if all of the following are true:

- every required assignment step has an explicit implementation and analysis plan
- both required datasets are used where mandated
- Wine-only NN scope is enforced for Steps 4 and 5
- no-leakage rules are explicit and followed
- the Wine feature-contract mismatch is resolved before any baseline comparison is trusted
- final outputs cover report-ready tables, figures, metrics, and reproducibility notes
- hypotheses are accepted or rejected with evidence, not implied informally

## Out of Scope

- writing the final report prose now
- implementing code in this spec document
- expanding the assignment to additional datasets
- replacing the Wine NN follow-on track with Adult
- using `t-SNE` as the core reduced input for supervised training
