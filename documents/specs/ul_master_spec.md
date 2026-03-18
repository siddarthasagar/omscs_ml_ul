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

### Algorithm Scope

Required algorithms:

- Clustering: `K-Means`, `Expectation Maximization / Gaussian Mixture Model`
- Linear dimensionality reduction: `PCA`, `ICA`, `Random Projection`

Extra-credit track:

- Non-linear manifold learning: `t-SNE`

### Tooling Scope

- Use `scikit-learn` for clustering, dimensionality reduction, and evaluation helpers.
- Use `PyTorch` only for the neural-network experiments in Steps 4 and 5.
- Reuse prior SL/OL project knowledge from `/Users/siddarthasagarchinne/github/omscs_ml_ol`, but do not trust it blindly where code and writeups disagree.

### Experiment Flow

- Use a two-phase process:
  - exploratory sweeps on train/validation only
  - report-grade repeated runs after settings are frozen

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

Risk:

- the OL repo code comments suggest `type` may have been kept, which would yield 12 features

Required resolution rule:

- before implementation begins, audit the prior OL backbone and checkpoints
- if the checkpoint expects 12 inputs, treat the checkpoint as non-authoritative for UL and rebuild the raw-feature Wine NN baseline under the report-aligned 11-feature contract

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

Use Wine only.

Compare the same NN family on:

- raw Wine features
- PCA features
- ICA features
- RP features

Rules:

- keep train/validation/test split discipline fixed
- keep optimizer and training controls fixed unless the baseline audit proves the prior OL setup is inconsistent with the canonical Wine contract
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

## Required External Dependencies

- Prior OL repo: `/Users/siddarthasagarchinne/github/omscs_ml_ol`
- Assignment docs in `documents/canvas/assignment/md`
- Supplemental readings in `documents/canvas/Supplemental_Readings`

## Open Issues To Track

1. Confirm whether the prior OL Wine baseline truly used 11 or 12 input features.
2. Confirm the exact optimizer and stopping controls to inherit for the Wine NN baseline.
3. Confirm whether final report-grade repetitions should mirror the OL 10-seed convention or use a smaller UL-specific repetition budget.
