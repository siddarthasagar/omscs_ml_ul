# UL Implementation Plan

## Goal

Convert the UL master spec into a reproducible implementation and report-ready analysis workflow without drifting from the assignment contract or the prior project baseline.

## Phase Breakdown

### Phase 0: Baseline Audit And Environment Lock

Objective:

- resolve inherited ambiguity before starting experiments

Tasks:

- inspect the OL repo’s Wine preprocessing path, backbone definition, and any saved-model input assumptions
- confirm whether the report-aligned Wine contract is 11 features or whether the old checkpoint was trained with 12
- lock the canonical split logic, random seeds, optimizer, batch size, and stopping rule for the Wine NN follow-on experiments
- document any justified divergence from OL in the reproducibility notes

Exit criteria:

- one authoritative Wine baseline contract exists
- any mismatch between code and old report text is explicitly resolved

### Phase 1: Data Loading And Preprocessing Backbone

Objective:

- build one reliable preprocessing contract per dataset for all later UL experiments

Tasks:

- Adult:
  - load raw data
  - separate target
  - split into train/validation/test
  - one-hot categorical columns
  - standardize numeric columns
- Wine:
  - load raw data
  - normalize target choice
  - drop duplicate target field
  - drop `type` under the canonical 11-feature contract
  - standardize continuous inputs
- ensure all preprocessors fit on training only

Outputs:

- reproducible dataset loaders
- saved metadata describing feature counts and split sizes

Exit criteria:

- feature counts and label distributions are stable and documented
- no-leakage checks pass

### Phase 2: Step 1 Raw Clustering

Objective:

- characterize cluster structure in the original feature spaces

Tasks:

- run K-Means sweeps on Adult and Wine
- run GMM sweeps on Adult and Wine
- evaluate K-Means with silhouette, Calinski-Harabasz, Davies-Bouldin, and seed stability
- evaluate GMM with BIC, AIC, and seed stability
- freeze one primary K-Means setting and one primary GMM setting per dataset

Outputs:

- per-dataset raw clustering comparison table
- cluster selection rationale
- initial hypothesis notes

Exit criteria:

- one chosen raw-data K-Means model and one chosen raw-data GMM model per dataset

### Phase 3: Step 2 Raw Dimensionality Reduction

Objective:

- characterize how each linear DR method changes the data geometry

Tasks:

- PCA on Adult and Wine
- ICA on Adult and Wine
- RP on Adult and Wine with repeated random seeds
- choose component counts with method-specific rules:
  - PCA: explained variance and elbow review
  - ICA: kurtosis and convergence
  - RP: reconstruction error and stability

Outputs:

- DR diagnostic plots and tables
- frozen component count per method and dataset
- notes on interpretability and expected downstream effects

Exit criteria:

- one chosen PCA dimensionality, ICA dimensionality, and RP dimensionality per dataset

### Phase 4: Step 3 Clustering In Reduced Spaces

Objective:

- compare raw-space and reduced-space clustering quality

Tasks:

- re-run K-Means and GMM on each PCA, ICA, and RP representation for both datasets
- summarize all dataset x DR x clustering combinations in compact tables
- identify whether reduced spaces improve stability, separation, or interpretability

Outputs:

- summary tables for the 12 reduced-space clustering combinations
- short list of strongest and weakest combinations

Exit criteria:

- clear ranking or narrative of which reduced spaces help clustering and which do not

### Phase 5: Step 4 Wine NN On Reduced Inputs

Objective:

- measure whether linear dimensionality reduction changes Wine NN behavior

Tasks:

- establish a raw-feature Wine NN baseline under the audited canonical contract
- retrain the same NN family on PCA, ICA, and RP Wine inputs
- keep split logic and training controls fixed
- compare:
  - Macro-F1
  - accuracy
  - training time
  - convergence behavior

Outputs:

- raw vs PCA vs ICA vs RP Wine NN comparison table
- training-behavior plots or compact convergence diagnostics

Exit criteria:

- one defensible conclusion about whether DR helps or hurts the Wine NN baseline

### Phase 6: Step 5 Wine NN With Cluster-Derived Features

Objective:

- test whether cluster information adds predictive value on Wine

Tasks:

- derive one-hot K-Means assignment features
- derive K-Means centroid-distance features
- derive GMM posterior-probability features
- append each feature family to the canonical raw Wine feature set
- retrain the same Wine NN family on each augmented representation

Outputs:

- comparison of raw baseline vs each cluster-derived augmentation
- analysis of whether hard or soft cluster information is more useful

Exit criteria:

- one defensible conclusion about whether cluster-derived features add useful signal

### Phase 7: Extra Credit t-SNE

Objective:

- add a nonlinear visualization layer without contaminating the core methodology

Tasks:

- run t-SNE on Adult and Wine using the canonical processed inputs
- generate 2D visualizations with label overlays and optional cluster overlays
- use the visuals only as explanatory support

Outputs:

- report-ready t-SNE figures
- short interpretation notes

Exit criteria:

- visuals provide insight without becoming the basis for core model selection

## Experiment Staging

### Exploratory Stage

Use reduced budgets and fast sweeps to select promising settings.

Rules:

- selection uses training and validation only
- labels are not used for cluster/component selection
- any unstable configuration is discarded early

### Report-Grade Stage

Run only frozen settings with repeated seeds.

Default seed plan:

- start with split seed `42`
- use repeated seeds `42` to `51` unless later runtime constraints force a documented reduction

## Planned Artifact Set

The implementation should eventually produce:

- dataset metadata summaries
- clustering metric tables
- DR diagnostic tables and plots
- reduced-space clustering comparison tables
- Wine NN comparison tables for Step 4
- Wine NN comparison tables for Step 5
- t-SNE figures
- reproducibility notes with commands, seeds, and data assumptions

## Validation Gates

The work should not advance to the next phase unless the current gate passes.

### Gate 1: Baseline Integrity

- Wine feature-contract mismatch resolved
- preprocessing contract documented
- split discipline fixed

### Gate 2: Unsupervised Integrity

- cluster/component selection is label-free
- train-only fitting confirmed
- unstable settings removed

### Gate 3: Supervised Follow-On Integrity

- Step 4 and Step 5 use only Wine
- same NN family used across raw and transformed comparisons
- metric reporting includes Macro-F1 and convergence behavior

### Gate 4: Report Integrity

- every figure/table has a clear takeaway
- the strongest results and strongest failures are both represented
- the 8-page report cap is respected by prioritizing comparison tables over redundant plots

## Risks And Mitigations

### Risk: Wine baseline mismatch contaminates comparisons

Mitigation:

- resolve baseline before writing any comparative claims

### Risk: Adult clustering is hard to interpret

Mitigation:

- emphasize geometry limits and preprocessing effects rather than forcing artificial semantic cluster stories

### Risk: Report bloat

Mitigation:

- prefer compact comparison tables and a small number of high-signal figures

### Risk: Random Projection results vary across seeds

Mitigation:

- treat RP as a repeated method and report stability explicitly

## Immediate Next Actions

1. Audit the OL Wine baseline and settle the 11-vs-12 feature contract.
2. Convert this plan into concrete implementation tasks and file structure once coding begins.
3. Start the literature and source-collection pass for the required peer-reviewed references.
