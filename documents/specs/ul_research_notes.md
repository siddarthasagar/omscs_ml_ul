# UL Research Notes

## Purpose

These notes translate the assignment, FAQ, and prior project evidence into implementation-relevant research guidance. They are not the final report. They exist to keep the future implementation and analysis tied to grounded hypotheses rather than ad hoc exploration.

## Assignment Distillation

The UL assignment is asking for more than algorithm execution. It wants a comparative analysis of:

- how clustering behaves on the raw datasets
- how linear dimensionality reduction changes structure
- how clustering behavior changes after dimensionality reduction
- how reduced or cluster-derived representations affect a previously established neural-network baseline

The report must therefore be designed around comparisons, not isolated experiments.

## Non-Negotiable Constraints

- Adult and Wine must both appear in Steps 1 to 3.
- Only one dataset may be used for Steps 4 and 5.
- Selection of cluster counts or component counts should be label-free.
- Test data must not be used to fit unsupervised transforms or clustering models.
- The writeup must be concise and formal. Large bullet lists in the final report are penalized.

## Why Wine Is The Right NN Follow-On Dataset

Wine was selected for Steps 4 and 5 because the prior SL and OL work already showed a stronger structural case for unsupervised analysis:

- the feature space was described as heavily overlapping
- the silhouette score on standardized features was reported as negative
- linear and non-linear supervised baselines both remained under 0.50 Macro-F1
- the earlier writeups repeatedly described adjacent quality classes as chemically hard to separate

This makes Wine the better candidate for asking whether representation learning or cluster-derived information can help.

## Inherited Project Context From SL And OL

### Adult

Prior findings:

- large tabular dataset
- mixed numeric and categorical features
- approximately 96 dimensions after one-hot encoding
- class imbalance near 3:1
- prior reports argued the decision boundary is close to linear

Implication for UL:

- dimensionality reduction may help compression and interpretability more than raw predictive gain
- clustering quality may be weaker or harder to interpret because one-hot sparsity can distort Euclidean geometry

### Wine

Prior findings from report text:

- 6,497 rows
- 11 physicochemical features
- 8 quality classes
- severe overlap among adjacent quality labels
- low supervised ceiling across multiple model families

Implication for UL:

- clustering may not align cleanly with labels, but that is itself analytically useful
- PCA, ICA, and RP may reveal different structure or failure modes
- cluster-derived features may capture coarse regimes even when direct class prediction remains difficult

## Critical Baseline Mismatch

The prior OL repo contains a baseline inconsistency:

- the SL and OL report text describe Wine as 11 features with `type` removed
- the OL code comments suggest `type` may have been retained, which would create a 12-feature input
- the raw CSV contains `class`, `type`, and `quality`

Planning conclusion:

- the reports, not the comments, define the canonical analysis story
- however, the old checkpoint cannot be trusted without an audit
- the implementation plan must resolve this before comparing any UL follow-on NN results to the old baseline

## Algorithm-Specific Research Guidance

### K-Means

Useful when:

- cluster geometry is roughly spherical in the chosen feature space
- Euclidean distance is meaningful after preprocessing

Risks:

- sensitive to scaling
- weak on non-spherical or overlapping structure
- may produce unstable or uninformative clusters on sparse mixed-type Adult data

Recommended evaluation focus:

- silhouette score
- Calinski-Harabasz
- Davies-Bouldin
- qualitative stability across seeds

### GMM / EM

Useful when:

- clusters may be elliptical or soft rather than hard partitions
- posterior probabilities are useful downstream

Risks:

- covariance choice strongly affects fit
- can overfit component counts if only likelihood is used

Recommended evaluation focus:

- BIC
- AIC
- seed stability
- posterior interpretability

### PCA

Most useful for:

- variance concentration
- redundancy detection
- compact linear representations

Important evidence to collect:

- explained variance spectrum
- cumulative explained variance
- cluster behavior before and after projection

### ICA

Most useful for:

- finding non-Gaussian directions
- separating signals that PCA may blur

Important evidence to collect:

- absolute kurtosis of components
- convergence behavior
- whether component axes correspond to interpretable feature combinations

### Random Projection

Most useful for:

- computationally cheap compression
- geometry preservation in a distance-oriented sense

Important evidence to collect:

- reconstruction error using pseudo-inverse
- run-to-run variation across random seeds
- downstream clustering sensitivity

### t-SNE

Usefulness:

- strong for visualizing local neighborhoods and separation patterns

Risk:

- easily over-interpreted as a metric-preserving embedding

Planning rule:

- use as visualization and qualitative support only
- do not use for core model selection or NN training inputs

## Hypothesis Frame For The UL Report

### Adult Hypothesis

Adult is likely to show weak, coarse cluster structure after preprocessing because the one-hot encoded feature space mixes sparse categorical directions with scaled numeric features. PCA or RP may improve compactness and downstream clustering stability modestly, but strong semantic clusters are not expected. Any NN gain from reduced or cluster-derived features is expected to be limited.

### Wine Hypothesis

Wine is likely to benefit more from representation changes than from direct raw-feature classification because its labels overlap heavily in the original physicochemical space. PCA may capture dominant global variance, ICA may surface more discriminative non-Gaussian directions, and cluster-derived soft features from GMM may help the NN by exposing coarse latent regimes even if label purity remains low.

### Extra-Credit Hypothesis

t-SNE is unlikely to produce a feature space suitable for supervised training, but it may provide a clearer visual explanation of why some class boundaries remain weak and why certain clustering results are unstable or locally plausible.

## External Literature Targets

Minimum external reading goals for the final report:

- one peer-reviewed source supporting clustering or EM/GMM interpretation
- one peer-reviewed source supporting ICA or dimensionality reduction interpretation
- optional third source supporting random projection or manifold learning tradeoffs

Likely source directions based on materials already present in the repo:

- Hyvarinen and Oja on ICA
- a clustering survey for K-Means and EM/GMM interpretation
- Burges or related dimensionality reduction references for PCA/RP framing

## Research Risks

1. Adult clustering may be difficult to interpret meaningfully due to sparse mixed-type geometry.
2. Wine label overlap may remain so strong that cluster-label alignment looks weak even when unsupervised structure is real.
3. The baseline mismatch from the OL repo may otherwise contaminate Step 4 and Step 5 comparisons.
4. The 8-page report cap will force strict prioritization of visuals and tables.
