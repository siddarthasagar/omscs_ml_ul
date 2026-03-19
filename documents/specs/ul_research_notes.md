# UL Research Notes

## Purpose

These notes translate the assignment, FAQ, and prior project evidence into implementation-relevant research guidance. They are not the final report. They exist to keep the future implementation and analysis tied to grounded hypotheses rather than ad hoc exploration.

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
