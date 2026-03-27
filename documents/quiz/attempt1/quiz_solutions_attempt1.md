# CS-7641 UL Unit Quiz — Complete Solutions
**Score: 14.87 / 28 | Attempt 1**
> ⚠️ Correct answers are hidden. Analysis is based on ML theory + your exact selections read from screenshots.

**Legend:** ✅ Correct selection | ❌ Wrong selection | ➕ Should have selected (missed) | ⬜ Correctly left blank

---

## Question Set 1: Clustering

---

### Question 1 — 0/2 pts ❌

**Scenario:**
> You are analyzing a high-dimensional dataset (d = 100) with the following characteristics:
> - There are three underlying clusters with very different sizes (one large, two small).
> - The data contains moderate Gaussian noise.
> - The clusters are roughly spherical in their original feature space.
>
> You are tasked with selecting a clustering method that is robust in high-dimensional, noisy, and imbalanced settings.
> **Which of the following statements are true in this scenario?**

**Your selections: 1 ✅, 2 ❌, 3 ✅, 4 ⬜, 5 ❌, 6 ➕**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| 1 | K-Means may be biased toward larger clusters due to its use of Euclidean distance and sum-of-squares optimization. | ✅ Selected | **TRUE** | K-Means minimizes total within-cluster SSE. Larger clusters dominate the objective, pulling centroids toward them. |
| 2 | Single linkage clustering is generally preferred in high-dimensional noisy settings due to its chaining property. | ❌ Selected | **FALSE** | The chaining property is a *weakness* — one noisy point can bridge two distant clusters. Single linkage is actively avoided in noisy, high-dimensional settings. |
| 3 | GMM with EM can handle unequal cluster sizes better than K-Means by learning separate covariance structures and mixing coefficients. | ✅ Selected | **TRUE** | GMMs learn per-cluster mixing weights π_j (size) and covariance Σ_j (shape). K-Means assumes equal, spherical clusters with no such flexibility. |
| 4 | Dimensionality has little effect on clustering because K-Means scales naturally to high dimensions. | ⬜ Skipped | **FALSE** | Correctly skipped. Curse of dimensionality makes all Euclidean distances concentrate — breaking K-Means distance-based assignments. |
| 5 | Using PCA before clustering will always improve clustering results by preserving local cluster structures. | ❌ Selected | **FALSE** | "Always" is wrong. PCA maximizes variance, not cluster separability. If discriminative signal lives in low-variance directions, PCA discards it. PCA preserves *global* variance, not *local* cluster structure. |
| 6 | Density-based methods like DBSCAN may fail in this scenario due to varying cluster sizes and the curse of dimensionality. | ➕ Missed | **TRUE** | Two reasons: (a) In d=100, ε-ball density estimates degrade — everything looks sparse. (b) One large + two small clusters makes a single global ε/minPts threshold inadequate. |

**✅ Should have selected: 1, 3, 6**
**❌ Mistakes: Selected 2 (chaining is a weakness) and 5 (PCA doesn't "always" help). Missed 6.**

---

### Question 2 — 0/2 pts ❌

**Scenario:**
> Two founders, Mr. Mayonnaise and Dr. Mustard, are competing in the premium condiments market. They've collected store-level and customer-level features (e.g., weekly unit sales, price, promo spend, shelf space, store size, region, and simple product flags like "organic"/"spicy"). They want to discover actionable market segments using K-Means and GMMs.
> **Which statements are true in this context? Select all that apply.**

**Your selections: A ✅, B ❌, C ⬜, D ❌, E ⬜, F ✅**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| A | Standardizing numeric features (e.g., z-scoring sales, price, and promo) is important so no single scale-dominant variable dictates distances. | ✅ Selected | **TRUE** | K-Means uses Euclidean distance. Weekly unit sales (thousands) would completely dominate binary 0/1 flags without standardization. |
| B | Because GMMs are probabilistic, they inherently handle unscaled features without issues. | ❌ Selected | **FALSE** | Probabilistic ≠ scale-invariant. GMMs estimate covariance matrices from data — a feature on a huge scale distorts the covariance structure just as badly as in K-Means. |
| C | Adding dozens of one-hot flags for every minor product variant always improves K-Means segmentation. | ⬜ Skipped | **FALSE** | Correctly skipped. Sparse high-dimensional one-hot vectors degrade Euclidean distance meaningfulness and add noise. |
| D | Choosing the number of clusters k by maximizing training likelihood alone guarantees the most useful segments. | ❌ Selected | **FALSE** | Training likelihood increases monotonically with k — at k=n each point is its own cluster and likelihood is maximized trivially. BIC/AIC/silhouette/elbow are needed. |
| E | Increasing k always yields more homogeneous and more actionable segments for marketing. | ⬜ Skipped | **FALSE** | Correctly skipped. Higher k → more homogeneous but micro-segments become hard to act on and may overfit noise. |
| F | K-Means is sensitive to initialization; multiple restarts or k-means++ help avoid poor local minima. | ✅ Selected | **TRUE** | K-Means only guarantees a local minimum of SSE. k-means++ seeds centroids with probability proportional to D², giving much better starting points. |

**✅ Should have selected: A, F only**
**❌ Mistakes: Selected B (GMMs still need scaling) and D (training likelihood alone is not enough).**

---

### Questions 3–6 — K-Means & EM Numerical

**Scenario:**
> You are given the following 1D dataset of points: **D = {1, 2, 3, 8, 9, 10, 16, 17, 18}**
> Assume the data come from three clusters. You will compare one iteration of K-Means and EM for Gaussian Mixture Models (GMM).
>
> **K-Means initial centroids:** μ₁ = 2, μ₂ = 9, μ₃ = 17
>
> **EM initial parameters:**
> - Means: μ₁ = 2, μ₂ = 9, μ₃ = 17
> - Shared, known variance: σ² = 9 (i.e., standard deviation σ = 3)
> - Mixing coefficients: π₁ = π₂ = π₃ = 1/3
>
> **New data point: x_new = 7**

---

### Question 3 — 1/1 pts ✅

**Question:** Based on Euclidean distance, which cluster would x_new = 7 be assigned to?

**Calculation:**
```
|7 − μ₁| = |7 − 2|  =  5
|7 − μ₂| = |7 − 9|  =  2  ← minimum
|7 − μ₃| = |7 − 17| = 10
```

**✅ Answer: Cluster 2** — correct.

---

### Questions 4, 5, 6 — EM Responsibilities — 1/1 each ✅

**Question 4:** Under EM, what is the (normalized) responsibility of Cluster 1? Round to 3 decimal places.
**Question 5:** Under EM, what is the (normalized) responsibility of Cluster 2? Round to 3 decimal places.
**Question 6:** Under EM, what is the (normalized) responsibility of Cluster 3? Round to 3 decimal places.

**E-step formula:**

$$\gamma(z=j \mid x=7) = \frac{\pi_j \cdot \mathcal{N}(7;\,\mu_j,\,\sigma^2)}{\displaystyle\sum_{k=1}^{3} \pi_k \cdot \mathcal{N}(7;\,\mu_k,\,\sigma^2)}$$

Since all π_j = 1/3 and the 1/√(2πσ²) prefactor is identical for all clusters, it cancels. Only the exponent terms matter:

**Step 1 — Unnormalized values** (σ² = 9):
```
Cluster 1: exp(−(7−2)²  / 18) = exp(−25/18)  = exp(−1.3889) = 0.24935
Cluster 2: exp(−(7−9)²  / 18) = exp(−4/18)   = exp(−0.2222) = 0.80073
Cluster 3: exp(−(7−17)² / 18) = exp(−100/18) = exp(−5.5556) = 0.00387
```

**Step 2 — Sum:**
```
Total = 0.24935 + 0.80073 + 0.00387 = 1.05395
```

**Step 3 — Normalize:**
```
γ(Cluster 1) = 0.24935 / 1.05395 = 0.2366
γ(Cluster 2) = 0.80073 / 1.05395 = 0.7597
γ(Cluster 3) = 0.00387 / 1.05395 = 0.0037
```
**Check:** 0.2366 + 0.7597 + 0.0037 = 1.000 ✅

| Question | Correct Answer | Your Answer | Result |
|---|---|---|---|
| Q4 — Cluster 1 | **0.2366** | 0.2366 | ✅ |
| Q5 — Cluster 2 | **0.7597** | 0.7597 | ✅ |
| Q6 — Cluster 3 | **0.0037** | 0.0037 | ✅ |

---

## Question Set 2: Feature Selection

---

### Question 7 — 1.5/2 pts ⚠️ Partial

**Scenario:**
> You are developing a classifier for a biomedical dataset with 500 features and only 120 labeled samples. You aim to improve generalization and interpretability by selecting a subset of features before training.
> **Which of the following statements are true?**

**Your selections: 1 ➕, 2 ✅, 3 ⬜, 4 ✅, 5 ⬜, 6 ✅**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| 1 | High dimensionality with few samples can lead to overfitting if feature selection is not applied. | ➕ Missed | **TRUE** | Classic p >> n problem: 500 features, 120 samples. The model has far more degrees of freedom than training examples and will memorize the training set. This is the *primary motivation* for feature selection here. |
| 2 | Wrapper methods may overfit in small datasets because they evaluate performance on the training set. | ✅ Selected | **TRUE** | With small n, CV estimates inside the wrapper are noisy. The selected feature subset can be overfit to the specific training sample. |
| 3 | Filter methods are preferred when feature interactions are important to capture. | ⬜ Skipped | **FALSE** | Correctly skipped. Filter methods evaluate features *individually* — they completely miss synergistic interactions between features. Wrappers/embeddings are needed for interactions. |
| 4 | Decision tree-based models can serve as embedded methods for feature selection by naturally ranking features based on split criteria. | ✅ Selected | **TRUE** | Random Forests produce feature importances from Gini/entropy split contributions — embedded feature selection with no separate step needed. |
| 5 | Including all features ensures the model captures all possible patterns, especially in small datasets. | ⬜ Skipped | **FALSE** | Correctly skipped. With 500 features and 120 samples, using all features virtually guarantees overfitting. |
| 6 | Cross-validation should be nested when using wrapper methods to avoid selection bias. | ✅ Selected | **TRUE** | If you use the same CV folds for feature selection AND evaluation, information leaks — selected features have already "seen" the test fold. Nested CV has a separate outer evaluation loop. |

**✅ Should have selected: 1, 2, 4, 6**
**❌ Mistake: Missed Statement 1** — the most fundamental motivation for feature selection in this scenario. This cost you 0.5 pts.

---

### Question 8 — 1.2/2 pts ⚠️ Partial

**Scenario:**
> You are selecting features for a classification model and evaluating both their statistical relevance to the target and their usefulness for improving model performance.
> **Which of the following statements are true?**

**Your selections: A ✅, B ✅, C ✅, D ➕, E ⬜, F ➕**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| A | A feature can be statistically relevant to the target variable but still be useless in the model if it is redundant with other features. | ✅ Selected | **TRUE** | If feature A correlates with the target AND with feature B (already in model), A adds zero marginal information. Relevance ≠ usefulness when redundancy exists. |
| B | Filtering methods rank features based on relevance, not usefulness, which may ignore interaction effects. | ✅ Selected | **TRUE** | Univariate filter metrics (Pearson r, mutual info) score each feature in isolation — they cannot detect that features X and Y are jointly predictive even when individually weak. |
| C | A feature with low correlation to the target might still be useful if it improves model performance when combined with other features. | ✅ Selected | **TRUE** | XOR-type interactions: two features individually uncorrelated with target but together perfectly predictive. Non-linear models can exploit this. |
| D | Wrapping methods can capture both relevance and usefulness by directly evaluating subsets based on model accuracy. | ➕ Missed | **TRUE** | Wrappers evaluate *subsets* — they inherently capture joint effects, redundancy, and interactions that filter methods miss entirely. |
| E | Relevance guarantees usefulness, so highly relevant features should always be included in the final model. | ⬜ Skipped | **FALSE** | Correctly skipped. A feature can be strongly correlated with the target yet completely redundant if another feature already captures that information. |
| F | A feature may appear irrelevant individually but become useful when combined with others in a non-linear model. | ➕ Missed | **TRUE** | Same interaction logic as C but from the model's perspective. Tree-based and kernel models can extract value from features with no univariate signal whatsoever. |

**✅ Should have selected: A, B, C, D, F**
**❌ Mistake: Missed D and F** — both describe how wrappers and non-linear models catch what filters miss. Cost you 0.8 pts.

---

### Questions 9–12 — Feature Selection Numerical

**Scenario:**
> You are working as a data scientist for an agricultural cooperative in the Netherlands. Your team is developing a machine learning model to predict whether a new crop treatment (fertilizer + irrigation schedule) will lead to a successful yield (1 = success, 0 = failure).
>
> Each field trial contains the following six features:
> - **SoilMoisture:** Average moisture content over the season (%)
> - **NitrogenLevel:** Initial nitrogen concentration in soil (mg/kg)
> - **IrrigationFreq:** Number of irrigation events during growing season
> - **AvgTemp:** Average temperature during the season (°C)
> - **WeedDensity:** Pre-treatment weed density (plants per m²)
> - **CowsPerAcre:** Number of cows per acre in adjacent pasture
>
> **Filter-Based Method (Pearson Correlation with Yield Outcome):**
>
> | Feature | Correlation with Yield Success |
> |---|---|
> | SoilMoisture | 0.79 |
> | NitrogenLevel | 0.64 |
> | IrrigationFreq | 0.58 |
> | AvgTemp | 0.15 |
> | WeedDensity | 0.05 |
> | CowsPerAcre | 0.31 |
>
> You select the top 3 features using this filter method.
>
> **Wrapper-Based Method (Forward Selection with Logistic Regression):**
> After evaluating different combinations of features using 5-fold cross-validation:
>
> | Feature Subset | CV Accuracy |
> |---|---|
> | SoilMoisture, NitrogenLevel, IrrigationFreq | 0.87 |
> | SoilMoisture, IrrigationFreq, CowsPerAcre | 0.85 |
> | SoilMoisture, NitrogenLevel | 0.82 |
> | SoilMoisture, IrrigationFreq | 0.81 |
> | NitrogenLevel, IrrigationFreq | 0.75 |
> | SoilMoisture, CowsPerAcre | 0.78 |

---

### Question 9 — 1/1 pts ✅

**Question:** Based on the filtering approach, which 3 features are selected?

**Calculation:** Sort by |correlation| descending:
```
SoilMoisture  = 0.79  ← 1st
NitrogenLevel = 0.64  ← 2nd
IrrigationFreq = 0.58 ← 3rd
CowsPerAcre   = 0.31
AvgTemp       = 0.15
WeedDensity   = 0.05
```

**✅ Your answer: SoilMoisture, NitrogenLevel, IrrigationFreq** — correct.

---

### Question 10 — 0/1 pts ❌

**Question:** Suppose you mistakenly use CowsPerAcre instead of NitrogenLevel. According to the wrapper results, how much does the cross-validation accuracy change?

**Calculation:**
```
Correct 3-feature set:   {SoilMoisture, NitrogenLevel, IrrigationFreq} → 0.87
Mistaken 3-feature set:  {SoilMoisture, IrrigationFreq, CowsPerAcre}   → 0.85

Change = 0.87 − 0.85 = 0.02
```

**✅ Correct Answer: 0.02**
**Your Answer: 0.33** ❌ — You likely compared the wrong pair of rows. Always map the exact feature subsets to the table rows, not just pick any two rows.

---

### Question 11 — 1/1 pts ✅

**Question:** What is the marginal gain in accuracy from adding IrrigationFreq to the pair {SoilMoisture, NitrogenLevel}?

**Calculation:**
```
{SoilMoisture, NitrogenLevel}               → 0.82
{SoilMoisture, NitrogenLevel, IrrigationFreq} → 0.87

Marginal gain = 0.87 − 0.82 = 0.05
```

**✅ Answer: 0.05** — correct.

---

### Question 12 — 0/1 pts ❌

**Question:** Compute the marginal gain in accuracy from adding CowsPerAcre to the subset {SoilMoisture, IrrigationFreq}, based on wrapper results.

**Calculation:**
```
{SoilMoisture, IrrigationFreq}              → 0.81
{SoilMoisture, IrrigationFreq, CowsPerAcre} → 0.85

Marginal gain = 0.85 − 0.81 = 0.04
```

**✅ Correct Answer: 0.04**
**Your Answer: 0.03** ❌ — Off by 0.01. Simple subtraction error: 0.85 − 0.81 = **0.04**, not 0.03. Double-check arithmetic on these.

---

## Question Set 3: Feature Transformation

---

### Question 13 — 2/2 pts ✅

**Scenario:**
> You are building a regression model to predict house prices based on features such as size, age, number of rooms, and location. You are considering two feature transformation techniques:
> - Principal Component Analysis (PCA) to reduce dimensionality.
> - Polynomial feature expansion to capture nonlinear relationships.
>
> **Which of the following statements are true in this scenario?**

**Your selections: 1 ⬜, 2 ✅, 3 ⬜, 4 ⬜, 5 ✅, 6 ⬜**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| 1 | PCA always improves model accuracy because it reduces noise in the input data. | ⬜ Skipped | **FALSE** | Correctly skipped. PCA can discard signal if it's in low-variance directions. "Always" is false. |
| 2 | Polynomial features can help the model capture nonlinear patterns such as diminishing returns on house size. | ✅ Selected | **TRUE** | Adding x², x·z lets a linear model fit curves. Diminishing returns = concave relationship = captured by a quadratic term. |
| 3 | PCA expands the feature space by introducing higher-order combinations of existing features. | ⬜ Skipped | **FALSE** | Correctly skipped. PCA *reduces* dimensionality. Polynomial expansion *expands* it. These are exact opposites. |
| 4 | Polynomial feature expansion reduces the chance of overfitting by simplifying the feature space. | ⬜ Skipped | **FALSE** | Correctly skipped. Polynomial expansion *increases* feature count and overfitting risk. |
| 5 | PCA may improve performance if the original features are highly correlated, by removing redundancy. | ✅ Selected | **TRUE** | PCA decorrelates features. Multicollinear inputs (e.g., house size + number of rooms) → PCA removes redundancy and stabilizes regression coefficients. |
| 6 | Both PCA and polynomial expansion preserve the interpretability of the original feature space. | ⬜ Skipped | **FALSE** | Correctly skipped. PCA components are abstract linear combinations of all features — uninterpretable. High-degree polynomial terms are equally opaque. |

**✅ Perfect — 2/2.**

---

### Question 14 — 0.67/2 pts ⚠️ Partial

**Scenario:**
> You are working with gene expression data containing 10,000 genes (features) measured across only 80 patient samples. You plan to use PCA or ICA as a preprocessing step before clustering and visualization.
> **Which of the following statements correctly describe the behavior or risks of PCA and ICA in this setting?**

**Your selections: 1 ⬜, 2 ✅, 3 ⬜, 4 ✅, 5 ❌, 6 ➕**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| 1 | ICA is always preferred over PCA in high-dimensional biological datasets, since it produces independent components. | ⬜ Skipped | **FALSE** | Correctly skipped. ICA requires n >> p and non-Gaussian sources. With 80 samples and 10,000 genes, ICA is highly unstable — PCA is far safer. |
| 2 | PCA may produce components aligned with noise if too many are retained in a high-dimensional setting. | ✅ Selected | **TRUE** | In p >> n settings, most eigenvalues of the sample covariance matrix correspond to noise. Retaining too many PCs includes these noise directions. |
| 3 | PCA guarantees recovery of biologically interpretable factors if the top components explain most variance. | ⬜ Skipped | **FALSE** | Correctly skipped. PCA finds axes of maximum *variance*, not biological meaning. High variance ≠ biological interpretability. |
| 4 | Both PCA and ICA may become unstable when features are highly collinear, as is common in gene expression data. | ✅ Selected | **TRUE** | Collinearity → near-singular covariance matrix → unstable PCA eigendecomposition; ICA mixing matrix inversion becomes ill-conditioned. |
| 5 | Overfitting is not a concern in PCA since there are no model parameters to train. | ❌ Selected | **FALSE** | **This is the wrong one.** In p >> n, sample covariance eigenvectors are unreliable estimates of the true population eigenvectors — this *is* a form of statistical overfitting, even without explicit learnable parameters. |
| 6 | ICA may identify spurious components if the true latent sources are not sufficiently non-Gaussian. | ➕ Missed | **TRUE** | ICA's identifiability fundamentally relies on source non-Gaussianity. If sources are Gaussian (or nearly so), ICA is unidentifiable and returns meaningless noise components. |

**✅ Should have selected: 2, 4, 6**
**❌ Mistake: Selected 5 (PCA can overfit in p>>n — it absolutely can) instead of 6 (ICA fails for near-Gaussian sources).**

---

### Questions 15–17 — PCA/ICA Numerical (Eel Hatchery)

**Scenario:**
> You manage a commercial eel hatchery with monitoring systems deployed across multiple tanks. For each tank, sensors continuously measure 7 environmental features: Water Temperature, Dissolved Oxygen, pH Level, Ammonia Concentration, Salinity, Light Exposure, Turbidity.
>
> You collect 1,000 time points per tank, standardized into dataset X ∈ ℝ^(1000×7). The hatchery system is governed by 5 latent processes, but only 3 dimensions strongly affect variability in observed data. You apply both PCA and ICA before clustering the tanks into 6 categories based on environmental dynamics.
>
> **PCA Results:**
> Eigenvalues of sample covariance matrix:
> **λ = [3.2, 2.1, 1.0, 0.4, 0.2, 0.05, 0.05]**
> You decide to retain the top 3 principal components for clustering.
>
> **ICA Results:**
> ICA returns 7 statistically independent components.
> Excess kurtosis values:
> **kurtosis = [2.7, 1.8, 1.1, 0.2, 0.1, −0.1, 0.8]**
> You retain the 3 most non-Gaussian components for clustering.

---

### Question 15 — 1/1 pts ✅

**Question:** What percentage of the total variance is retained by the first 3 principal components? Round to the nearest whole number.

**Calculation:**
```
Total variance = 3.2 + 2.1 + 1.0 + 0.4 + 0.2 + 0.05 + 0.05 = 7.0

Top 3 variance = 3.2 + 2.1 + 1.0 = 6.3

% retained = (6.3 / 7.0) × 100 = 90%
```

**✅ Answer: 90%** — correct.

---

### Question 16 — 0/1 pts ❌

**Question:** Compute the sum of the highest excess kurtosis values (for the 3 retained ICA components).

**Kurtosis values:** [2.7, 1.8, 1.1, 0.2, 0.1, −0.1, 0.8]

Sorted descending:
```
2.7, 1.8, 1.1, 0.8, 0.2, 0.1, −0.1
```

Top 3 highest = **2.7, 1.8, 1.1**

```
Sum = 2.7 + 1.8 + 1.1 = 5.6
```

**✅ Correct Answer: 5.6**
**Your Answer: 5.7** ❌ — Very likely misread 1.1 as 1.2, or accidentally included 0.8 as the 3rd value instead of 1.1. Re-read the list carefully: **1.1 is the 3rd largest, not 0.8**.

---

### Question 17 — 2/2 pts ✅

**Scenario:**
> You observe that PCA explains a majority of the variance with the first several components, while ICA shows the highest excess kurtosis in its first several components. However, ICA also identifies 2 additional components with mild excess kurtosis (0.3 and 0.1). Suppose your downstream task is unsupervised clustering of eel behavioral states.
> **Which of the following best explains why choosing more than 3 components (e.g., 5) might degrade clustering performance? Choose all that apply.**

**Your selections: 1 ✅, 2 ✅, 3 ⬜, 4 ⬜**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| 1 | Including additional ICA components improves completeness but introduces directions with low non-Gaussianity that may correspond to noise. | ✅ Selected | **TRUE** | Components 4–5 have kurtosis near 0 (0.2, 0.1) — nearly Gaussian = noise. Adding them pollutes the clustering space with noise dimensions. |
| 2 | ICA components beyond the top 3 may still be independent but add no clustering signal, reducing silhouette score due to overfitting to noise. | ✅ Selected | **TRUE** | Independence ≠ informativeness. Noise dimensions dilute cluster separation — more dimensions with no signal → silhouette scores degrade. |
| 3 | PCA and ICA components beyond the third are typically correlated, violating assumptions of the clustering algorithm. | ⬜ Skipped | **FALSE** | Correctly skipped. ICA components are by definition statistically independent. This statement is factually wrong about ICA. |
| 4 | Orthogonal components in PCA are better suited for K-Means clustering than the non-orthogonal ICA ones, especially when using more than 3. | ⬜ Skipped | **FALSE** | Correctly skipped. ICA components are also orthogonalized via whitening. Orthogonality alone doesn't explain why >3 hurts clustering. |

**✅ Perfect — 2/2.**

---

## Question Set 4: Manifold Learning

---

### Question 18 — 0/2 pts ❌

**Scenario:**
> You are handed a 32×32 grayscale dataset of handwritten "A"s where variation is driven primarily by two latent factors: angle and size. You want a 2D embedding that reveals this curved 2D manifold without crowding or tearing. Consider the following statements about Sammon Mapping, Isomap, Laplacian Eigenmaps, t-SNE, and UMAP.
> **Which of the following statements are correct? Select all that apply.**

**Your selections: A ✅, B ❌, C ⬜, D ➕, E ⬜, F ❌**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| A | Isomap constructs a k-NN graph, approximates geodesic distances via shortest paths on that graph, and then applies classical MDS to those distances. | ✅ Selected | **TRUE** | Exactly correct — this is the Isomap algorithm verbatim: build k-NN graph → run Dijkstra/Floyd-Warshall for shortest paths → apply classical MDS on the geodesic distance matrix. |
| B | Sammon Mapping minimizes an *unweighted* squared error between high- and low-dimensional distances, intentionally emphasizing large pairwise distances to avoid crowding. | ❌ Selected | **FALSE** | Two errors: (1) Sammon's stress IS weighted — each squared error term is divided by the original high-dimensional distance d*_ij: **stress = Σ (d*_ij − d_ij)² / d*_ij**. (2) Dividing by d*_ij gives *larger* weight to *small* distances — it emphasizes *local/small* structure, the exact opposite of what the statement claims. |
| C | t-SNE preserves global pairwise distances and relative cluster areas, making it reliable for density interpretation and global geometry. | ⬜ Skipped | **FALSE** | Correctly skipped. t-SNE is explicitly designed for *local* structure only. Global distances and cluster sizes/spacing are meaningless in t-SNE output. |
| D | Laplacian Eigenmaps obtains an embedding from the smallest non-trivial eigenvectors of the generalized eigenproblem Lf = λDf, thereby preserving local neighborhoods. | ➕ Missed | **TRUE** | Exactly correct — L is the graph Laplacian, D is the degree matrix. The smallest non-trivial eigenvectors give the embedding. The local neighborhood graph structure is directly encoded in L. |
| E | UMAP requires random initialization in unsupervised mode and cannot inductively map new samples into an existing embedding. | ⬜ Skipped | **FALSE** | Correctly skipped. UMAP *can* perform out-of-sample extension using transform() on new points from a trained model. Random initialization is optional, not required. |
| F | Isomap is immune to short-circuit errors from spurious edges when k is chosen too large, because its geodesic distances are computed exactly from the manifold equation. | ❌ Selected | **FALSE** | The *opposite* is true. When k is too large, spurious long-range edges appear in the k-NN graph, creating shortcuts that corrupt geodesic distance estimates. Isomap is *notoriously vulnerable* to this. There is no "manifold equation" — distances are approximated from data, not computed analytically. |

**✅ Should have selected: A and D**
**❌ Mistakes:**
- **B**: Sammon Mapping uses a *weighted* stress that emphasizes *small* distances, not large ones.
- **F**: Isomap IS vulnerable to short-circuit errors — it is NOT immune.
- **D**: Missed — Laplacian Eigenmaps eigenproblem description is exactly correct.

---

### Question 19 — 0.5/2 pts ⚠️ Partial

**Scenario:**
> Your conservation analytics team is building a production embedding for Roseate Spoonbills (Platalea ajaja) using high-dimensional data: GPS telemetry, accelerometry, plumage indices, foraging-site salinity, prey proxies, colony metadata, and seasonal weather. You need reliable kNN retrieval, interpretable inter-group distances, stable re-runs, and a well-defined out-of-sample mapping for newly streamed tags.
> **Which statements correctly explain why 2D t-SNE/UMAP are poor choices for this production problem? Select all that apply.**

**Your selections: A ✅, B ✅, C ➕, D ➕, E ❌, F ⬜**

| # | Statement | Your Pick | Verdict | Explanation |
|---|---|---|---|---|
| A | Their 2D maps distort global geometry and densities, so quantitative kNN/retrieval assumptions about distances and neighborhood mass do not hold. | ✅ Selected | **TRUE** | t-SNE and UMAP warp inter-cluster distances to equalize visual cluster sizes. Nearest neighbors in 2D do not reliably correspond to true nearest neighbors in high-D — kNN retrieval fails. |
| B | They exhibit run-to-run stochasticity and fragile hyperparameter sensitivity, undermining stability required for monitoring and reporting. | ✅ Selected | **TRUE** | t-SNE and UMAP with random init give different layouts each run. A production monitoring system requires consistent coordinates across time — stochastic embeddings make this impossible. |
| C | Out-of-sample mapping is not reliably parametric or metric-preserving, making it risky to place new streaming tags consistently into the same space. | ➕ Missed | **TRUE** | t-SNE has *no* out-of-sample extension whatsoever. UMAP has an approximate one, but it's not metric-preserving. For new streaming tags that must be placed consistently into the existing embedding, this is a critical failure mode. |
| D | Axes lack variance-explained semantics; cluster area/spacing is easily misread as prevalence/similarity, encouraging incorrect ecological conclusions. | ➕ Missed | **TRUE** | Unlike PCA (where PC1 = direction of max variance), t-SNE/UMAP axes are meaningless. Cluster sizes in t-SNE are artificially equalized regardless of true population sizes — misread as prevalence, this leads to wrong ecological conclusions about the spoonbills. |
| E | They provide a guaranteed metric-preserving encoder by default, so distances in 2D can be safely used for population-level inference across seasons. | ❌ Selected | **FALSE** | This is the *opposite* of what t-SNE/UMAP do. They explicitly sacrifice metric preservation for visual cluster separation. Selecting this as a "reason they're a poor choice" is internally contradictory — the statement describes a positive property they don't even have. |
| F | They automatically correct batch effects (tag hardware generations, survey years), ensuring cross-cohort comparability without extra preprocessing. | ⬜ Skipped | **FALSE** | Correctly skipped. Neither method corrects batch effects. That requires dedicated methods like Harmony, ComBat, or scVI. |

**✅ Should have selected: A, B, C, D**
**❌ Mistakes:**
- **E**: Completely backwards — t-SNE/UMAP are NOT metric-preserving. This is false and was selected.
- **C**: Missed — no out-of-sample mapping for t-SNE, and UMAP's is unreliable. Critical for streaming tags.
- **D**: Missed — axes are meaningless and cluster areas are artificially equalized, dangerous for ecological inference.

> **Note on scoring (0.5/2):** With negative marking, selecting the wrong E likely cancelled one of your correct answers (A or B). Correct selections of A, B, C, D should give full 2/2.

---

## Full Summary Table

| Q | Topic | Score | Max | Your Selections | Correct Answer | Key Mistake |
|---|---|---|---|---|---|---|
| 1 | Clustering — high-dim properties | 0 | 2 | 1, 2, 3, 5 | **1, 3, 6** | Selected 2 (chaining = weakness) & 5 (PCA "always"). Missed 6 (DBSCAN fails). |
| 2 | K-Means/GMM market segmentation | 0 | 2 | A, B, D, F | **A, F** | Selected B (GMMs still need scaling) & D (training likelihood alone ≠ useful k). |
| 3 | K-Means assignment x_new=7 | 1 | 1 | Cluster 2 | **Cluster 2** | — |
| 4 | EM responsibility Cluster 1 | 1 | 1 | 0.2366 | **0.2366** | — |
| 5 | EM responsibility Cluster 2 | 1 | 1 | 0.7597 | **0.7597** | — |
| 6 | EM responsibility Cluster 3 | 1 | 1 | 0.0037 | **0.0037** | — |
| 7 | Feature selection — 500 features | 1.5 | 2 | 2, 4, 6 | **1, 2, 4, 6** | Missed statement 1 (p>>n → overfitting). |
| 8 | Relevance vs. usefulness | 1.2 | 2 | A, B, C | **A, B, C, D, F** | Missed D (wrappers evaluate subsets) & F (non-linear interactions). |
| 9 | Filter method — top 3 features | 1 | 1 | SoilMoisture, NitrogenLevel, IrrigationFreq | **Same** | — |
| 10 | Wrapper — accuracy change | 0 | 1 | 0.33 | **0.02** | Compared wrong table rows. 0.87−0.85=0.02. |
| 11 | Marginal gain IrrigationFreq | 1 | 1 | 0.05 | **0.05** | — |
| 12 | Marginal gain CowsPerAcre | 0 | 1 | 0.03 | **0.04** | Subtraction error. 0.85−0.81=0.04. |
| 13 | PCA vs Polynomial features | 2 | 2 | 2, 5 | **2, 5** | — |
| 14 | PCA/ICA — gene expression | 0.67 | 2 | 2, 4, 5 | **2, 4, 6** | Selected 5 (PCA can overfit in p>>n — FALSE). Missed 6 (ICA fails for Gaussian sources). |
| 15 | % variance retained by top 3 PCs | 1 | 1 | 90 | **90%** | — |
| 16 | Sum of top 3 kurtosis values | 0 | 1 | 5.7 | **5.6** | Misread 1.1 as 1.2. Top 3 are 2.7+1.8+1.1=5.6. |
| 17 | Why >3 components hurts clustering | 2 | 2 | 1, 2 | **1, 2** | — |
| 18 | Manifold learning properties | 0 | 2 | A, B, F | **A, D** | B wrong (Sammon emphasizes small not large distances). F wrong (Isomap IS vulnerable). Missed D (Laplacian Eigenmaps). |
| 19 | Why t-SNE/UMAP poor for production | 0.5 | 2 | A, B, E | **A, B, C, D** | E is false and backwards. Missed C (no reliable out-of-sample) and D (axes are meaningless). |
| **Total** | | **14.87** | **28** | | **~27–28** | |

---

## 10 Concepts That Tripped You Up

1. **Single linkage chaining** — the chaining property makes single linkage *fragile* in noisy settings, not robust. It's a known weakness, not a feature.

2. **PCA "always" helps clustering** — PCA maximizes variance, not cluster separability. If discriminative signal is in low-variance directions, PCA destroys it. Never trust "always."

3. **GMMs and feature scaling** — probabilistic ≠ scale-invariant. GMMs estimate covariance matrices which are just as distorted by unscaled features as K-Means distances.

4. **Sammon Mapping stress formula** — **stress = Σ (d*−d)² / d*** — dividing by d* gives *larger* weight to *small* distances. It emphasizes local structure, not global. The statement said "unweighted" and "emphasizes large distances" — both wrong.

5. **Isomap short-circuit errors** — when k is too large, long-range edges create geodesic shortcuts. Isomap is *vulnerable* to this, not immune. There's no analytic manifold equation.

6. **Laplacian Eigenmaps** — solves generalized eigenproblem **Lf = λDf** (L = graph Laplacian, D = degree matrix). The smallest non-trivial eigenvectors give the embedding.

7. **PCA overfitting in p >> n** — in high-dimensional settings with few samples, sample covariance eigenvectors are unreliable. This IS a form of statistical overfitting even without explicit parameters.

8. **ICA requires non-Gaussianity** — ICA's identifiability theorem (Comon, 1994) requires sources to be non-Gaussian. Gaussian sources → ICA is unidentifiable → spurious components.

9. **t-SNE/UMAP are NOT metric-preserving** — axes are meaningless, cluster sizes are artificially equalized, inter-cluster distances are distorted. Never use for quantitative distance/density inference.

10. **Wrapper marginal gain** — always: **Gain = Accuracy(S ∪ {f}) − Accuracy(S)**. Map feature names exactly to table rows — don't eyeball similar-looking rows.