# Project FAQ — CS7641 UL Spring 2026

Captures explanations for recurring methodological questions raised during implementation.
**Rule:** Any time a "why does X work this way?" question is asked and answered, it goes here.

---

## Phase 2 — Raw Clustering

### Q: The phase 2 script finished very quickly. Is the speed correct?

**A: Yes — expected and correct.**

Wine is 3,897 rows × 12 features (after 60/20/20 split). Adult is 27,132 rows × 104 OHE features.
The sweep runs 38 total fits: 19 k values × 2 datasets × 2 algorithms (KMeans + GMM).
Both use sklearn's C-optimized implementations (KMeans uses BLAS, GMM uses scipy linalg).
Wine finishes in ~6 seconds; Adult in ~2.7 minutes — dominated by Adult GMM on 104 features.

---

### Q: Does KMeans use the right number of iterations? Is there any hyperparameter optimization?

**A: Yes, defaults are correct. No HPO — intentional.**

- **KMeans:** `max_iter=300` (sklearn default), `n_init="auto"` (resolves to 10 random re-inits using k-means++ initialization). Each restart selects the best result by inertia. This is more than sufficient for datasets of this size.
- **GMM:** `max_iter=100` (sklearn default), `reg_covar=1e-3` (added explicitly to prevent singular covariance matrices in high-dimensional Adult OHE space).
- **No HPO beyond the k/n_components sweep.** The assignment requires selecting K *without* labels — pure sweep over metrics (silhouette, BIC, AIC, CH, DB) is the correct methodology. Grid-searching max_iter or other internal params is out of scope and would not change the label-free selection.

---

### Q: How do you select K from the Phase 2 sweep? What are the actual chosen values?

**A: Label-free selection using metric convergence. Chosen values below.**

Selection logic (no labels used at any point):
- **KMeans:** prefer k where silhouette peaks, CH peaks, DB troughs, and the inertia elbow bends. When metrics agree, take the lowest k (parsimony).
- **GMM:** prefer n where BIC reaches its minimum (or elbow before overfitting). AIC is excluded as a primary criterion because it monotonically decreases — it favors more components by design and is only shown for reference.

| Dataset | Algorithm | Chosen K | Primary signal |
|---------|-----------|----------|----------------|
| Wine    | KMeans    | **2**    | All four metrics agree: silhouette=0.340 (peak), CH=1497 (peak), DB=1.335 (trough), inertia drops sharply 2→3 then flattens |
| Wine    | GMM       | **7**    | BIC local minimum at n=7 (60841); silhouette still positive (0.040) |
| Adult   | KMeans    | **8**    | Best silhouette (0.114), CH local peak (2583), DB improving; elbow is gradual due to high dimensionality |
| Adult   | GMM       | **7**    | BIC elbow around n=6-7 (-8290938); silhouette reasonable (0.058) |

These values are **frozen** for use in Phase 4 (clustering in reduced spaces) and Phase 6 (cluster-derived NN features).

---

### Q: Why does Wine KMeans converge to K=2 when there are 8 wine quality classes?

**A: Clustering captures structure in feature space, not label space.**

Wine has 8 quality classes (3–9), but the physicochemical features (alcohol, acidity, sulfates, etc.) cluster naturally around wine type (`type` = red/white, encoded 0/1). Red and white wines have structurally different profiles, so K=2 reflects the dominant geometric partition. Quality labels are a finer-grained subjective rating that is not fully recoverable from raw features — this mismatch is expected and is part of what Phase 2 is meant to reveal. We use labels only post-hoc (Phase 4 analysis, Phase 5 NN training).

---

### Q: Why does Adult have low silhouette scores (0.09–0.11) across all k values?

**A: High-dimensional OHE spaces are notoriously hard to cluster with Euclidean metrics.**

Adult after one-hot encoding has 104 binary/continuous features. Euclidean distance in high-dimensional binary spaces suffers from the curse of dimensionality — distances between points become nearly uniform, which flattens the silhouette score. This is expected and is exactly why Phase 3 (dimensionality reduction) exists: PCA/ICA/RP compress Adult to a lower-dimensional space before re-clustering in Phase 4.

---

### Q: What do the clustering evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, BIC, AIC) mean?

**A: They are internal validation metrics used to evaluate the quality of clustering without using ground truth labels.**

**K-Means Metrics:**
*   **Silhouette Score:** Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Range is -1 to +1. **Higher is better.** A value near +1 indicates points are far away from neighboring clusters.
*   **Calinski-Harabasz Index (Variance Ratio Criterion):** The ratio of the sum of between-cluster variance to within-cluster variance. **Higher is better.** It favors dense and well-separated convex (spherical) clusters.
*   **Davies-Bouldin Index:** The average similarity measure of each cluster with its most similar cluster. **Lower is better.** A lower score means clusters are far apart and less dispersed.

**GMM Metrics:**
*   **BIC (Bayesian Information Criterion):** Estimates model likelihood while heavily penalizing complexity (number of components). **Lower is better.** It is generally preferred for choosing the final $n\_components$ since its strict penalty prevents overfitting.
*   **AIC (Akaike Information Criterion):** Similar to BIC, it balances goodness of fit with complexity, but its penalty for adding more components is less harsh. **Lower is better**, but it can sometimes suggest a higher number of components than BIC.

---

### Q: Do EM and GMM mean the same thing?

**A: In the context of this assignment and Scikit-Learn, yes, they are used interchangeably.**

Technically, they refer to two different things:
*   **GMM (Gaussian Mixture Model)** is the **model**. It assumes that the dataset is made up of a mixture of several different Gaussian (normal) distributions.
*   **EM (Expectation-Maximization)** is the **algorithm** used to train that model. It is the mathematical process used to find the best parameters (means, variances, and weights) for those Gaussian distributions.

When we use `sklearn.mixture.GaussianMixture` in our code, Scikit-Learn is automatically using the Expectation-Maximization (EM) algorithm under the hood to fit the model. By implementing GMM, we have successfully fulfilled the assignment's "EM/GMM" requirement.

---

## Phase 3 — Dimensionality Reduction

### Q: What do the dimensionality reduction diagnostic plots (PCA Variance, ICA Kurtosis, RP Stability) mean?

**A: They help determine the optimal number of dimensions (components) to retain for each dataset.**

**PCA (Principal Component Analysis):**
*   **Explained Variance:** Measures how much of the dataset's total variance (information) is captured by each principal component. The plot shows variance dropping off as the component number increases. **Higher is better.**
*   **Cumulative Variance:** The running total of explained variance. **Higher is better.** You typically look for an "elbow" in the curve or a point where a target threshold (e.g., 80% to 95% of total variance) is reached, allowing you to discard less informative dimensions while preserving the core structure of the data.

**ICA (Independent Component Analysis):**
*   **Kurtosis:** A statistical measure of the "tailedness" of a distribution. ICA assumes the underlying source signals are non-Gaussian. It searches for components that maximize this non-Gaussianity. **Higher absolute value is better.** High absolute kurtosis (either very spiky or very flat distributions) indicates a component that captures a strong independent signal. A common selection strategy is to sort components by kurtosis and retain those with the highest absolute values, looking for a sharp drop-off.

**RP (Random Projection):**
*   **Reconstruction Error & Stability:** Random Projection is based on the Johnson-Lindenstrauss lemma, which states that high-dimensional points can be randomly projected into a lower-dimensional space while mostly preserving the distances between them. The stability plot typically shows the reconstruction error (or distance distortion) across multiple random seeds as the number of components increases. **Lower error and lower variance across seeds is better.** You want to choose a dimensionality where the error stabilizes and the variation between different random seeds is acceptably low, forming an "elbow" of stability.

---

### Q: What is the main takeaway from the ICA results (Kurtosis)?

**A: ICA successfully acts like a "metal detector" to unmix hidden, independent, non-Gaussian source signals from the raw data.**

Unlike PCA, which compresses data to preserve variance, ICA attempts to unmix overlapping signals (like trying to separate two voices recorded on the same microphone). It does this by measuring **Kurtosis**. According to the Central Limit Theorem, mixed signals look like a normal bell curve (Gaussian, kurtosis ≈ 0). Therefore, to find the independent source signals, ICA looks for components that are highly *non-Gaussian* (very "spiky" or very flat distributions), resulting in a high absolute kurtosis.

*   **Wine Takeaway:** Out of the 8 components tested, Component 6 had an absolutely massive kurtosis spike (~77), with a few others showing strong signals. The rest were close to 0 (Gaussian noise). This suggests there are a handful of truly distinct, independent physicochemical drivers hidden within the 11 raw features.
*   **Adult Takeaway:** Out of the 22 components tested, one component had a staggering kurtosis of ~143. In the high-dimensional, one-hot encoded space, this indicates ICA found a highly specific, isolated feature (likely a demographic outlier or an extremely imbalanced categorical variable) that acts as a strong, independent signal.

---

### Q: Why must the RP stability figure include n_components in the title?

**A: Because RP's target dimensionality is the critical context — without it the figure is uninterpretable in the report.**

The RP stability plot shows reconstruction error on the Y-axis and random seeds on the X-axis. A reader looking at this graph cannot infer *what dimensionality* was being tested. In our pipeline, RP uses the exact same `n_components` as PCA (Wine=8, Adult=22), which represents a specific, deliberate compression level chosen to preserve 90% of variance. Without this in the title, the graph only shows "stability across seeds" with no indication of how aggressively the data was compressed. The fix: read `n_components` from the first row of the DataFrame and embed it in the title: `RP Reconstruction Error Across Seeds (n_components=8)`.

---

### Q: Why must the ICA kurtosis figure show the selection threshold and n_selected?

**A: Without both, the reader cannot verify or reproduce the component selection decision.**

The ICA kurtosis bar chart sorted by |kurtosis| shows *which components are most non-Gaussian*, but a reader has no way to know where we drew the line. Our selection rule is: retain all components with |kurtosis| ≥ median(|kurtosis|), with a floor of 2. Without a visible threshold line, the cutoff is invisible. Without `n_selected` in the title, the reader must count bars manually. The fix: compute `threshold = median(|kurtosis|)` from the DataFrame, draw a dashed red axhline labelled with its value and the resulting count, and embed `n_selected=N` in the title.

---

### Q: What does the assignment mean by "explain any scaling or whitening decisions"?

**A: It is asking you to justify how you preprocessed the data's variance before and during dimensionality reduction.**

*   **Scaling Decision (Our implementation):** PCA is highly sensitive to the initial scale of the data. If one feature ranges from 0-1000 and another from 0-1, PCA will incorrectly assume the first feature is the most important just because its raw variance is larger. We handled this in **Phase 1** by applying `StandardScaler` to the continuous variables in both datasets. Our decision is to ensure all features have zero mean and unit variance so PCA treats them equally.
*   **Whitening Decision (Our implementation):** Whitening is an optional transformation that removes correlation between features and forces them all to have a variance of 1 *after* the projection.
    *   **For PCA:** We explicitly decided **not** to use whitening (`whiten=False` is the scikit-learn default, which we kept). We want the principal components to reflect the true magnitude of the variance they explain.
    *   **For ICA:** Whitening is a strict mathematical prerequisite for the algorithm to find independent signals. We relied on Scikit-Learn's `FastICA` default behavior, which automatically whitens the data (`whiten='arbitrary-variance'`) under the hood before extracting the independent components.

---

## Phase 4 — Clustering in Reduced Spaces

### Q: What do the Phase 4 bar charts and heatmaps represent?

**A: They visualize whether dimensionality reduction made the clusters better or worse compared to clustering on raw data.**

In Phase 4, we take the three reduced datasets (PCA, ICA, RP) and run K-Means and GMM on them using the exact same number of clusters we found in Phase 2. We then compare the resulting internal clustering metrics.

**Interpreting the Bar Charts (e.g., `adult_phase4_bar.png`):**
*   **Layout:** 2×3 subplots — top row is KMeans (Silhouette, CH, DB), bottom row is GMM (Silhouette, BIC, AIC). Each subplot has its own Y-axis in raw metric units.
*   **X-axis:** The different dataset versions (Raw, PCA, ICA, RP).
*   **Dashed baseline:** A horizontal dashed line marks the Raw bar height in each subplot, making it immediately visible whether a DR method beat or missed the baseline.
*   **Takeaway:** You are looking for bars that cross *above* the dashed baseline. If the Silhouette subplot for PCA sits above the line, it proves that compressing the data helped K-Means find denser, better-separated clusters.

**Interpreting the Heatmap (`phase4_clustering_heatmap.png`):**
*   This is a consolidated view showing the exact metric values for every combination of Dataset × DR Method × Clustering Algorithm.
*   It allows you to quickly scan for the highest (or lowest) values across all experiments at once.

### Q: What is the main takeaway from the Phase 4 results?

**A: Dimensionality Reduction generally *improves* Euclidean-based clustering (K-Means) on high-dimensional sparse data, but can hurt probabilistic clustering (GMM) if critical variance is discarded.**

*   **For Adult (The High-Dimensional Case):** The raw Adult dataset had 104 OHE features with KMeans silhouette=0.114. After PCA (22d) silhouette rose to 0.140, RP (22d) to 0.139, ICA (11d) to 0.138 — a consistent ~20% improvement across all DR methods. *Takeaway: DR stripped the sparse binary noise, letting KMeans measure meaningful Euclidean distances for the first time.*
*   **For Wine (The Low-Dimensional Case):** Wine KMeans improved under PCA (sil=0.357 vs raw 0.340) but degraded under ICA (0.194) and RP (0.260). GMM was mixed — ICA actually helped GMM (sil=0.225 vs raw 0.040) while PCA and RP hurt it. *Takeaway: When the raw feature space is already dense and low-dimensional, DR results are algorithm-dependent — PCA preserved the two-cluster geometry, but ICA's aggressive 4-component selection destroyed the KMeans structure while exposing GMM-friendly independent signals.*

---

### Q: Why do the Bar Charts use different metrics for K-Means and GMM, but the Heatmap uses the same metrics for both?

**A: The Bar Charts show internal algorithmic validity, while the Heatmap shows cross-algorithm geometric comparison.**

*   **Bar Charts (Native Evaluation):** K-Means builds clusters based on Euclidean distance, so it is evaluated using distance metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin). GMM is a probabilistic model that fits Gaussian distributions, so it is evaluated using information-theoretic likelihood metrics (BIC, AIC). The bar charts correctly use each algorithm's native metrics to measure internal validity.
*   **Heatmap (Cross-Comparison):** Step 3 of the assignment requires a "clear comparison" across all combinations. You cannot compare an Inertia score to a BIC score directly. To solve this, the heatmap forces GMM to output "hard" cluster assignments (assigning each point to its most likely cluster). Once GMM has hard labels, we can calculate Euclidean distance metrics (Silhouette, CH, DB) on the GMM clusters. This allows a direct, 1-to-1 comparison of how well dimensionality reduction improved geometric cluster cohesion across *both* algorithmic philosophies.

---

## Phase 5 — Neural Networks on Reduced Inputs

### Q: Why does the Phase 5 design say "only input_dim changes — all other config fixed"?

**A: It is the controlled experiment principle — you can only attribute a performance difference to the input representation if everything else is held constant.**

Phase 5 trains the same `Linear(input_dim, 100) → ReLU → Linear(100, 8)` network on four variants of the Wine input: raw (12d), PCA (8d), ICA (4d), RP (8d). The only thing that changes is `input_dim` in the first linear layer. The optimizer (Adam), learning rate (1e-3), betas, weight decay, batch size, number of epochs, loss function, and architecture width are all identical across variants.

If any of those were allowed to vary — e.g., ICA trained for 40 epochs while raw trained for 20 — a final F1 difference could not be attributed to the representation alone. It might just mean ICA got more training time. By locking every hyperparameter except input shape, any F1 difference in the boxplot is *solely caused by the input representation*, which is the scientific question Phase 5 is designed to answer.

---

### Q: What do the Phase 5 boxplots and learning curves represent?

**A: They measure whether compressing the input data helped or hurt the Neural Network's ability to predict Wine Quality.**

*   **The Boxplot (`phase5_f1_boxplot.png`):** Shows the distribution of the final Validation Macro-F1 scores across 10 random seeds. The dashed horizontal line is the median score of the Raw dataset. You use this to see if the PCA, ICA, or RP boxes sit *above* (improved performance) or *below* (degraded performance) the raw baseline.
*   **The Learning Curves:** Show the epoch-by-epoch training dynamics. The solid line is the mean across 10 seeds, and the shaded region is the standard deviation (showing stability). You use this to see if dimensionality reduction made the network train faster, slower, or less stably.

### Q: What is the main takeaway from the Phase 5 results?

**A: For a dense, low-dimensional dataset like Wine, linear dimensionality reduction acts purely as lossy compression, permanently destroying predictive signal and lowering the Neural Network's performance ceiling.**

Because the Wine dataset only has 11 continuous features, it does not suffer from the "curse of dimensionality" or extreme sparsity. 
1.  **Performance Drop:** As seen in the boxplot, RAW > PCA > RP > ICA. Every single reduction method lowered the median Macro-F1 score. 
2.  **Information Loss vs. Noise Removal:** If the data had 1,000 noisy features, PCA might improve the neural network by acting as a denoiser. But here, reducing 11 features to 8 (PCA) simply threw away ~6% of the variance. For a Neural Network trying to map complex non-linear boundaries to 8 different Wine Quality classes, that discarded variance contained critical predictive information.
3.  **Training Dynamics:** The learning curves prove that the network *could* still learn on the reduced data (the curves are smooth and stable), but it mathematically asymptotes at a worse loss because it lacks the necessary input dimensions to separate the overlapping classes.

---

### Q: Is it a better idea to switch to the Adult dataset because dimensionality reduction negatively affected Wine?

**A: Definitively NO. You should absolutely stick with Wine. A "negative" result is often a fantastic result, provided you can explain why it happened mathematically.**

If you switched to the Adult dataset, you would likely see the neural network perform better on the PCA data than the Raw data. But that is a predictable, somewhat "boring" story: *"Adult had 104 sparse features, PCA removed the noise, and the neural network did better."*

With **Wine**, you have a much more sophisticated, mathematically interesting case to write about:
1.  **The Information Bottleneck:** The raw Wine dataset is incredibly dense. It has 11 features that are all highly relevant to the chemical makeup of wine. It does not suffer from sparsity or the curse of dimensionality. 
2.  **The "Lossy" Compression:** When you forced Wine through PCA down to 8 components, PCA threw away ~6% of the variance because it deemed it statistically "unimportant." But for a Neural Network trying to untangle heavily overlapping, non-linear class boundaries (predicting 8 different quality scores), that 6% variance was actually the subtle signal it needed to differentiate a "Quality 5" wine from a "Quality 6" wine.
3.  **The Scientific Conclusion:** This allows you to conclude your report with a powerful ML lesson: *Linear Dimensionality Reduction is not a magic bullet.* While it acts as a powerful denoiser for sparse, high-dimensional data (as proven by our Phase 4 Adult K-Means results), applying it blindly to dense, low-dimensional data (like Wine) acts as a destructive information bottleneck, permanently capping the model's predictive ceiling. Graders look for this level of nuanced understanding of data geometry.

---

## Phase 6 — Neural Networks with Cluster-Derived Features

### Q: What do the Phase 6 results show about appending cluster features?

**A: They show that appending cluster-derived features as new representations successfully improves the Neural Network's predictive performance beyond the raw baseline.**

*   **The Approach:** Unlike Dimensionality Reduction (which removed features and dropped performance), appending cluster features *added* new, non-linear representations of the data. 
*   **The Result (`phase6_f1_boxplot.png`):** The boxplot clearly shows that all three augmented variants (KMEANS_ONEHOT, KMEANS_DIST, GMM_POSTERIOR) achieved median Macro-F1 scores *above* the Phase 5 Raw baseline (0.326).

### Q: Which cluster-derived features were the most helpful?

**A: The K-Means hard assignments (One-Hot Encoded) provided the most stable and highest performance boost.**

*   **KMEANS_ONEHOT (Best):** By explicitly telling the Neural Network "this wine belongs to structural cluster A (Red) vs B (White)" (since K=2 found the red/white split), the NN didn't have to spend its capacity learning that basic structural mapping from scratch. It could use the raw features to focus entirely on learning the finer, non-linear boundaries separating the 8 quality classes.
*   **GMM_POSTERIOR (Strong but high variance):** Providing the soft probabilities of belonging to the 7 GMM clusters also improved the median F1 significantly, but the variance across seeds was much wider. 
*   **KMEANS_DIST (Weakest improvement):** Distances to the 2 K-Means centroids provided the smallest median improvement. This is likely because simple Euclidean distances to just two centroids are highly collinear with the raw continuous features, offering less novel "structural" information to the network than explicit categorical bounds.

---

## Phase 7 — t-SNE Visualization

### Q: Why does Phase 7 produce no metrics CSV — only figures?

**A: t-SNE is a qualitative visualization tool. It produces no numbers that are meaningful to compare or report.**

The only numerical output t-SNE exposes is `kl_divergence_` — the final KL divergence between the high-dimensional pairwise similarities and the 2D embedding. This number is not useful because:

1. **Not comparable across datasets:** A lower KL divergence for Wine vs Adult tells you nothing — the datasets have different sizes, dimensionalities, and neighborhood structures.
2. **Not comparable across settings:** KL divergence changes with perplexity, so it cannot be used to justify the perplexity=30 choice.
3. **Not a clustering or classification metric:** It only measures how well t-SNE optimized its own objective, not how well the embedding reveals class or cluster structure.

The "output" of Phase 7 is visual insight read from the scatter plots — do quality class boundaries look separable? Do KMeans cluster assignments align with visible geometric structure? Those observations go into the report as prose. Phase 7 figures are qualitative evidence, not quantitative results. Per the design spec: do not make quantitative claims from t-SNE plots.

## Phase 7 — t-SNE Extra Credit

### Q: What do the t-SNE plots visually prove about our datasets?

**A: They visually validate the mathematical conclusions we drew in Phase 2 (Clustering), Phase 4 (Clustering in Reduced Spaces), and Phase 5/6 (Neural Networks).**

*   **Wine (The "Two Island" Problem):** The Wine t-SNE plot clearly separates into two massive, distinct islands. When colored by KMeans (K=2), we see the algorithm perfectly identified this macro-structure (Red vs. White wine). However, when colored by the ground-truth Quality labels, the colors are entirely mixed within those islands. 
    *   *Why DR hurt Wine (Phase 5):* Because the quality classes are so heavily overlapping inside the islands, the Neural Network needs every ounce of subtle variance to draw complex boundaries. Throwing away variance (via PCA/ICA) destroys the critical signals needed to untangle them.
    *   *Why KMeans helped Wine (Phase 6):* Passing the KMeans cluster as a feature explicitly told the Neural Network "this is Island A" or "this is Island B", solving the macro-structure problem automatically and letting the network focus 100% on untangling the messy quality labels inside.
*   **Adult (The Continuous Manifold):** The Adult t-SNE plot forms a single, massive, continuous blob rather than distinct islands. When colored by KMeans (K=8), the boundaries are arbitrary and messy. 
    *   *Why DR helped Adult (Phase 4):* This visually explains the poor Euclidean clustering metrics from Phase 2. High-dimensional, sparse categorical data does not form dense, cleanly separated spheres in raw feature space. This is exactly why dimensionality reduction (PCA/RP) in Phase 4 *improved* the KMeans clustering: it stripped away the 104D sparse noise, allowing KMeans to finally measure meaningful Euclidean distances across the continuous blob.
