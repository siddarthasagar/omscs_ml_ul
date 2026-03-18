# Unsupervised Assignment FAQ
**CS 7641: Machine Learning**

---

## Overview

You have completed two assignments that framed problems with labels for classification or continuous values for regression. Those settings provided clear targets. In this third assignment, you remove labels and work in an unsupervised setting.

Because of that shift, the assignment is more open ended. There is no single correct approach. This FAQ is meant to keep you from biasing your choices and to help you present a rigorous analysis.

---

## FAQ

### What exactly am I doing?

The assignment description defines each step. At a high level, you will:

1. Run clustering on each of your two datasets.
2. Run dimensionality reduction on each of your two datasets.
3. Run dimensionality reduction followed by clustering on each of your two datasets.
4. On either dataset (Wine or Adult), but only one dataset, apply each linear dimensionality reduction method and compare against your best neural network from the OL report.
5. On the same dataset, use only cluster derived features as inputs to your network and compare against your OL baseline.

Most students use scikit-learn for clustering and dimensionality reduction and PyTorch for neural networks. You may mix libraries as needed or implement pieces in PyTorch when appropriate. Required algorithms by step:

- **Clustering:** Expectation Maximization and K-Means.
- **Dimensionality reduction:** PCA, ICA, and Random Projection.
- **Step 3** repeats step 1 after transforming data with step 2.
- **Step 4** refits your best OL neural network on each transformed space.
- **Step 5** refits your best OL neural network on features derived from clustering.

---

### Library notes in practice

- **K-Means:** `sklearn.cluster.KMeans`. In PyTorch you can implement Lloyd iterations with `torch.cdist` and tensor updates or use a thin wrapper package.
- **Expectation Maximization for GMM:** `sklearn.mixture.GaussianMixture`. In PyTorch you can code the E and M steps with `torch.distributions.MultivariateNormal` or a probabilistic library that runs on PyTorch.
- **PCA:** `sklearn.decomposition.PCA`. In PyTorch use `torch.pca_lowrank` or `torch.linalg.svd` and project with the learned loadings.
- **ICA:** `sklearn.decomposition.FastICA`. PyTorch has no built-in ICA; use scikit-learn or implement FastICA with whitening via SVD and a fixed point update.
- **Random Projection:** `sklearn.random_projection` for Gaussian or Sparse RP. In PyTorch, sample a fixed random matrix with `torch.randn` or a sparse initializer and multiply with `X @ R`; compute a pseudo-inverse with `torch.linalg.pinv` for reconstruction error.
- **Neural networks:** use your OL code path in PyTorch. Keep the optimizer, scheduler, batch size, and early stopping logic comparable across input spaces so the comparison is fair.

---

### Is there extra credit?

Yes. You may add a manifold learning method as a fourth dimensionality reduction technique for analysis and visualization such as Isomap, LLE, UMAP, or t-SNE with care for hyperparameters and scale. A great place to start is here: https://sites.gatech.edu/omscs7641/2024/03/10/no-straight-lines-here-the-wacky-world-of-non-linear-manifold-learning/

---

### How do I use the clustering output from step 1 for step 5?

There is no single preferred approach. You can use cluster labels one-hot or target encoded, distances to centroids, soft responsibilities from EM, or other cluster statistics. You may append to the original features, replace them, or mix them. Justify your design.

---

### What plots should I include for the unsupervised parts?

A sensible process for steps 1–3 is:

1. Choose the number of clusters or components without using ground truth labels.
2. Evaluate the resulting clustering or projection. For evaluation you may reference labels, but selection must remain label-free.

**Concrete heuristics:**

- **PCA:** inspect explained variance or singular values to select component count.
- **ICA:** inspect absolute kurtosis of components and consider whitening via PCA as in FastICA.
- **Random Projection:** inspect reconstruction error using a pseudo-inverse of the projection matrix.

Remember that components are linear combinations of original features. Inspect loadings to interpret what each component represents.

---

### What about plots for the supervised parts?

Reuse your OL evaluation habits. Ask whether dimensionality reduction or cluster features changed generalization, convergence behavior, or wall clock time. Learning curves and timing analyses are appropriate. Explain why you observe differences.

---

### What does "optimal" mean here?

It depends on your stated objective. If you want high separation and dense clusters, use metrics aligned with that such as silhouette, Calinski-Harabasz, and Davies-Bouldin. You may mix metrics across algorithms if you justify your choices. **Justify, justify, justify.**

---

### I swept cluster counts or component counts. How do I pick a configuration?

Use a clear rule. Qualitative rules include elbow or knee detection. Quantitative rules include thresholds on a metric that reflect domain knowledge. State your rule and stick to it.

---

### I got more features after projection.

That can happen, for example when adding cluster statistics. Dimensionality reduction is not guaranteed to reduce dimensionality in every pipeline. If the projection is not helpful, analyze why in light of the algorithm assumptions.

---

### Do I need train and test splits?

Yes. Hold out your test set and never use it to fit unsupervised models. Fit unsupervised models on the training set. When evaluating supervised performance, transform the test set with models fitted on the training set.

---

### All of my features are categorical.

You can try clustering designed for categorical data such as KModes or use a mixed-type distance such as Gower. If a method underperforms on your data, explain why and support your claim.

---

### My input layer changed. Should I refit my neural network?

Yes. Refit on each transformed space. You may see faster convergence or lower training time even if peak accuracy is unchanged. Keep training controls consistent so the comparison is fair.

---

### What about Random Projection relative to PCA and ICA?

Random Projection is efficient because the core step is multiplying by a random matrix. Sparse random projection can reduce cost further with little variance increase. RP assumes pairwise Euclidean distances are meaningful. If K-Means or KNN struggle on your data, RP may also struggle. If Euclidean geometry is appropriate, RP is a strong first option. The term "Gaussian" refers to the sampling of the random matrix, not your input feature distributions.

---

## Guidance for Selecting Component Counts and Further Reading

**PCA** — Use explained variance or singular values from `sklearn.decomposition.PCA` or from `torch.pca_lowrank` to choose how much variance to retain.

**ICA** — Assume a number of statistically independent sources. Use absolute kurtosis to guide selection. FastICA typically applies PCA whitening first.

**Random Projection** — Minimizing reconstruction error approximates maximizing preserved variance. In scikit-learn, compute reconstruction with inverse components or use a pseudo-inverse. In PyTorch use `torch.linalg.pinv`.

---

## References and Additional Resources

- **How to Evaluate Features after Dimensionality Reduction.** Course blog. Shikun Liu and Theodore LaGrow. Practical guide to judging DR outputs with metrics and visual checks. Useful when deciding component counts and validating that transformed spaces help downstream tasks.

- **No Straight Lines Here: The Wacky World of Non-Linear Manifold Learning.** Course blog. Aviral Agrawal and Theodore LaGrow. Short overview of non-linear DR methods and where they shine or fail. Helpful for thinking about UMAP and t-SNE as analysis and visualization tools rather than feature engines for classifiers.

- **Independent Component Analysis.** Aapo Hyvärinen, Juha Karhunen, Erkki Oja. Book-length treatment of ICA theory and practice. Good for understanding whitening, non-Gaussianity, and the statistical assumptions behind FastICA and related methods.

- **Independent Component Analysis: Algorithms and Applications.** Aapo Hyvärinen, Erkki Oja. Survey-style paper that connects the objective functions, contrast functions, and algorithms used for ICA. Useful when justifying kurtosis-based model selection and interpreting components.

- **Fast and Robust Fixed Point Algorithms for Independent Component Analysis.** Aapo Hyvärinen. NeurIPS paper introducing the fixed-point FastICA approach. Helpful for explaining why whitening plus a simple nonlinearity can yield efficient ICA in practice.

---

> **Final note:** Components are interpretable as combinations of original features. Use loadings to relate components back to features and to generate hypotheses about what the model found.
