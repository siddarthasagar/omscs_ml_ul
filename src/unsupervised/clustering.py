import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def run_kmeans_sweep(
    X: np.ndarray,
    k_range: range,
    seed: int,
) -> pd.DataFrame:
    """
    Fit KMeans for each k in k_range and compute cluster quality metrics.

    Selection is label-free: silhouette, Calinski-Harabasz, Davies-Bouldin.

    Returns:
        DataFrame with columns [k, silhouette, calinski_harabasz, davies_bouldin]
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        records.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame(records)


def run_gmm_sweep(
    X: np.ndarray,
    n_range: range,
    seed: int,
    reg_covar: float = 1e-3,
) -> pd.DataFrame:
    """
    Fit GaussianMixture for each n_components in n_range and compute BIC/AIC.

    Also includes silhouette on hard (argmax) assignments for comparison with
    KMeans results in Phase 4.

    reg_covar regularizes the covariance to avoid singular matrices in
    high-dimensional spaces (e.g. Adult after OHE → 104 features).

    Returns:
        DataFrame with columns [n_components, bic, aic, silhouette]
    """
    X64 = X.astype(np.float64)  # GMM numerics are more stable in float64
    records = []
    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=seed, reg_covar=reg_covar)
        gmm.fit(X64)
        labels = gmm.predict(X64)
        records.append(
            {
                "n_components": n,
                "bic": gmm.bic(X),
                "aic": gmm.aic(X),
                "silhouette": silhouette_score(X, labels),
            }
        )
    return pd.DataFrame(records)
