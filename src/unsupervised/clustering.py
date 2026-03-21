import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder


def run_kmeans_sweep(
    X: np.ndarray,
    k_range: range,
    seed: int,
) -> pd.DataFrame:
    """
    Fit KMeans for each k in k_range and compute cluster quality metrics.

    Selection is label-free: silhouette, Calinski-Harabasz, Davies-Bouldin.

    Returns:
        DataFrame with columns [k, inertia, silhouette, calinski_harabasz, davies_bouldin]
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        records.append(
            {
                "k": k,
                "inertia": km.inertia_,
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


# ── Phase 6: cluster feature builders ─────────────────────────────────────────

def make_kmeans_onehot(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit KMeans(k) on X_train, append one-hot cluster assignment to each split.
    Returns (X_train_aug, X_val_aug) with shape (n, original_dim + k).
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    train_labels = km.fit_predict(X_train).reshape(-1, 1)
    val_labels = km.predict(X_val).reshape(-1, 1)

    enc = OneHotEncoder(categories=[list(range(k))], sparse_output=False)
    train_oh = enc.fit_transform(train_labels).astype(np.float32)
    val_oh = enc.transform(val_labels).astype(np.float32)

    return np.hstack([X_train, train_oh]), np.hstack([X_val, val_oh])


def make_kmeans_dist(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit KMeans(k) on X_train, append Euclidean distance to each centroid.
    Returns (X_train_aug, X_val_aug) with shape (n, original_dim + k).
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    km.fit(X_train)

    def _dists(X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(
            X[:, np.newaxis, :] - km.cluster_centers_[np.newaxis, :, :], axis=2
        ).astype(np.float32)

    return np.hstack([X_train, _dists(X_train)]), np.hstack([X_val, _dists(X_val)])


def make_gmm_posterior(
    X_train: np.ndarray,
    X_val: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit GMM(n_components=n) on X_train, append soft posterior P(component|x).
    Returns (X_train_aug, X_val_aug) with shape (n_samples, original_dim + n).
    """
    gmm = GaussianMixture(n_components=n, random_state=seed, reg_covar=1e-3)
    gmm.fit(X_train.astype(np.float64))

    train_post = gmm.predict_proba(X_train.astype(np.float64)).astype(np.float32)
    val_post = gmm.predict_proba(X_val.astype(np.float64)).astype(np.float32)

    return np.hstack([X_train, train_post]), np.hstack([X_val, val_post])
