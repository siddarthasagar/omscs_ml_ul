"""Dimensionality reduction wrappers: PCA, ICA, Random Projection, t-SNE."""

import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import SparseRandomProjection


def fit_pca(
    X_train: np.ndarray,
    n_components: int | None = None,
) -> tuple[PCA, np.ndarray]:
    """
    Fit PCA on X_train and return (fitted_pca, X_transformed).

    Args:
        X_train: Training data, shape (n_samples, n_features).
        n_components: Number of components. None = keep all (full spectrum).

    Returns:
        (pca, X_transformed) where pca.explained_variance_ratio_ gives the spectrum.
    """
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_train)
    return pca, X_transformed


def fit_ica(
    X_train: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[FastICA, np.ndarray]:
    """
    Fit FastICA on X_train and return (fitted_ica, kurtosis_array).

    Args:
        X_train: Training data.
        n_components: Number of independent components to extract.
        seed: Random state for reproducibility.

    Returns:
        (ica, kurtosis_array) where kurtosis_array[i] is the excess kurtosis
        of component i (higher absolute kurtosis = more non-Gaussian = more useful).
    """
    ica = FastICA(n_components=n_components, random_state=seed, max_iter=1000)
    X_transformed = ica.fit_transform(X_train)
    kurtosis_array = np.array(
        [stats.kurtosis(X_transformed[:, i]) for i in range(n_components)]
    )
    return ica, kurtosis_array


def fit_rp(
    X_train: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[SparseRandomProjection, np.ndarray]:
    """
    Fit SparseRandomProjection on X_train and return (fitted_rp, X_transformed).

    Args:
        X_train: Training data.
        n_components: Target dimensionality.
        seed: Random state for reproducibility.

    Returns:
        (rp, X_transformed).
    """
    rp = SparseRandomProjection(n_components=n_components, random_state=seed)
    X_transformed = rp.fit_transform(X_train)
    return rp, X_transformed


def rp_reconstruction_error(rp: SparseRandomProjection, X: np.ndarray) -> float:
    """
    Compute reconstruction error for a fitted SparseRandomProjection via pseudo-inverse.

    Args:
        rp: Fitted SparseRandomProjection.
        X: Original data used to fit or transform.

    Returns:
        Mean squared reconstruction error (scalar).
    """
    components = (
        rp.components_.toarray()
        if hasattr(rp.components_, "toarray")
        else rp.components_
    )
    X_projected = X @ components.T
    components_pinv = np.linalg.pinv(components)
    X_reconstructed = X_projected @ components_pinv.T
    return float(np.mean((X - X_reconstructed) ** 2))


def fit_tsne(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Fit t-SNE and return 2D embedding. Visualization only — never used as NN input.

    Args:
        X: Data to embed.
        seed: Random state.

    Returns:
        embedding of shape (n_samples, 2).
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    return tsne.fit_transform(X)
