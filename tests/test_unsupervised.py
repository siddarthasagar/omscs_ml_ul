import inspect

import numpy as np

from src.unsupervised.clustering import run_gmm_sweep, run_kmeans_sweep

# Small synthetic data — fast for tests
RNG = np.random.default_rng(42)
X_SMALL = RNG.standard_normal((200, 4)).astype(np.float32)


# ── Gate 2a: label-free interface ─────────────────────────────────────────────


def test_kmeans_label_free():
    sig = inspect.signature(run_kmeans_sweep)
    assert "y" not in sig.parameters, "run_kmeans_sweep must not accept a label arg"
    assert "labels" not in sig.parameters, (
        "run_kmeans_sweep must not accept a label arg"
    )


def test_gmm_label_free():
    sig = inspect.signature(run_gmm_sweep)
    assert "y" not in sig.parameters
    assert "labels" not in sig.parameters


# ── Gate 2b: output schema ────────────────────────────────────────────────────


def test_kmeans_output_schema():
    df = run_kmeans_sweep(X_SMALL, k_range=range(2, 5), seed=42)
    assert list(df.columns) == [
        "k",
        "inertia",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    ]
    assert len(df) == 3
    assert df["k"].tolist() == [2, 3, 4]


def test_gmm_bic_aic_present():
    df = run_gmm_sweep(X_SMALL, n_range=range(2, 5), seed=42)
    assert "bic" in df.columns
    assert "aic" in df.columns
    assert "n_components" in df.columns
    assert len(df) == 3


# ── Gate 2c: metric ranges ────────────────────────────────────────────────────


def test_kmeans_silhouette_range():
    df = run_kmeans_sweep(X_SMALL, k_range=range(2, 4), seed=42)
    assert df["silhouette"].between(-1, 1).all()


def test_gmm_bic_finite():
    df = run_gmm_sweep(X_SMALL, n_range=range(2, 5), seed=42)
    assert df["bic"].notna().all(), "BIC values must be finite"
    assert df["aic"].notna().all(), "AIC values must be finite"
    assert (df["bic"] > 0).all(), "BIC should be positive"
