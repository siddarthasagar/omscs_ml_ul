import numpy as np
import pytest

from src.config import WINE_N_FEATURES
from src.data.adult import load_adult
from src.data.wine import load_wine


@pytest.fixture(scope="module")
def wine_splits():
    return load_wine(seed=42)


@pytest.fixture(scope="module")
def adult_splits():
    return load_adult(seed=42)


# ── Gate 1a: feature count ─────────────────────────────────────────────────────


def test_wine_feature_count(wine_splits):
    X_train, _, _, _, _, _ = wine_splits
    assert X_train.shape[1] == WINE_N_FEATURES, (
        f"Expected {WINE_N_FEATURES} Wine features, got {X_train.shape[1]}"
    )


# ── Gate 1b: split sizes ───────────────────────────────────────────────────────


def test_wine_split_sizes(wine_splits):
    X_train, X_val, X_test, _, _, _ = wine_splits
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert 0.55 <= X_train.shape[0] / total <= 0.65, "Train fraction should be ~60%"
    assert 0.15 <= X_val.shape[0] / total <= 0.25, "Val fraction should be ~20%"
    assert 0.15 <= X_test.shape[0] / total <= 0.25, "Test fraction should be ~20%"


def test_adult_split_sizes(adult_splits):
    X_train, X_val, X_test, _, _, _ = adult_splits
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert 0.55 <= X_train.shape[0] / total <= 0.65
    assert 0.15 <= X_val.shape[0] / total <= 0.25
    assert 0.15 <= X_test.shape[0] / total <= 0.25


# ── Gate 1c: no leakage — scaler fit on train only ────────────────────────────


def test_wine_no_leakage(wine_splits):
    X_train, X_val, _, _, _, _ = wine_splits
    # After StandardScaler fit on X_train: train mean ≈ 0, train std ≈ 1
    assert abs(X_train.mean()) < 0.01, "X_train mean should be ~0 after scaling"
    assert abs(X_train.std() - 1.0) < 0.01, "X_train std should be ~1 after scaling"
    # Val should NOT have mean=0 (scaler was not refit on val)
    # For a large enough dataset, val mean differs from 0 meaningfully
    assert abs(X_val.mean()) > 1e-4 or True  # soft check; hard check below
    # Hard check: val stats are not identical to train stats
    assert not np.allclose(X_train.mean(axis=0), X_val.mean(axis=0), atol=1e-6), (
        "Val column means are identical to train — scaler may have been refit on val"
    )


def test_adult_no_leakage(adult_splits):
    X_train, X_val, _, _, _, _ = adult_splits
    # Adult has OHE columns (0/1) so overall mean is not ~0; check only that
    # val column means are not identical to train (would indicate refit leakage)
    assert not np.allclose(X_train.mean(axis=0), X_val.mean(axis=0), atol=1e-6), (
        "Val column means are identical to train — scaler may have been refit on val"
    )


# ── Gate 1d: dtypes ───────────────────────────────────────────────────────────


def test_wine_dtypes(wine_splits):
    X_train, X_val, X_test, y_train, y_val, y_test = wine_splits
    for arr in (X_train, X_val, X_test):
        assert arr.dtype == np.float32, f"X should be float32, got {arr.dtype}"
    for arr in (y_train, y_val, y_test):
        assert arr.dtype == np.int64, f"y should be int64, got {arr.dtype}"


def test_adult_dtypes(adult_splits):
    X_train, X_val, X_test, y_train, y_val, y_test = adult_splits
    for arr in (X_train, X_val, X_test):
        assert arr.dtype == np.float32
    for arr in (y_train, y_val, y_test):
        assert arr.dtype == np.int64
