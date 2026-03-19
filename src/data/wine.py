from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import DATA_DIR, WINE_N_FEATURES


def load_wine(
    seed: int = 42,
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, split, and preprocess the Wine Quality dataset.

    Preprocessing contract (matches OL backbone exactly):
    - Split on RAW data before dropping any columns (preserves SL/OL test set)
    - 80/20 → train+val / test, then 75/25 of 80% → train / val (60/20/20 final)
    - Drop `quality` (leakage) and `class` (target); keep `type` (numeric 0/1)
    - StandardScaler fit on X_train only, applied to val and test
    - LabelEncoder fit on y_train only

    Returns:
        X_train, X_val, X_test  — float32, shape[1] == WINE_N_FEATURES (12)
        y_train, y_val, y_test  — int64
    """
    path = Path(data_dir) / "wine.csv"
    frame = pd.read_csv(path)

    target = frame["class"].to_numpy()

    # Split 1: 80/20 on raw data — matches SL/OL test set exactly
    full_train_df, test_df, y_full_train, y_test_raw = train_test_split(
        frame,
        target,
        test_size=0.2,
        random_state=seed,
        stratify=target,
    )

    # Split 2: 75/25 of full_train → 60/20/20 overall
    train_df, val_df, y_train_raw, y_val_raw = train_test_split(
        full_train_df,
        y_full_train,
        test_size=0.25,
        random_state=seed,
        stratify=y_full_train,
    )

    # Drop target and leakage columns after split
    drop_cols = ["class", "quality"]
    train_features = train_df.drop(columns=drop_cols)
    val_features = val_df.drop(columns=drop_cols)
    test_features = test_df.drop(columns=drop_cols)

    # All remaining columns are numeric (type is already 0/1 int)
    numeric_cols = list(train_features.columns)
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric_cols)],
        remainder="drop",
    )
    X_train = preprocessor.fit_transform(train_features).astype(np.float32)
    X_val = preprocessor.transform(val_features).astype(np.float32)
    X_test = preprocessor.transform(test_features).astype(np.float32)

    assert X_train.shape[1] == WINE_N_FEATURES, (
        f"Expected {WINE_N_FEATURES} features, got {X_train.shape[1]}"
    )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw).astype(np.int64)
    y_val = encoder.transform(y_val_raw).astype(np.int64)
    y_test = encoder.transform(y_test_raw).astype(np.int64)

    return X_train, X_val, X_test, y_train, y_val, y_test
