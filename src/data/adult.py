from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import DATA_DIR


def load_adult(
    seed: int = 42,
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, split, and preprocess the Adult Income dataset.

    Preprocessing contract:
    - Split on RAW data before any transforms (preserves SL/OL test set)
    - 80/20 → train+val / test, then 75/25 → train / val (60/20/20 final)
    - ColumnTransformer: StandardScaler for numeric, OneHotEncoder for categorical
    - All transforms fit on X_train only
    - LabelEncoder fit on y_train only

    Returns:
        X_train, X_val, X_test  — float32
        y_train, y_val, y_test  — int64
    """
    path = Path(data_dir) / "adult.csv"
    frame = pd.read_csv(path)

    target = frame["class"].to_numpy()

    # Split 1: 80/20 on raw data
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

    # Drop target after split
    train_features = train_df.drop(columns=["class"])
    val_features = val_df.drop(columns=["class"])
    test_features = test_df.drop(columns=["class"])

    categorical_cols = list(
        train_features.select_dtypes(include=["object", "string"]).columns
    )
    numeric_cols = [c for c in train_features.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    X_train = preprocessor.fit_transform(train_features).astype(np.float32)
    X_val = preprocessor.transform(val_features).astype(np.float32)
    X_test = preprocessor.transform(test_features).astype(np.float32)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw).astype(np.int64)
    y_val = encoder.transform(y_val_raw).astype(np.int64)
    y_test = encoder.transform(y_test_raw).astype(np.int64)

    return X_train, X_val, X_test, y_train, y_val, y_test
