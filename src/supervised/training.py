"""Wine NN training loop — single seed, returns per-epoch history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    NN_BETAS,
    NN_LR,
    NN_MAX_EPOCHS,
    NN_TRAIN_BATCH_SIZE,
    NN_VAL_BATCH_SIZE,
    NN_WEIGHT_DECAY,
)
from src.supervised.nn_baseline import WineNN


def train_wine_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """
    Train WineNN for one seed and return per-epoch history.

    Architecture and all hyperparameters are locked (see steering/tech.md).
    Only input_dim is inferred from X_train.shape[1].

    Args:
        X_train: float32 array, shape (n_train, input_dim)
        y_train: int64 array, shape (n_train,)
        X_val:   float32 array, shape (n_val, input_dim)
        y_val:   int64 array, shape (n_val,)
        seed:    Random seed for weight init and DataLoader shuffling.

    Returns:
        DataFrame with columns [epoch, train_loss, val_loss, val_f1].
        epoch is 1-indexed.
    """
    torch.manual_seed(seed)

    device = torch.device("cpu")
    input_dim = X_train.shape[1]
    model = WineNN(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NN_LR,
        betas=NN_BETAS,
        weight_decay=NN_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        ),
        batch_size=NN_TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        ),
        batch_size=NN_VAL_BATCH_SIZE,
        shuffle=False,
    )

    history = []
    for epoch in range(1, NN_MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss, n_samples = 0.0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch.to(device))
            loss = criterion(logits, y_batch.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            n_samples += len(y_batch)
        train_loss = total_loss / n_samples

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss_total, val_n = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device))
                val_loss_total += criterion(logits, y_batch.to(device)).item() * len(
                    y_batch
                )
                val_n += len(y_batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_targets.append(y_batch.numpy())
        val_loss = val_loss_total / val_n
        val_f1 = f1_score(
            np.concatenate(all_targets),
            np.concatenate(all_preds),
            average="macro",
            zero_division=0,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
            }
        )

    return pd.DataFrame(history)
