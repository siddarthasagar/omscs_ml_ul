"""Plotting utilities for UL experiment artifacts."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_kmeans_sweep(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> Path:
    """
    2×2 figure: Elbow (inertia), Silhouette, Calinski-Harabasz, Davies-Bouldin vs k.
    Saves to out_dir/{dataset_name}_kmeans.png.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{dataset_name.title()} — K-Means sweep", fontsize=13)

    axes[0, 0].plot(df["k"], df["inertia"], marker="o")
    axes[0, 0].set(title="Elbow (Inertia)", xlabel="k", ylabel="Inertia")

    axes[0, 1].plot(df["k"], df["silhouette"], marker="o", color="tab:orange")
    axes[0, 1].set(title="Silhouette (↑)", xlabel="k", ylabel="Silhouette")

    axes[1, 0].plot(df["k"], df["calinski_harabasz"], marker="o", color="tab:green")
    axes[1, 0].set(title="Calinski-Harabasz (↑)", xlabel="k", ylabel="CH Score")

    axes[1, 1].plot(df["k"], df["davies_bouldin"], marker="o", color="tab:red")
    axes[1, 1].set(title="Davies-Bouldin (↓)", xlabel="k", ylabel="DB Score")

    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_kmeans.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_gmm_sweep(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> Path:
    """
    1×2 figure: BIC+AIC (same axes), Silhouette vs n_components.
    Saves to out_dir/{dataset_name}_gmm.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{dataset_name.title()} — GMM sweep", fontsize=13)

    axes[0].plot(df["n_components"], df["bic"], marker="o", label="BIC")
    axes[0].plot(df["n_components"], df["aic"], marker="s", label="AIC")
    axes[0].set(title="BIC / AIC (↓)", xlabel="n_components", ylabel="Score")
    axes[0].legend()

    axes[1].plot(df["n_components"], df["silhouette"], marker="o", color="tab:orange")
    axes[1].set(title="Silhouette (↑)", xlabel="n_components", ylabel="Silhouette")

    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_gmm.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
