"""Plotting utilities for UL experiment artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_pca_variance(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> Path:
    """
    1×2 figure: per-component explained variance bar + cumulative variance line.
    df must have cols: component, explained_variance, cumulative_variance.
    Saves to out_dir/{dataset_name}_pca_variance.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{dataset_name.title()} — PCA Explained Variance", fontsize=13)

    axes[0].bar(df["component"], df["explained_variance"], color="tab:blue")
    axes[0].set(
        title="Per-Component Variance",
        xlabel="Component",
        ylabel="Explained Variance Ratio",
    )

    axes[1].plot(
        df["component"], df["cumulative_variance"], marker="o", color="tab:blue"
    )
    axes[1].axhline(0.90, color="tab:red", linestyle="--", label="90% threshold")
    axes[1].set(
        title="Cumulative Variance",
        xlabel="Component",
        ylabel="Cumulative Explained Variance",
    )
    axes[1].legend()

    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_pca_variance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ica_kurtosis(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> Path:
    """
    Bar chart of absolute kurtosis per ICA component, sorted descending.
    df must have cols: component, kurtosis.
    Draws a dashed median-kurtosis threshold line and notes n_selected in title.
    Saves to out_dir/{dataset_name}_ica_kurtosis.png.
    """
    df_sorted = df.reindex(df["kurtosis"].abs().sort_values(ascending=False).index)

    abs_kurt = df_sorted["kurtosis"].abs()
    threshold = float(df["kurtosis"].abs().median())
    n_selected = max(int((df["kurtosis"].abs() >= threshold).sum()), 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(df_sorted)), abs_kurt, color="tab:purple")
    ax.axhline(
        threshold,
        color="tab:red",
        linestyle="--",
        label=f"Median threshold = {threshold:.2f}  →  {n_selected} selected",
    )
    ax.legend(fontsize=9)
    ax.set(
        title=(
            f"{dataset_name.title()} — ICA Absolute Kurtosis "
            f"(n_selected={n_selected}  ↑ = more non-Gaussian)"
        ),
        xlabel="Component (sorted by |kurtosis|)",
        ylabel="|Kurtosis|",
    )
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted["component"].astype(int), fontsize=7)

    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_ica_kurtosis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_phase4_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Heatmap of silhouette scores across all 12 Phase 4 combinations.
    df must have cols: dataset, dr_method, clusterer, silhouette.
    Rows = (dataset, clusterer), Cols = DR method.
    Saves to out_dir/phase4_clustering_heatmap.png.
    """
    combos = [
        ("wine", "KMeans"),
        ("wine", "GMM"),
        ("adult", "KMeans"),
        ("adult", "GMM"),
    ]
    dr_methods = ["PCA", "ICA", "RP"]
    row_labels = [f"{d.title()} {c}" for d, c in combos]

    matrix = np.full((len(combos), len(dr_methods)), np.nan)
    for i, (ds, cl) in enumerate(combos):
        for j, dr in enumerate(dr_methods):
            mask = (
                (df["dataset"] == ds)
                & (df["clusterer"] == cl)
                & (df["dr_method"] == dr)
            )
            rows = df[mask]
            if not rows.empty:
                matrix[i, j] = rows["silhouette"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.5)
    fig.colorbar(im, ax=ax, label="Silhouette")

    ax.set_xticks(range(len(dr_methods)))
    ax.set_xticklabels(dr_methods)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Phase 4 — Silhouette in Reduced Spaces", fontsize=13)

    for i in range(len(combos)):
        for j in range(len(dr_methods)):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9
                )

    fig.tight_layout()
    out_path = out_dir / "phase4_clustering_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_phase4_comparison(
    df_reduced: pd.DataFrame,
    df_raw: pd.DataFrame,
    dataset_name: str,
    out_dir: Path,
) -> Path:
    """
    Grouped bar: silhouette for raw vs PCA vs ICA vs RP, grouped by clusterer.
    df_reduced: Phase 4 rows for this dataset (cols: dr_method, clusterer, silhouette).
    df_raw: Phase 2 rows for this dataset at frozen K (cols: clusterer, silhouette).
    Saves to out_dir/{dataset_name}_phase4_bar.png.
    """
    clusterers = ["KMeans", "GMM"]
    dr_labels = ["Raw", "PCA", "ICA", "RP"]
    x = np.arange(len(clusterers))
    width = 0.18
    offsets = (
        np.linspace(-(len(dr_labels) - 1) / 2, (len(dr_labels) - 1) / 2, len(dr_labels))
        * width
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]

    for k, (label, color, offset) in enumerate(zip(dr_labels, colors, offsets)):
        values = []
        for cl in clusterers:
            if label == "Raw":
                row = df_raw[(df_raw["clusterer"] == cl)]
                values.append(row["silhouette"].iloc[0] if not row.empty else 0.0)
            else:
                row = df_reduced[
                    (df_reduced["dr_method"] == label) & (df_reduced["clusterer"] == cl)
                ]
                values.append(row["silhouette"].iloc[0] if not row.empty else 0.0)
        ax.bar(x + offset, values, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(clusterers)
    ax.set_ylabel("Silhouette Score")
    ax.set_title(
        f"{dataset_name.title()} — Silhouette: Raw vs Reduced Spaces", fontsize=13
    )
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_phase4_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rp_stability(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> Path:
    """
    Line plot of reconstruction error per seed across RP seeds.
    df must have cols: seed, reconstruction_error.
    Saves to out_dir/{dataset_name}_rp_stability.png.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["seed"], df["reconstruction_error"], marker="o", color="tab:brown")
    ax.axhline(
        df["reconstruction_error"].mean(), color="tab:red", linestyle="--", label="Mean"
    )
    ax.fill_between(
        df["seed"],
        df["reconstruction_error"].mean() - df["reconstruction_error"].std(),
        df["reconstruction_error"].mean() + df["reconstruction_error"].std(),
        alpha=0.2,
        color="tab:red",
        label="±1 std",
    )
    n_components = int(df["n_components"].iloc[0])
    ax.set(
        title=f"{dataset_name.title()} — RP Reconstruction Error Across Seeds (n_components={n_components})",
        xlabel="Seed",
        ylabel="MSE Reconstruction Error",
    )
    ax.legend()

    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_rp_stability.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
