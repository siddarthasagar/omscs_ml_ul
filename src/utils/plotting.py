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
    1×3 subplot heatmap: Silhouette, Calinski-Harabasz, Davies-Bouldin.
    Each subplot has its own color scale (per-metric normalization).
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

    # (metric_col, label, cmap, fmt)
    metrics = [
        ("silhouette", "Silhouette (↑)", "RdYlGn", ".3f"),
        ("calinski_harabasz", "Calinski-Harabasz (↑)", "RdYlGn", ".0f"),
        ("davies_bouldin", "Davies-Bouldin (↓)", "RdYlGn_r", ".3f"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase 4 — Clustering Metrics in Reduced Spaces", fontsize=13)

    for ax, (col, label, cmap, fmt) in zip(axes, metrics):
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
                    matrix[i, j] = rows[col].iloc[0]

        valid = matrix[~np.isnan(matrix)]
        vmin, vmax = (valid.min(), valid.max()) if len(valid) else (0, 1)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(dr_methods)))
        ax.set_xticklabels(dr_methods)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(label, fontsize=11)

        for i in range(len(combos)):
            for j in range(len(dr_methods)):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        format(matrix[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=8,
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
    2×3 subplot grid: top row = KMeans metrics (Silhouette, CH, DB),
    bottom row = GMM metrics (Silhouette, BIC, AIC).
    Each subplot: X-axis = Raw/PCA/ICA/RP, with a dashed baseline from the Raw bar.
    df_reduced: Phase 4 rows for this dataset.
    df_raw: Phase 2 rows at frozen K with all metric columns.
    Saves to out_dir/{dataset_name}_phase4_bar.png.
    """
    dr_labels = ["Raw", "PCA", "ICA", "RP"]
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
    x = np.arange(len(dr_labels))
    width = 0.6

    # (clusterer, metric_col, ylabel, higher_is_better)
    subplots = [
        ("KMeans", "silhouette", "Silhouette", True),
        ("KMeans", "calinski_harabasz", "Calinski-Harabasz", True),
        ("KMeans", "davies_bouldin", "Davies-Bouldin", False),
        ("GMM", "silhouette", "Silhouette", True),
        ("GMM", "bic", "BIC", False),
        ("GMM", "aic", "AIC", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"{dataset_name.title()} — Raw vs Reduced Space Clustering", fontsize=13
    )

    for ax, (clusterer, metric, ylabel, higher) in zip(axes.flat, subplots):
        values = []
        for label in dr_labels:
            if label == "Raw":
                row = df_raw[df_raw["clusterer"] == clusterer]
                val = (
                    row[metric].iloc[0]
                    if (
                        not row.empty
                        and metric in row.columns
                        and pd.notna(row[metric].iloc[0])
                    )
                    else np.nan
                )
            else:
                row = df_reduced[
                    (df_reduced["dr_method"] == label)
                    & (df_reduced["clusterer"] == clusterer)
                ]
                val = (
                    row[metric].iloc[0]
                    if (not row.empty and pd.notna(row[metric].iloc[0]))
                    else np.nan
                )
            values.append(val)

        _bars = ax.bar(x, values, width, color=colors)

        # Dashed baseline from Raw value
        raw_val = values[0]
        if not np.isnan(raw_val):
            ax.axhline(
                raw_val,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Raw baseline",
            )

        direction = "↑ better" if higher else "↓ better"
        ax.set_title(f"{clusterer} — {ylabel} ({direction})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dr_labels)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"{dataset_name}_phase4_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_f1_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    title: str = "Wine NN Macro-F1 Comparison",
    out_name: str = "f1_boxplot.png",
    baseline_val: float | None = None,
    baseline_label: str | None = None,
) -> Path:
    """
    Boxplot of val_f1 across seeds, one box per variant.
    df must have cols: variant, val_f1. Variants are taken from df in their natural order.

    Args:
        baseline_val:  If provided, draws a dashed horizontal line at this value.
        baseline_label: Legend label for the baseline line. Defaults to "Baseline = {val}".
    Saves to out_dir/{out_name}.
    """
    variants = list(dict.fromkeys(df["variant"]))  # preserve insertion order, deduplicate
    data = [df[df["variant"] == v]["val_f1"].values for v in variants]
    palette = ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[v.upper() for v in variants], patch_artist=True)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if baseline_val is not None:
        label = baseline_label or f"Baseline = {baseline_val:.3f}"
        ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.2, label=label)

    ax.set(title=title, xlabel="Input Variant", ylabel="Val Macro-F1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_learning_curves(history_df: pd.DataFrame, out_dir: Path) -> Path:
    """
    All variants overlaid on shared axes: 1×3 subplots (Train Loss, Val Loss, Val F1).
    Each variant is one colored line (mean across seeds) with a ±1 std shaded band.
    history_df must have cols: variant, seed, epoch, train_loss, val_loss, val_f1.
    Saves to out_dir/phase5_learning_curves.png.
    """
    variants = ["raw", "pca", "ica", "rp"]
    colors = {"raw": "tab:gray", "pca": "tab:blue", "ica": "tab:orange", "rp": "tab:green"}
    n_seeds = history_df["seed"].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Phase 5 — Wine NN Learning Curves ({n_seeds} seeds, mean ± std)", fontsize=12)

    panels = [
        (axes[0], "train_loss", "Train Loss",  "Loss"),
        (axes[1], "val_loss",   "Val Loss",    "Loss"),
        (axes[2], "val_f1",     "Val Macro-F1","F1"),
    ]

    for ax, metric, title, ylabel in panels:
        for v in variants:
            agg = (
                history_df[history_df["variant"] == v]
                .groupby("epoch")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            epochs = agg["epoch"]
            color = colors[v]
            ax.plot(epochs, agg["mean"], color=color, label=v.upper())
            ax.fill_between(epochs, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            alpha=0.15, color=color)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "phase5_learning_curves.png"
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
