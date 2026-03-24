"""
Phase 2: Step 1 — Clustering on raw data.

Sweeps K-Means and GMM over k/n_components in range(2, 21) on X_train for
both Wine and Adult. Saves 4 CSVs + 4 PNGs to phase2_clustering artifacts.

Also computes ARI between frozen-K cluster assignments and ground-truth labels
(Wine: quality class + type; Adult: income class) and saves ari_results.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SEED_EXPLORE
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.unsupervised.clustering import run_gmm_sweep, run_kmeans_sweep
from src.utils.logger import configure_logger
from src.utils.plotting import plot_gmm_sweep, plot_kmeans_sweep

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase2_clustering"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase2_clustering"

# Frozen K from ADR-002
FROZEN_K = {"wine": {"kmeans": 2, "gmm": 7}, "adult": {"kmeans": 8, "gmm": 7}}


def compute_ari(X_train: np.ndarray, y_true: np.ndarray, name: str, log) -> list[dict]:
    """Cluster at frozen K and compute ARI vs ground-truth labels."""
    records = []
    k_km = FROZEN_K[name]["kmeans"]
    k_gm = FROZEN_K[name]["gmm"]

    km_labels = KMeans(
        n_clusters=k_km, random_state=SEED_EXPLORE, n_init="auto"
    ).fit_predict(X_train)
    gm_labels = GaussianMixture(
        n_components=k_gm,
        random_state=SEED_EXPLORE,
        covariance_type="diag",
        reg_covar=1e-3,
    ).fit_predict(X_train)

    # ARI vs the primary label (quality for wine, income for adult)
    for clusterer, labels, k in [("KMeans", km_labels, k_km), ("GMM", gm_labels, k_gm)]:
        ari = adjusted_rand_score(y_true, labels)
        log.info("  %s %s K=%d ARI vs class=%.4f", name.upper(), clusterer, k, ari)
        records.append(
            {
                "dataset": name,
                "clusterer": clusterer,
                "K": k,
                "label": "class",
                "ARI": round(ari, 4),
            }
        )

    # Wine also gets ARI vs binary type label (recovered from StandardScaled feature index 11)
    if name == "wine":
        type_labels = (X_train[:, 11] > 0).astype(int)
        for clusterer, labels, k in [
            ("KMeans", km_labels, k_km),
            ("GMM", gm_labels, k_gm),
        ]:
            ari_type = adjusted_rand_score(type_labels, labels)
            log.info("  Wine %s K=%d ARI vs type=%.4f", clusterer, k, ari_type)
            records.append(
                {
                    "dataset": name,
                    "clusterer": clusterer,
                    "K": k,
                    "label": "type",
                    "ARI": round(ari_type, 4),
                }
            )

    return records


def main() -> None:
    run_id = "phase2"
    log = configure_logger(run_id)
    log.info("Phase 2 start — run_id=%s", run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    wine_splits = load_wine(seed=SEED_EXPLORE)
    adult_splits = load_adult(seed=SEED_EXPLORE)
    datasets = {
        "wine": (wine_splits[0], wine_splits[3]),  # X_train, y_train (quality class)
        "adult": (adult_splits[0], adult_splits[3]),  # X_train, y_train (income class)
    }

    sweep_range = range(2, 21)

    for name, (X_train, _) in tqdm(datasets.items(), desc="Datasets"):
        log.info("── %s | shape=%s ──", name.upper(), X_train.shape)

        log.info("  K-Means sweep (k=2..20) ...")
        kmeans_df = run_kmeans_sweep(X_train, k_range=sweep_range, seed=SEED_EXPLORE)
        kmeans_path = OUTPUT_DIR / f"{name}_kmeans.csv"
        kmeans_df.to_csv(kmeans_path, index=False)
        log.info("  Saved  → %s", kmeans_path)
        fig_path = plot_kmeans_sweep(kmeans_df, name, FIGURES_DIR)
        log.info("  Figure → %s", fig_path)

        log.info("  GMM sweep (n=2..20) ...")
        gmm_df = run_gmm_sweep(X_train, n_range=sweep_range, seed=SEED_EXPLORE)
        gmm_path = OUTPUT_DIR / f"{name}_gmm.csv"
        gmm_df.to_csv(gmm_path, index=False)
        log.info("  Saved  → %s", gmm_path)
        fig_path = plot_gmm_sweep(gmm_df, name, FIGURES_DIR)
        log.info("  Figure → %s", fig_path)

    # ── ARI at frozen K ────────────────────────────────────────────────────────
    log.info("── ARI computation at frozen K ──")
    ari_records = []
    for name, (X_train, y_train) in datasets.items():
        ari_records.extend(compute_ari(X_train, y_train, name, log))

    ari_df = pd.DataFrame(ari_records)
    ari_path = OUTPUT_DIR / "ari_results.csv"
    ari_df.to_csv(ari_path, index=False)
    log.info("ARI results → %s", ari_path)
    log.info("\n%s", ari_df.to_string(index=False))

    log.info("── Phase 2 complete. Artifacts:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        log.info("   %s", p)
    for p in sorted(FIGURES_DIR.glob("*.png")):
        log.info("   %s", p)


if __name__ == "__main__":
    main()
