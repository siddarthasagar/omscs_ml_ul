"""
Phase 4: Clustering in reduced spaces.

For each of 12 combinations (2 datasets × 3 DR methods × 2 clusterers), applies
the frozen n_components from Phase 3 to reduce X_train, then clusters at the
frozen K from Phase 2 (label-free selection).

Produces:
  artifacts/metrics/phase4_clustering/summary_table.csv  (12 rows)
  artifacts/figures/phase4_clustering/phase4_clustering_heatmap.png
  artifacts/figures/phase4_clustering/{wine,adult}_phase4_bar.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SEED_EXPLORE
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.unsupervised.reduction import fit_ica, fit_pca, fit_rp
from src.utils.logger import configure_logger
from src.utils.plotting import plot_phase4_comparison, plot_phase4_heatmap

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase4_clustering"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase4_clustering"
PHASE2_METRICS = ARTIFACTS_DIR / "metrics" / "phase2_clustering"

# ── Frozen values from ADR-002 (K) and Phase 3 design (n_components) ──────────
FROZEN = {
    "wine": {"pca": 8, "ica": 4, "rp": 8, "kmeans_k": 2, "gmm_n": 7},
    "adult": {"pca": 22, "ica": 11, "rp": 22, "kmeans_k": 8, "gmm_n": 7},
}


def reduce(X_train: np.ndarray, dr: str, n: int, seed: int) -> np.ndarray:
    """Apply a DR method and return X_reduced."""
    if dr == "PCA":
        _, X_r = fit_pca(X_train, n_components=n)
    elif dr == "ICA":
        ica, _ = fit_ica(X_train, n_components=n, seed=seed)
        X_r = ica.transform(X_train)
    elif dr == "RP":
        _, X_r = fit_rp(X_train, n_components=n, seed=seed)
    else:
        raise ValueError(f"Unknown DR method: {dr}")
    return X_r


def cluster_kmeans(X: np.ndarray, k: int, seed: int) -> dict:
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "inertia": km.inertia_,
        "bic": None,
        "aic": None,
    }


def cluster_gmm(X: np.ndarray, n: int, seed: int) -> dict:
    X64 = X.astype(np.float64)
    gmm = GaussianMixture(n_components=n, random_state=seed, reg_covar=1e-3)
    gmm.fit(X64)
    labels = gmm.predict(X64)
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "inertia": None,
        "bic": gmm.bic(X64),
        "aic": gmm.aic(X64),
    }


def load_phase2_baseline(dataset: str) -> pd.DataFrame:
    """Load Phase 2 CSV results at frozen K for raw-space baseline."""
    f = FROZEN[dataset]
    records = []

    km_csv = PHASE2_METRICS / f"{dataset}_kmeans.csv"
    if km_csv.exists():
        km_df = pd.read_csv(km_csv)
        row = km_df[km_df["k"] == f["kmeans_k"]]
        if not row.empty:
            records.append(
                {"clusterer": "KMeans", "silhouette": row["silhouette"].iloc[0]}
            )

    gmm_csv = PHASE2_METRICS / f"{dataset}_gmm.csv"
    if gmm_csv.exists():
        gmm_df = pd.read_csv(gmm_csv)
        row = gmm_df[gmm_df["n_components"] == f["gmm_n"]]
        if not row.empty:
            records.append(
                {"clusterer": "GMM", "silhouette": row["silhouette"].iloc[0]}
            )

    return pd.DataFrame(records)


def run_dataset(name: str, X_train: np.ndarray, log) -> list[dict]:
    """Run all 6 combos for one dataset. Returns list of result dicts."""
    f = FROZEN[name]
    dr_methods = ["PCA", "ICA", "RP"]
    results = []

    for dr in tqdm(dr_methods, desc=f"  {name.upper()} DR", leave=False):
        n = f[dr.lower()]
        log.info("  %s | %s | n_components=%d", name.upper(), dr, n)
        X_r = reduce(X_train, dr, n, SEED_EXPLORE)
        log.info("    reduced shape: %s", X_r.shape)

        # KMeans at frozen k
        k = f["kmeans_k"]
        log.info("    KMeans k=%d ...", k)
        km_metrics = cluster_kmeans(X_r, k, SEED_EXPLORE)
        log.info(
            "    KMeans silhouette=%.4f  CH=%.1f  DB=%.4f",
            km_metrics["silhouette"],
            km_metrics["calinski_harabasz"],
            km_metrics["davies_bouldin"],
        )
        results.append(
            {
                "dataset": name,
                "dr_method": dr,
                "n_components": n,
                "clusterer": "KMeans",
                "k": k,
                **km_metrics,
            }
        )

        # GMM at frozen n
        gmm_n = f["gmm_n"]
        log.info("    GMM n=%d ...", gmm_n)
        gmm_metrics = cluster_gmm(X_r, gmm_n, SEED_EXPLORE)
        log.info(
            "    GMM   silhouette=%.4f  BIC=%.2f  AIC=%.2f",
            gmm_metrics["silhouette"],
            gmm_metrics["bic"] or 0,
            gmm_metrics["aic"] or 0,
        )
        results.append(
            {
                "dataset": name,
                "dr_method": dr,
                "n_components": n,
                "clusterer": "GMM",
                "k": gmm_n,
                **gmm_metrics,
            }
        )

    return results


def main() -> None:
    run_id = "phase4"
    log = configure_logger(run_id)
    log.info("Phase 4 start — run_id=%s", run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "wine": load_wine(seed=SEED_EXPLORE)[0],
        "adult": load_adult(seed=SEED_EXPLORE)[0],
    }

    all_results = []
    for name, X_train in datasets.items():
        log.info("── %s | shape=%s ──", name.upper(), X_train.shape)
        results = run_dataset(name, X_train, log)
        all_results.extend(results)

    summary_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    log.info("Summary table → %s  (%d rows)", csv_path, len(summary_df))
    assert len(summary_df) == 12, f"Expected 12 rows, got {len(summary_df)}"

    # ── Figures ────────────────────────────────────────────────────────────────
    fig_path = plot_phase4_heatmap(summary_df, FIGURES_DIR)
    log.info("Heatmap → %s", fig_path)

    for name in datasets:
        df_reduced = summary_df[summary_df["dataset"] == name].copy()
        df_raw = load_phase2_baseline(name)
        if df_raw.empty:
            log.warning(
                "  No Phase 2 baseline CSVs found for %s — bar chart will show raw=0",
                name,
            )
        fig_path = plot_phase4_comparison(df_reduced, df_raw, name, FIGURES_DIR)
        log.info("Bar chart → %s", fig_path)

    log.info("── Phase 4 complete. Silhouette summary:")
    for _, row in summary_df.iterrows():
        log.info(
            "  %s | %s | %s | k=%d | sil=%.4f",
            row["dataset"],
            row["dr_method"],
            row["clusterer"],
            row["k"],
            row["silhouette"],
        )


if __name__ == "__main__":
    main()
