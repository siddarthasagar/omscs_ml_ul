"""
Phase 4: Clustering in reduced spaces.

For each of 12 combinations (2 datasets × 3 DR methods × 2 clusterers):
  1. Reduce X_train using frozen n_components from Phase 3.
  2. Re-select optimal K in the reduced space using the same label-free criteria
     as Phase 2 (KMeans: joint silhouette/CH/DB; GMM: BIC minimum).
  3. Cluster at the new reduced-space K and record metrics.

Produces:
  artifacts/metrics/phase4_clustering/summary_table.csv          (12 rows)
  artifacts/metrics/phase4_clustering/{ds}_{dr}_{alg}_sweep.csv  (12 sweep CSVs)
  artifacts/figures/phase4_clustering/phase4_clustering_heatmap.png
  artifacts/figures/phase4_clustering/{wine,adult}_phase4_bar.png
  artifacts/figures/phase4_clustering/{wine,adult}_phase4_reduced_sweeps.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
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
from src.utils.plotting import (
    plot_phase4_comparison,
    plot_phase4_heatmap,
    plot_phase4_reduced_sweeps,
)

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase4_clustering"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase4_clustering"
PHASE2_METRICS = ARTIFACTS_DIR / "metrics" / "phase2_clustering"
METADATA = ARTIFACTS_DIR / "metadata"


def _load_frozen() -> dict:
    """Load frozen n_components (phase3.json). K is re-selected per reduced space."""
    for n in (2, 3):
        path = METADATA / f"phase{n}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run 'make phase{n}' first.")
    fn = json.loads((METADATA / "phase3.json").read_text())["frozen_n"]
    return {
        ds: {"pca": fn[ds]["pca"], "ica": fn[ds]["ica"], "rp": fn[ds]["rp"]}
        for ds in ("wine", "adult")
    }


FROZEN_N = _load_frozen()


def reduce(X_train: np.ndarray, dr: str, n: int, seed: int) -> np.ndarray:
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


def sweep_reduced_space(
    X_r: np.ndarray, k_range: range, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run KMeans and GMM sweeps on reduced-space X_r.
    Same column schema as Phase 2 sweep CSVs.
    """
    km_rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(X_r)
        labels = km.labels_
        km_rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X_r, labels),
                "calinski_harabasz": calinski_harabasz_score(X_r, labels),
                "davies_bouldin": davies_bouldin_score(X_r, labels),
            }
        )
    km_df = pd.DataFrame(km_rows)

    gmm_rows = []
    for n in k_range:
        gmm = GaussianMixture(n_components=n, random_state=seed, reg_covar=1e-3).fit(
            X_r.astype(np.float64)
        )
        labels = gmm.predict(X_r.astype(np.float64))
        gmm_rows.append(
            {
                "n_components": n,
                "bic": gmm.bic(X_r.astype(np.float64)),
                "aic": gmm.aic(X_r.astype(np.float64)),
                "silhouette": silhouette_score(X_r, labels),
            }
        )
    gmm_df = pd.DataFrame(gmm_rows)
    return km_df, gmm_df


def select_k_reduced(km_df: pd.DataFrame, gmm_df: pd.DataFrame) -> tuple[int, int]:
    """
    Same joint criteria as Phase 2 Step 1:
      KMeans: highest silhouette; ties broken by highest CH then lowest DB.
      GMM: lowest BIC.
    """
    best_sil = km_df["silhouette"].max()
    candidates = km_df[km_df["silhouette"] >= best_sil - 1e-6].copy()
    if len(candidates) > 1:
        candidates = candidates.sort_values(
            ["calinski_harabasz", "davies_bouldin"],
            ascending=[False, True],
        )
    km_k = int(candidates.iloc[0]["k"])
    gmm_k = int(gmm_df.loc[gmm_df["bic"].idxmin(), "n_components"])
    return km_k, gmm_k


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


def load_phase2_baseline(dataset: str, frozen_k: dict) -> pd.DataFrame:
    """Load Phase 2 CSV results at raw-space frozen K for bar chart baseline."""
    records = []

    km_csv = PHASE2_METRICS / f"{dataset}_kmeans.csv"
    if km_csv.exists():
        km_df = pd.read_csv(km_csv)
        row = km_df[km_df["k"] == frozen_k[dataset]["kmeans"]]
        if not row.empty:
            r = row.iloc[0]
            records.append(
                {
                    "clusterer": "KMeans",
                    "silhouette": r["silhouette"],
                    "calinski_harabasz": r["calinski_harabasz"],
                    "davies_bouldin": r["davies_bouldin"],
                    "inertia": r["inertia"],
                    "bic": None,
                    "aic": None,
                }
            )

    gmm_csv = PHASE2_METRICS / f"{dataset}_gmm.csv"
    if gmm_csv.exists():
        gmm_df = pd.read_csv(gmm_csv)
        row = gmm_df[gmm_df["n_components"] == frozen_k[dataset]["gmm"]]
        if not row.empty:
            r = row.iloc[0]
            records.append(
                {
                    "clusterer": "GMM",
                    "silhouette": r["silhouette"],
                    "calinski_harabasz": None,
                    "davies_bouldin": None,
                    "inertia": None,
                    "bic": r["bic"],
                    "aic": r["aic"],
                }
            )

    return pd.DataFrame(records)


def run_dataset(
    name: str, X_train: np.ndarray, k_range: range, log
) -> tuple[list[dict], dict]:
    """
    Run all 6 combos for one dataset.
    Returns (results, sweep_data) where sweep_data is used for the sweep figure.
    """
    fn = FROZEN_N[name]
    dr_methods = ["PCA", "ICA", "RP"]
    results = []
    sweep_data: dict[str, tuple] = {}

    # Raw cluster assignments at raw-space frozen K (for ARI comparison only)
    frozen_k_raw = json.loads((METADATA / "phase2.json").read_text())["frozen_k"]
    raw_km = KMeans(
        n_clusters=frozen_k_raw[name]["kmeans"],
        random_state=SEED_EXPLORE,
        n_init="auto",
    ).fit_predict(X_train)
    raw_gmm = GaussianMixture(
        n_components=frozen_k_raw[name]["gmm"],
        random_state=SEED_EXPLORE,
        covariance_type="diag",
        reg_covar=1e-3,
    ).fit_predict(X_train.astype(np.float64))

    for dr in tqdm(dr_methods, desc=f"  {name.upper()} DR", leave=False):
        n = fn[dr.lower()]
        log.info("  %s | %s | n_components=%d", name.upper(), dr, n)
        X_r = reduce(X_train, dr, n, SEED_EXPLORE)

        # Re-select K in this reduced space
        km_sweep_df, gmm_sweep_df = sweep_reduced_space(X_r, k_range, SEED_EXPLORE)
        km_k, gmm_k = select_k_reduced(km_sweep_df, gmm_sweep_df)
        log.info("    reduced K — KMeans=%d  GMM=%d", km_k, gmm_k)

        # Save sweep CSVs
        dr_lower = dr.lower()
        km_sweep_df.to_csv(
            OUTPUT_DIR / f"{name}_{dr_lower}_kmeans_sweep.csv", index=False
        )
        gmm_sweep_df.to_csv(
            OUTPUT_DIR / f"{name}_{dr_lower}_gmm_sweep.csv", index=False
        )

        sweep_data[dr_lower] = (km_sweep_df, gmm_sweep_df, km_k, gmm_k)

        # KMeans at reduced-space K
        km_metrics = cluster_kmeans(X_r, km_k, SEED_EXPLORE)
        red_km = KMeans(
            n_clusters=km_k, random_state=SEED_EXPLORE, n_init="auto"
        ).fit_predict(X_r)
        ari_km = adjusted_rand_score(raw_km, red_km)
        km_metrics["raw_vs_reduced_ari"] = round(ari_km, 4)
        log.info(
            "    KMeans k=%d  silhouette=%.4f  CH=%.1f  DB=%.4f  ARI=%.4f",
            km_k,
            km_metrics["silhouette"],
            km_metrics["calinski_harabasz"],
            km_metrics["davies_bouldin"],
            ari_km,
        )
        results.append(
            {
                "dataset": name,
                "dr_method": dr,
                "n_components": n,
                "clusterer": "KMeans",
                "k": km_k,
                **km_metrics,
            }
        )

        # GMM at reduced-space K
        gmm_metrics = cluster_gmm(X_r, gmm_k, SEED_EXPLORE)
        red_gmm = GaussianMixture(
            n_components=gmm_k,
            random_state=SEED_EXPLORE,
            covariance_type="diag",
            reg_covar=1e-3,
        ).fit_predict(X_r.astype(np.float64))
        ari_gmm = adjusted_rand_score(raw_gmm, red_gmm)
        gmm_metrics["raw_vs_reduced_ari"] = round(ari_gmm, 4)
        log.info(
            "    GMM   n=%d  silhouette=%.4f  BIC=%.2f  AIC=%.2f  ARI=%.4f",
            gmm_k,
            gmm_metrics["silhouette"],
            gmm_metrics["bic"] or 0,
            gmm_metrics["aic"] or 0,
            ari_gmm,
        )
        results.append(
            {
                "dataset": name,
                "dr_method": dr,
                "n_components": n,
                "clusterer": "GMM",
                "k": gmm_k,
                **gmm_metrics,
            }
        )

    return results, sweep_data


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

    k_range = range(2, 21)
    all_results = []
    reduced_k_meta: dict = {}

    for name, X_train in datasets.items():
        log.info("── %s | shape=%s ──", name.upper(), X_train.shape)
        results, sweep_data = run_dataset(name, X_train, k_range, log)
        all_results.extend(results)

        # Collect reduced K values for metadata
        for dr_lower, (_, _, km_k, gmm_k) in sweep_data.items():
            reduced_k_meta[f"{name}_{dr_lower}"] = {"kmeans": km_k, "gmm": gmm_k}

    summary_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "summary_table.csv"
    summary_df.to_csv(csv_path, index=False)
    log.info("Summary table → %s  (%d rows)", csv_path, len(summary_df))
    assert len(summary_df) == 12, f"Expected 12 rows, got {len(summary_df)}"

    log.info("── Phase 4 complete. Silhouette summary (reduced-space K):")
    for _, row in summary_df.iterrows():
        log.info(
            "  %s | %s | %s | k=%d | sil=%.4f",
            row["dataset"],
            row["dr_method"],
            row["clusterer"],
            row["k"],
            row["silhouette"],
        )

    # ── Metadata JSON ──────────────────────────────────────────────────────────
    meta: dict = {
        "reduced_k": reduced_k_meta,
        "silhouette": {},
        "ari_raw_vs_reduced": {},
    }
    for _, row in summary_df.iterrows():
        key = f"{row['dataset']}_{row['clusterer'].lower()}_{row['dr_method'].lower()}"
        meta["silhouette"][key] = round(float(row["silhouette"]), 4)
        meta["ari_raw_vs_reduced"][key] = round(float(row["raw_vs_reduced_ari"]), 4)
    meta_dir = ARTIFACTS_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "phase4.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Metadata → %s", meta_path)
    for fig in visualize(meta_path):
        log.info("  Figure → %s", fig)


def visualize(checkpoint: Path) -> list[Path]:
    """Regenerate phase 4 figures from saved CSVs and metadata (no re-running)."""
    meta = json.loads(checkpoint.read_text())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figs: list[Path] = []

    for ds in ("wine", "adult"):
        sweep_data: dict = {}
        for dr in ("pca", "ica", "rp"):
            km_df = pd.read_csv(OUTPUT_DIR / f"{ds}_{dr}_kmeans_sweep.csv")
            gmm_df = pd.read_csv(OUTPUT_DIR / f"{ds}_{dr}_gmm_sweep.csv")
            km_k = meta["reduced_k"][f"{ds}_{dr}"]["kmeans"]
            gmm_k = meta["reduced_k"][f"{ds}_{dr}"]["gmm"]
            sweep_data[dr] = (km_df, gmm_df, km_k, gmm_k)
        figs.append(plot_phase4_reduced_sweeps(sweep_data, ds, FIGURES_DIR))

    summary_df = pd.read_csv(OUTPUT_DIR / "summary_table.csv")
    figs.append(plot_phase4_heatmap(summary_df, FIGURES_DIR))

    frozen_k_raw = json.loads((ARTIFACTS_DIR / "metadata" / "phase2.json").read_text())[
        "frozen_k"
    ]
    for ds in ("wine", "adult"):
        df_reduced = summary_df[summary_df["dataset"] == ds].copy()
        df_raw = load_phase2_baseline(ds, frozen_k_raw)
        figs.append(plot_phase4_comparison(df_reduced, df_raw, ds, FIGURES_DIR))

    return figs


if __name__ == "__main__":
    main()
