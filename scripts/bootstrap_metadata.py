"""
Bootstrap artifacts/metadata/phase{2..6}.json from existing metric CSVs.

Run this once after adding metadata write code to phase scripts, so that
run_phase_8_report_tables.py can read from JSONs immediately without
re-running the full (slow) pipeline.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR

# FROZEN_K is imported from the analysis script — it is the human decision
# recorded after reviewing the sweep report (see FROZEN_K comment there).
from scripts.run_phase_2_k_analysis import FROZEN_K  # noqa: E402

METRICS = ARTIFACTS_DIR / "metrics"
META_DIR = ARTIFACTS_DIR / "metadata"
META_DIR.mkdir(parents=True, exist_ok=True)


def _derive_frozen_n() -> dict:
    """Derive frozen n_components from phase 3 metric CSVs (no hardcoding)."""
    p = METRICS / "phase3_reduction"
    result = {}
    for ds in ("wine", "adult"):
        pca_df = pd.read_csv(p / f"{ds}_pca.csv")
        ica_df = pd.read_csv(p / f"{ds}_ica.csv")
        rp_df = pd.read_csv(p / f"{ds}_rp_stability.csv")

        # PCA: first component where cumulative variance >= 90%
        n_pca = int(np.searchsorted(pca_df["cumulative_variance"].values, 0.90) + 1)

        # ICA: components with |kurtosis| >= median(|kurtosis|), floor at 2
        abs_kurt = np.abs(ica_df["kurtosis"].values)
        n_ica = int(np.sum(abs_kurt >= np.median(abs_kurt)))
        n_ica = max(n_ica, 2)

        # RP: same target dim as PCA (read from stability CSV to confirm)
        n_rp = int(rp_df["n_components"].iloc[0])

        result[ds] = {"pca": n_pca, "ica": n_ica, "rp": n_rp}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2
# ─────────────────────────────────────────────────────────────────────────────
def build_phase2() -> dict:
    p = METRICS / "phase2_clustering"

    def km_row(ds):
        df = pd.read_csv(p / f"{ds}_kmeans.csv")
        return df[df["k"] == FROZEN_K[ds]["kmeans"]].iloc[0]

    def gm_row(ds):
        df = pd.read_csv(p / f"{ds}_gmm.csv")
        return df[df["n_components"] == FROZEN_K[ds]["gmm"]].iloc[0]

    ari_df = pd.read_csv(p / "ari_results.csv")

    def ari(ds, clusterer, label):
        row = ari_df[
            (ari_df["dataset"] == ds)
            & (ari_df["clusterer"] == clusterer)
            & (ari_df["label"] == label)
        ]
        return round(float(row.iloc[0]["ARI"]), 4) if not row.empty else None

    meta = {"frozen_k": FROZEN_K}
    for ds in ("wine", "adult"):
        kr = km_row(ds)
        gr = gm_row(ds)
        meta[ds] = {
            "kmeans": {
                "silhouette": round(float(kr["silhouette"]), 4),
                "calinski_harabasz": round(float(kr["calinski_harabasz"]), 2),
                "davies_bouldin": round(float(kr["davies_bouldin"]), 4),
            },
            "gmm": {
                "silhouette": round(float(gr["silhouette"]), 4),
                "bic": round(float(gr["bic"]), 2),
                "aic": round(float(gr["aic"]), 2),
            },
        }

    # ARI — wine has type+class labels, adult has only class
    meta["wine"]["ari"] = {
        "kmeans_type": ari("wine", "KMeans", "type"),
        "kmeans_class": ari("wine", "KMeans", "class"),
        "gmm_type": ari("wine", "GMM", "type"),
        "gmm_class": ari("wine", "GMM", "class"),
    }
    meta["adult"]["ari"] = {
        "kmeans_class": ari("adult", "KMeans", "class"),
        "gmm_class": ari("adult", "GMM", "class"),
    }
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3
# ─────────────────────────────────────────────────────────────────────────────
def build_phase3() -> dict:
    p = METRICS / "phase3_reduction"
    frozen_n = _derive_frozen_n()

    def pca_stats(ds, n_components):
        df = pd.read_csv(p / f"{ds}_pca.csv")
        n_features = len(df)
        pc1_var_pct = round(float(df.iloc[0]["explained_variance"]) * 100, 1)
        cumvar_at_n = float(
            df[df["component"] == n_components].iloc[0]["cumulative_variance"]
        )
        cumvar_at_n_pct = round(cumvar_at_n * 100, 1)
        comp_ratio_pct = round(n_components / n_features * 100)
        comp_ratio_x = round(n_features / n_components, 1)
        return {
            "pc1_var_pct": pc1_var_pct,
            "cumvar_at_n_pct": cumvar_at_n_pct,
            "comp_ratio_pct": comp_ratio_pct,
            "comp_ratio_x": comp_ratio_x,
        }, n_features

    wine_pca_stats, wine_n = pca_stats("wine", frozen_n["wine"]["pca"])
    adult_pca_stats, adult_n = pca_stats("adult", frozen_n["adult"]["pca"])

    return {
        "frozen_n": frozen_n,
        "wine": {
            "n_features": wine_n,
            "pca": wine_pca_stats,
        },
        "adult": {
            "n_features": adult_n,
            "pca": adult_pca_stats,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4
# ─────────────────────────────────────────────────────────────────────────────
def build_phase4() -> dict:
    df = pd.read_csv(METRICS / "phase4_clustering" / "summary_table.csv")
    meta: dict = {"silhouette": {}, "ari_raw_vs_reduced": {}}
    for _, row in df.iterrows():
        key = f"{row['dataset']}_{row['clusterer'].lower()}_{row['dr_method'].lower()}"
        meta["silhouette"][key] = round(float(row["silhouette"]), 4)
        meta["ari_raw_vs_reduced"][key] = round(float(row["raw_vs_reduced_ari"]), 4)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5
# ─────────────────────────────────────────────────────────────────────────────
def build_phase5() -> dict:
    df = pd.read_csv(METRICS / "phase5_nn_reduced" / "comparison_table.csv")
    variants = ["raw", "pca", "ica", "rp"]

    def vmean(col, v):
        return round(float(df[df["variant"] == v][col].mean()), 4)

    return {
        "mean_f1": {v: vmean("val_f1_final", v) for v in variants},
        "mean_timing_s": {v: vmean("train_time_s", v) for v in variants},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6
# ─────────────────────────────────────────────────────────────────────────────
def build_phase6() -> dict:
    df = pd.read_csv(METRICS / "phase6_nn_cluster" / "comparison_table.csv")
    variants = ["kmeans_onehot", "kmeans_dist", "gmm_posterior"]
    return {
        "mean_f1": {
            v: round(float(df[df["variant"] == v]["val_f1_final"].mean()), 4)
            for v in variants
        },
        "input_dim": {
            v: int(df[df["variant"] == v]["input_dim"].iloc[0]) for v in variants
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    builders = {
        2: build_phase2,
        3: build_phase3,
        4: build_phase4,
        5: build_phase5,
        6: build_phase6,
    }
    for n, fn in builders.items():
        meta = fn()
        path = META_DIR / f"phase{n}.json"
        path.write_text(json.dumps(meta, indent=2))
        print(f"  phase{n}.json → {path}")

    print(f"\nAll metadata written to {META_DIR}/")


if __name__ == "__main__":
    main()
