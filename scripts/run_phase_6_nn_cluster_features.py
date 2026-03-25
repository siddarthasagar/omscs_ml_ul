"""
Phase 6: Wine NN with cluster-derived features appended to the raw 12 inputs.

Three variants (frozen K from ADR-002: KMeans k=2, GMM n=7):
  kmeans_onehot  — raw(12) + one-hot assignment(2)   → input_dim=14
  kmeans_dist    — raw(12) + centroid distances(2)   → input_dim=14
  gmm_posterior  — raw(12) + GMM posteriors(7)       → input_dim=19

Cluster models are fit on X_train (seed=42), then applied to val.
NN training config is identical to Phase 5 (locked, see steering/tech.md).

Produces:
  artifacts/metrics/phase6_nn_cluster/{variant}/seed{seed}.csv  (30 files)
  artifacts/metrics/phase6_nn_cluster/comparison_table.csv      (30 rows)
  artifacts/figures/phase6_nn_cluster/phase6_f1_boxplot.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SEED_EXPLORE, SEEDS_REPORT
from src.data.wine import load_wine
from src.supervised.training import train_wine_nn
from src.unsupervised.clustering import (
    make_gmm_posterior,
    make_kmeans_dist,
    make_kmeans_onehot,
)
from src.utils.logger import configure_logger
from src.utils.plotting import plot_f1_comparison

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase6_nn_cluster"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase6_nn_cluster"
PHASE5_METRICS = ARTIFACTS_DIR / "metrics" / "phase5_nn_reduced"

# Frozen K from ADR-002
KMEANS_K = 2
GMM_N = 7


def build_augmented_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build all three augmented variants. Cluster models always fit with SEED_EXPLORE
    so the augmentation is stable across NN seeds.
    """
    return {
        "kmeans_onehot": make_kmeans_onehot(
            X_train, X_val, k=KMEANS_K, seed=SEED_EXPLORE
        ),
        "kmeans_dist": make_kmeans_dist(X_train, X_val, k=KMEANS_K, seed=SEED_EXPLORE),
        "gmm_posterior": make_gmm_posterior(X_train, X_val, n=GMM_N, seed=SEED_EXPLORE),
    }


def load_phase5_raw_median() -> float | None:
    """Load the Phase 5 raw-variant median F1 to use as baseline on the boxplot."""
    cmp = PHASE5_METRICS / "comparison_table.csv"
    if not cmp.exists():
        return None
    df = pd.read_csv(cmp)
    raw_vals = df[df["variant"] == "raw"]["val_f1_final"]
    return float(raw_vals.median()) if not raw_vals.empty else None


def main() -> None:
    run_id = "phase6"
    log = configure_logger(run_id)
    log.info("Phase 6 start — run_id=%s", run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, _, y_train, y_val, _ = load_wine(seed=SEED_EXPLORE)
    log.info("Wine loaded — train=%s  val=%s", X_train.shape, X_val.shape)

    log.info("Building cluster-augmented splits (cluster fit seed=%d)...", SEED_EXPLORE)
    splits = build_augmented_splits(X_train, X_val)
    for v, (Xtr, Xv) in splits.items():
        log.info("  %s — train=%s  val=%s", v, Xtr.shape, Xv.shape)

    comparison_rows = []
    all_histories: list[pd.DataFrame] = []

    variants = ["kmeans_onehot", "kmeans_dist", "gmm_posterior"]
    for variant in variants:
        Xtr, Xv = splits[variant]
        var_dir = OUTPUT_DIR / variant
        var_dir.mkdir(exist_ok=True)

        log.info("── %s (input_dim=%d) ──", variant, Xtr.shape[1])
        for seed in tqdm(SEEDS_REPORT, desc=f"  {variant}", leave=False):
            hist = train_wine_nn(Xtr, y_train, Xv, y_val, seed=seed)
            hist.insert(0, "seed", seed)
            hist.insert(0, "variant", variant)

            hist_path = var_dir / f"seed{seed}.csv"
            hist.to_csv(hist_path, index=False)

            final_f1 = float(hist["val_f1"].iloc[-1])
            best_f1 = float(hist["val_f1"].max())
            log.info(
                "    seed=%d  final_f1=%.4f  best_f1=%.4f", seed, final_f1, best_f1
            )

            comparison_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "input_dim": Xtr.shape[1],
                    "val_f1_final": final_f1,
                    "val_f1_best": best_f1,
                }
            )
            all_histories.append(hist)

    comparison_df = pd.DataFrame(comparison_rows)
    cmp_path = OUTPUT_DIR / "comparison_table.csv"
    comparison_df.to_csv(cmp_path, index=False)
    log.info("Comparison table → %s  (%d rows)", cmp_path, len(comparison_df))
    assert len(comparison_df) == len(variants) * len(SEEDS_REPORT)

    # ── Boxplot — overlay Phase 5 raw baseline ─────────────────────────────────
    raw_median = load_phase5_raw_median()
    if raw_median is None:
        log.warning(
            "Phase 5 comparison_table.csv not found — boxplot will have no baseline"
        )

    f1_plot_df = comparison_df.rename(columns={"val_f1_final": "val_f1"})
    fig_path = plot_f1_comparison(
        f1_plot_df,
        FIGURES_DIR,
        title="Phase 6 — Wine NN Macro-F1: Cluster-Augmented Features (10 seeds)",
        out_name="phase6_f1_boxplot.png",
        baseline_val=raw_median,
        baseline_label=f"Phase 5 Raw baseline = {raw_median:.3f}"
        if raw_median
        else None,
    )
    log.info("F1 boxplot → %s", fig_path)

    # ── F1 summary ─────────────────────────────────────────────────────────────
    log.info("── Phase 6 complete. Val Macro-F1 summary (mean ± std over 10 seeds):")
    for v in variants:
        vdf = comparison_df[comparison_df["variant"] == v]["val_f1_final"]
        log.info(
            "  %s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
            v,
            vdf.mean(),
            vdf.std(),
            vdf.min(),
            vdf.max(),
        )

    if raw_median is not None:
        log.info("  Phase 5 raw baseline median: %.4f", raw_median)

    # ── Metadata JSON ──────────────────────────────────────────────────────────
    meta = {
        "mean_f1": {
            v: round(
                float(
                    comparison_df[comparison_df["variant"] == v]["val_f1_final"].mean()
                ),
                4,
            )
            for v in variants
        },
        "input_dim": {
            v: int(comparison_df[comparison_df["variant"] == v]["input_dim"].iloc[0])
            for v in variants
        },
    }
    meta_dir = ARTIFACTS_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "phase6.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Metadata → %s", meta_path)


if __name__ == "__main__":
    main()
