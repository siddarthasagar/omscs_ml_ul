"""
Phase 5: Wine NN trained on raw and DR-reduced inputs.

For each of 4 variants (raw, pca, ica, rp) × 10 seeds:
  - Fit DR on X_train (seed=42 for all DR fits — DR is deterministic per variant)
  - Reinit model + optimizer with the run seed, train for 20 epochs
  - Save per-epoch history CSV

Produces:
  artifacts/metrics/phase5_nn_reduced/{variant}/seed{seed}.csv  (40 files)
  artifacts/metrics/phase5_nn_reduced/comparison_table.csv      (40 rows)
  artifacts/figures/phase5_nn_reduced/phase5_f1_boxplot.png
  artifacts/figures/phase5_nn_reduced/{variant}_learning_curves.png  (4 files)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SEED_EXPLORE, SEEDS_REPORT
from src.data.wine import load_wine
from src.supervised.training import train_wine_nn
from src.unsupervised.reduction import fit_ica, fit_pca, fit_rp
from src.utils.logger import configure_logger
from src.utils.plotting import plot_f1_comparison, plot_learning_curves

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase5_nn_reduced"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase5_nn_reduced"

# Frozen n_components from Phase 3 design.md
FROZEN_N = {"pca": 8, "ica": 4, "rp": 8}


def build_reduced_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Fit each DR method on X_train (seed=42), transform train + val.
    Returns dict mapping variant name → (X_train_r, X_val_r).
    DR fit seed is always SEED_EXPLORE so the projection is stable across NN seeds.
    """
    splits = {"raw": (X_train, X_val)}

    pca, X_train_pca = fit_pca(X_train, n_components=FROZEN_N["pca"])
    X_val_pca = pca.transform(X_val)
    splits["pca"] = (X_train_pca.astype(np.float32), X_val_pca.astype(np.float32))

    ica, _ = fit_ica(X_train, n_components=FROZEN_N["ica"], seed=SEED_EXPLORE)
    X_train_ica = ica.transform(X_train).astype(np.float32)
    X_val_ica = ica.transform(X_val).astype(np.float32)
    splits["ica"] = (X_train_ica, X_val_ica)

    rp, X_train_rp = fit_rp(X_train, n_components=FROZEN_N["rp"], seed=SEED_EXPLORE)
    X_val_rp = rp.transform(X_val).astype(np.float32)
    splits["rp"] = (X_train_rp.astype(np.float32), X_val_rp)

    return splits


def main() -> None:
    run_id = "phase5"
    log = configure_logger(run_id)
    log.info("Phase 5 start — run_id=%s", run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, _, y_train, y_val, _ = load_wine(seed=SEED_EXPLORE)
    log.info("Wine loaded — train=%s  val=%s", X_train.shape, X_val.shape)

    log.info("Fitting DR methods on X_train (seed=%d)...", SEED_EXPLORE)
    splits = build_reduced_splits(X_train, X_val)
    for v, (Xtr, Xv) in splits.items():
        log.info("  %s — train=%s  val=%s", v, Xtr.shape, Xv.shape)

    comparison_rows = []
    all_histories: list[pd.DataFrame] = []

    variants = ["raw", "pca", "ica", "rp"]
    for variant in variants:
        Xtr, Xv = splits[variant]
        var_dir = OUTPUT_DIR / variant
        var_dir.mkdir(exist_ok=True)

        log.info("── %s (input_dim=%d) ──", variant.upper(), Xtr.shape[1])
        for seed in tqdm(SEEDS_REPORT, desc=f"  {variant}", leave=False):
            hist = train_wine_nn(Xtr, y_train, Xv, y_val, seed=seed)
            hist.insert(0, "seed", seed)
            hist.insert(0, "variant", variant)

            # Save per-run history
            hist_path = var_dir / f"seed{seed}.csv"
            hist.to_csv(hist_path, index=False)

            final_f1 = float(hist["val_f1"].iloc[-1])
            best_f1 = float(hist["val_f1"].max())
            log.info("    seed=%d  final_f1=%.4f  best_f1=%.4f", seed, final_f1, best_f1)

            comparison_rows.append({
                "variant": variant,
                "seed": seed,
                "input_dim": Xtr.shape[1],
                "val_f1_final": final_f1,
                "val_f1_best": best_f1,
            })
            all_histories.append(hist)

    # ── Summary table ──────────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(comparison_rows)
    cmp_path = OUTPUT_DIR / "comparison_table.csv"
    comparison_df.to_csv(cmp_path, index=False)
    log.info("Comparison table → %s  (%d rows)", cmp_path, len(comparison_df))
    assert len(comparison_df) == len(variants) * len(SEEDS_REPORT)

    # ── Figures ────────────────────────────────────────────────────────────────
    f1_plot_df = comparison_df.rename(columns={"val_f1_final": "val_f1"})
    raw_median = float(f1_plot_df[f1_plot_df["variant"] == "raw"]["val_f1"].median())
    fig_path = plot_f1_comparison(
        f1_plot_df,
        FIGURES_DIR,
        title="Phase 5 — Wine NN Macro-F1: Raw vs Reduced Inputs (10 seeds)",
        out_name="phase5_f1_boxplot.png",
        baseline_val=raw_median,
        baseline_label=f"Raw median = {raw_median:.3f}",
    )
    log.info("F1 boxplot → %s", fig_path)

    full_history = pd.concat(all_histories, ignore_index=True)
    fig_path = plot_learning_curves(full_history, FIGURES_DIR)
    log.info("Learning curves → %s", fig_path)

    # ── Gate 3 check ───────────────────────────────────────────────────────────
    log.info("── Gate 3: verifying only input_dim differs across variants ──")
    from src.config import NN_BETAS, NN_LR, NN_MAX_EPOCHS, NN_TRAIN_BATCH_SIZE, NN_WEIGHT_DECAY
    log.info("  lr=%s  betas=%s  wd=%s  batch=%s  epochs=%s",
             NN_LR, NN_BETAS, NN_WEIGHT_DECAY, NN_TRAIN_BATCH_SIZE, NN_MAX_EPOCHS)
    input_dims = comparison_df.groupby("variant")["input_dim"].first().to_dict()
    log.info("  input_dims: %s", input_dims)

    # ── F1 summary ─────────────────────────────────────────────────────────────
    log.info("── Phase 5 complete. Val Macro-F1 summary (mean ± std over 10 seeds):")
    for v in variants:
        vdf = comparison_df[comparison_df["variant"] == v]["val_f1_final"]
        log.info("  %s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                 v.upper(), vdf.mean(), vdf.std(), vdf.min(), vdf.max())


if __name__ == "__main__":
    main()
