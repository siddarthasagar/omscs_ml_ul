"""
Phase 3: Step 2 — Dimensionality reduction on raw data.

Runs PCA (full spectrum), ICA, and Random Projection on X_train for both
Wine and Adult. Saves 6 CSVs + 6 PNGs to phase3_reduction artifacts.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SEED_EXPLORE, SEEDS_REPORT, WINE_N_FEATURES
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.unsupervised.reduction import fit_ica, fit_pca, fit_rp, rp_reconstruction_error
from src.utils.logger import configure_logger
from src.utils.plotting import plot_ica_kurtosis, plot_pca_variance, plot_rp_stability

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase3_reduction"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase3_reduction"

# RP: use PCA n_components as the target dim; sweep these seeds for stability
RP_SEEDS = SEEDS_REPORT  # 42–51


def run_dataset(name: str, X_train: np.ndarray, log) -> dict:
    """Run PCA, ICA, RP on one dataset. Returns dict of frozen n_components."""
    n_features = X_train.shape[1]
    log.info("── %s | shape=%s ──", name.upper(), X_train.shape)

    # ── PCA (full spectrum) ────────────────────────────────────────────────────
    log.info("  PCA: fitting full spectrum (n_components=%d) ...", n_features)
    pca, _ = fit_pca(X_train, n_components=None)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(
        {
            "component": np.arange(1, n_features + 1),
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_variance": cumvar,
        }
    )
    pca_path = OUTPUT_DIR / f"{name}_pca.csv"
    pca_df.to_csv(pca_path, index=False)
    log.info("  PCA Saved  → %s", pca_path)
    fig_path = plot_pca_variance(pca_df, name, FIGURES_DIR)
    log.info("  PCA Figure → %s", fig_path)

    # Label-free selection: first component where cumvar >= 90%
    n_pca = int(np.searchsorted(cumvar, 0.90) + 1)
    log.info("  PCA frozen n_components=%d (cumvar=%.3f at that point)", n_pca, cumvar[n_pca - 1])

    # ── ICA ───────────────────────────────────────────────────────────────────
    log.info("  ICA: fitting n_components=%d ...", n_pca)
    ica, kurtosis_array = fit_ica(X_train, n_components=n_pca, seed=SEED_EXPLORE)
    ica_df = pd.DataFrame(
        {
            "component": np.arange(1, n_pca + 1),
            "kurtosis": kurtosis_array,
        }
    )
    ica_path = OUTPUT_DIR / f"{name}_ica.csv"
    ica_df.to_csv(ica_path, index=False)
    log.info("  ICA Saved  → %s", ica_path)
    fig_path = plot_ica_kurtosis(ica_df, name, FIGURES_DIR)
    log.info("  ICA Figure → %s", fig_path)

    # Label-free selection: components above the median absolute kurtosis
    abs_kurt = np.abs(kurtosis_array)
    n_ica = int(np.sum(abs_kurt >= np.median(abs_kurt)))
    n_ica = max(n_ica, 2)  # floor at 2
    log.info("  ICA frozen n_components=%d (above-median kurtosis threshold)", n_ica)

    # ── RP stability sweep ────────────────────────────────────────────────────
    log.info("  RP: stability sweep over %d seeds, n_components=%d ...", len(RP_SEEDS), n_pca)
    rp_records = []
    for seed in RP_SEEDS:
        rp, _ = fit_rp(X_train, n_components=n_pca, seed=seed)
        err = rp_reconstruction_error(rp, X_train)
        rp_records.append({"seed": seed, "n_components": n_pca, "reconstruction_error": err})
        log.info("    seed=%d  recon_error=%.6f", seed, err)
    rp_df = pd.DataFrame(rp_records)
    rp_path = OUTPUT_DIR / f"{name}_rp_stability.csv"
    rp_df.to_csv(rp_path, index=False)
    log.info("  RP  Saved  → %s", rp_path)
    fig_path = plot_rp_stability(rp_df, name, FIGURES_DIR)
    log.info("  RP  Figure → %s", fig_path)

    n_rp = n_pca  # RP uses same target dim as PCA (geometry-preserving compression)
    log.info(
        "  RP  frozen n_components=%d (same as PCA; mean_recon_error=%.6f ± %.6f)",
        n_rp,
        rp_df["reconstruction_error"].mean(),
        rp_df["reconstruction_error"].std(),
    )

    return {"pca": n_pca, "ica": n_ica, "rp": n_rp}


def main() -> None:
    run_id = f"phase3_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    log = configure_logger(run_id)
    log.info("Phase 3 start — run_id=%s", run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "wine": load_wine(seed=SEED_EXPLORE)[0],
        "adult": load_adult(seed=SEED_EXPLORE)[0],
    }

    frozen = {}
    for name, X_train in tqdm(datasets.items(), desc="Datasets"):
        frozen[name] = run_dataset(name, X_train, log)

    log.info("── Phase 3 complete. Frozen n_components:")
    for name, vals in frozen.items():
        log.info("   %s: PCA=%d  ICA=%d  RP=%d", name, vals["pca"], vals["ica"], vals["rp"])

    log.info("── Artifacts:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        log.info("   %s", p)
    for p in sorted(FIGURES_DIR.glob("*.png")):
        log.info("   %s", p)


if __name__ == "__main__":
    main()
