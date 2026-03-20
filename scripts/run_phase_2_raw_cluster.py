"""
Phase 2: Step 1 — Clustering on raw data.

Sweeps K-Means and GMM over k/n_components in range(2, 21) on X_train for
both Wine and Adult. Saves 4 CSVs + 4 PNGs to phase2_clustering artifacts.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SEED_EXPLORE
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.unsupervised.clustering import run_gmm_sweep, run_kmeans_sweep
from src.utils.logger import configure_logger
from src.utils.plotting import plot_gmm_sweep, plot_kmeans_sweep

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase2_clustering"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase2_clustering"


def main() -> None:
    run_id = f"phase2_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    log = configure_logger(run_id)
    log.info("Phase 2 start — run_id=%s", run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "wine": load_wine(seed=SEED_EXPLORE)[0],
        "adult": load_adult(seed=SEED_EXPLORE)[0],
    }

    sweep_range = range(2, 21)

    for name, X_train in tqdm(datasets.items(), desc="Datasets"):
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

    log.info("── Phase 2 complete. Artifacts:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        log.info("   %s", p)
    for p in sorted(FIGURES_DIR.glob("*.png")):
        log.info("   %s", p)


if __name__ == "__main__":
    main()
