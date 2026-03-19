"""
Phase 2: Step 1 — Clustering on raw data.

Sweeps K-Means and GMM over k/n_components in range(2, 21) on X_train for
both Wine and Adult. Saves 4 CSVs to artifacts/metrics/phase2_clustering/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.config import ARTIFACTS_DIR, SEED_EXPLORE
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.unsupervised.clustering import run_gmm_sweep, run_kmeans_sweep

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase2_clustering"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "wine": load_wine(seed=SEED_EXPLORE)[0],
        "adult": load_adult(seed=SEED_EXPLORE)[0],
    }

    sweep_range = range(2, 21)

    for name, X_train in tqdm(datasets.items(), desc="Datasets"):
        print(f"\n── {name.upper()} | shape={X_train.shape} ──")

        print("  K-Means sweep...")
        kmeans_df = run_kmeans_sweep(X_train, k_range=sweep_range, seed=SEED_EXPLORE)
        kmeans_path = OUTPUT_DIR / f"{name}_kmeans.csv"
        kmeans_df.to_csv(kmeans_path, index=False)
        print(f"  Saved → {kmeans_path}")

        print("  GMM sweep...")
        gmm_df = run_gmm_sweep(X_train, n_range=sweep_range, seed=SEED_EXPLORE)
        gmm_path = OUTPUT_DIR / f"{name}_gmm.csv"
        gmm_df.to_csv(gmm_path, index=False)
        print(f"  Saved → {gmm_path}")

    print("\n── Phase 2 complete. Artifacts:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"   {p}")


if __name__ == "__main__":
    main()
