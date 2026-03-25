"""
Phase 7 (Extra Credit): t-SNE visualizations for Wine and Adult.

For each dataset, produces two scatter plots:
  - Coloured by ground-truth class labels
  - Coloured by frozen KMeans cluster assignment (Wine k=2, Adult k=8 — chosen
    by human review of sweep metrics, see FROZEN_K in run_phase_2_k_analysis.py)

t-SNE is for qualitative visual support only. Embeddings are NOT saved as CSV
and are NEVER used as NN inputs. Do not make quantitative claims from these plots.

Produces (4 PNGs):
  artifacts/figures/phase7_tsne/wine_tsne_labels.png
  artifacts/figures/phase7_tsne/wine_tsne_clusters.png
  artifacts/figures/phase7_tsne/adult_tsne_labels.png
  artifacts/figures/phase7_tsne/adult_tsne_clusters.png
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.cluster import KMeans

from src.config import ARTIFACTS_DIR, SEED_EXPLORE
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.unsupervised.reduction import fit_tsne
from src.utils.logger import configure_logger
from src.utils.plotting import plot_tsne

FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase7_tsne"
METADATA = ARTIFACTS_DIR / "metadata"


def _load_frozen_k() -> dict:
    path = METADATA / "phase2.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run 'make phase2' first.")
    fk = json.loads(path.read_text())["frozen_k"]
    return {ds: fk[ds]["kmeans"] for ds in ("wine", "adult")}


FROZEN_K = _load_frozen_k()


def main() -> None:
    run_id = "phase7"
    log = configure_logger(run_id)
    log.info("Phase 7 start — run_id=%s", run_id)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "wine": load_wine(seed=SEED_EXPLORE),
        "adult": load_adult(seed=SEED_EXPLORE),
    }

    for name, splits in datasets.items():
        X_train, _, _, y_train, _, _ = splits
        k = FROZEN_K[name]

        log.info("── %s | shape=%s | k=%d ──", name.upper(), X_train.shape, k)

        # t-SNE embedding (slow for Adult — logged so user knows it's running)
        log.info("  Fitting t-SNE (perplexity=30, seed=%d)...", SEED_EXPLORE)
        embedding = fit_tsne(X_train, seed=SEED_EXPLORE)
        log.info("  Embedding shape: %s", embedding.shape)

        # Plot 1: ground-truth labels
        out_path = FIGURES_DIR / f"{name}_tsne_labels.png"
        plot_tsne(
            embedding,
            y_train,
            title=f"{name.title()} — t-SNE coloured by ground-truth class",
            out_path=out_path,
        )
        log.info("  Labels plot → %s", out_path)

        # Plot 2: KMeans cluster assignment
        log.info("  Fitting KMeans(k=%d) for cluster colours...", k)
        km = KMeans(n_clusters=k, random_state=SEED_EXPLORE, n_init="auto")
        cluster_labels = km.fit_predict(X_train)

        out_path = FIGURES_DIR / f"{name}_tsne_clusters.png"
        plot_tsne(
            embedding,
            cluster_labels,
            title=f"{name.title()} — t-SNE coloured by KMeans cluster (k={k})",
            out_path=out_path,
        )
        log.info("  Cluster plot → %s", out_path)

    log.info("Phase 7 complete — 4 PNGs in %s", FIGURES_DIR)
    log.info(
        "Reminder: t-SNE is qualitative only. Do not cite inter-cluster distances."
    )


if __name__ == "__main__":
    main()
