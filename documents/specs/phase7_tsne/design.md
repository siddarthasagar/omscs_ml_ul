# Phase 7 — Design (Extra Credit)

**Status:** [ ] OPTIONAL

## Configuration

- `perplexity=30` (sklearn default)
- `n_components=2` (2D visualization)
- `seed=SEED_EXPLORE` (42)
- Input: `X_train` (standardized, same as used in Phase 2)

## Figures

| Figure | Colour | Function |
|--------|--------|---------|
| `wine_tsne_labels.png` | Ground-truth quality class | `plot_tsne(embedding, labels, title, out_path)` |
| `wine_tsne_clusters.png` | Frozen KMeans cluster assignment (k=2) | `plot_tsne(embedding, cluster_labels, title, out_path)` |
| `adult_tsne_labels.png` | Ground-truth income class | `plot_tsne(embedding, labels, title, out_path)` |
| `adult_tsne_clusters.png` | Frozen KMeans cluster assignment (k=8) | `plot_tsne(embedding, cluster_labels, title, out_path)` |

## Purpose

Qualitative visual support for report narrative. Use to explain why class boundaries remain weak and why certain clustering results are locally plausible but globally noisy.

## Risk

t-SNE is easily over-interpreted. Distances between clusters in the 2D embedding are not meaningful — only local neighborhood structure is preserved. Do not make quantitative claims from t-SNE plots.
