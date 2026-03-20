# Phase 7 — Requirements (Extra Credit)

**Status:** [ ] OPTIONAL — not started

## What This Phase Must Produce

1. t-SNE 2D embeddings for Wine and Adult.
2. Four scatter plots: labels overlay + cluster assignment overlay per dataset.

## Acceptance Criteria

- SHALL produce 4 PNGs in `artifacts/figures/phase7_tsne/`.
- SHALL NOT save t-SNE embeddings as CSV or use them as NN inputs anywhere.
- SHALL NOT use t-SNE for clustering or model selection.
- SHALL log embedding shape and perplexity to `artifacts/logs/phase7_<ts>.log`.

## Output Locations

| Artifact | Path |
|----------|------|
| Figures | `artifacts/figures/phase7_tsne/` |
| Log | `artifacts/logs/phase7_<ts>.log` |

## Constraint

t-SNE is visualization only. This constraint is absolute — see steering/product.md.
