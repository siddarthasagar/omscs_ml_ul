# Phase 5 — Design

**Status:** [ ] NOT STARTED

## Architecture (locked from Phase 0 — see steering/tech.md)

`Linear(input_dim, 100) → ReLU → Linear(100, 8)` — no dropout.

| Variant | input_dim |
|---------|-----------|
| raw | 12 |
| pca | TBD (from Phase 3) |
| ica | TBD (from Phase 3) |
| rp | TBD (from Phase 3) |

Only input_dim changes. All other config (Adam, lr=1e-3, batch_size=128, max_epochs=20) is fixed.

## Training Config (locked)

See `steering/tech.md` — NN Config table. Do not modify for baseline runs.

## Gate 3 Check (manual, before writing report)

Confirm in code that all 4 variants use identical optimizer, lr, batch_size, max_epochs, loss function. Only `input_dim` in the first Linear layer differs.

## Figures

| Figure | Content | Function |
|--------|---------|---------|
| `phase5_f1_boxplot.png` | Boxplot of val_macro_f1 across 10 seeds per variant | `plot_f1_comparison` |
| `{variant}_learning_curves.png` | Mean ± std train/val loss + val_f1 across 10 seeds vs epoch | `plot_learning_curves` |

## Comparison Goal

Primary question: does any DR representation improve Wine NN Macro-F1 relative to the raw 12-feature baseline? Secondary: does convergence speed or stability differ across representations?
