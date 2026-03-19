from pathlib import Path

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED_EXPLORE: int = 42
SEEDS_REPORT: list[int] = list(range(42, 52))  # 42–51 inclusive

# ── Wine feature contract (Phase 0 audit: matches OL input_dim=12) ────────────
# 11 physicochemical features + `type` (numeric 0/1, StandardScaled)
# Dropped: `quality` (leakage), `class` (target)
WINE_N_FEATURES: int = 12
WINE_N_CLASSES: int = 8

# ── NN training config (locked from Phase 0 OL audit) ─────────────────────────
NN_LR: float = 1e-3
NN_BETAS: tuple[float, float] = (0.9, 0.999)
NN_WEIGHT_DECAY: float = 0.0
NN_TRAIN_BATCH_SIZE: int = 128
NN_VAL_BATCH_SIZE: int = 256
NN_MAX_EPOCHS: int = 20
NN_HIDDEN_DIM: int = 100

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path("data")
ARTIFACTS_DIR: Path = Path("artifacts")
