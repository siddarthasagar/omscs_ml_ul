"""
Phase 3: Step 2 — Dimensionality reduction on raw data.

Runs PCA (full spectrum), ICA, and Random Projection on X_train for both
Wine and Adult. Saves 6 CSVs + 6 PNGs to phase3_reduction artifacts.

Also generates PCA/ICA component loadings heatmaps saved to
artifacts/figures/analysis/ for use in the report.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, DATA_DIR, SEED_EXPLORE, SEEDS_REPORT
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.unsupervised.reduction import fit_ica, fit_pca, fit_rp, rp_reconstruction_error
from src.utils.logger import configure_logger
from src.utils.plotting import (
    plot_ica_kurtosis,
    plot_ica_loadings,
    plot_pca_loadings,
    plot_pca_variance,
    plot_rp_stability,
)

OUTPUT_DIR = ARTIFACTS_DIR / "metrics" / "phase3_reduction"
FIGURES_DIR = ARTIFACTS_DIR / "figures" / "phase3_reduction"
ANALYSIS_FIGURES_DIR = ARTIFACTS_DIR / "figures" / "analysis"

# RP: use PCA n_components as the target dim; sweep these seeds for stability
RP_SEEDS = SEEDS_REPORT  # 42–51

WINE_FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free SO₂",
    "total SO₂",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "type",
]


def get_adult_feature_names() -> list[str]:
    """Reconstruct Adult feature names (numeric + OHE categorical) from raw CSV."""
    frame = pd.read_csv(DATA_DIR / "adult.csv").drop(columns=["class"])
    cat_cols = list(frame.select_dtypes(include=["object", "string"]).columns)
    num_cols = [c for c in frame.columns if c not in cat_cols]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(frame[cat_cols])
    return num_cols + list(ohe.get_feature_names_out(cat_cols))


def run_dataset(
    name: str,
    X_train: np.ndarray,
    feature_names: list[str],
    log,
) -> dict:
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

    # Label-free selection: first component where cumvar >= 90%
    n_pca = int(np.searchsorted(cumvar, 0.90) + 1)
    log.info(
        "  PCA frozen n_components=%d (cumvar=%.3f at that point)",
        n_pca,
        cumvar[n_pca - 1],
    )

    for i in range(3):
        top = np.argsort(np.abs(pca.components_[i]))[::-1][:3]
        log.info("    PC%d top features: %s", i + 1, [feature_names[j] for j in top])

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

    # Label-free selection: components above the median absolute kurtosis
    abs_kurt = np.abs(kurtosis_array)
    n_ica = int(np.sum(abs_kurt >= np.median(abs_kurt)))
    n_ica = max(n_ica, 2)  # floor at 2
    log.info("  ICA frozen n_components=%d (above-median kurtosis threshold)", n_ica)

    # ── RP stability sweep ────────────────────────────────────────────────────
    log.info(
        "  RP: stability sweep over %d seeds, n_components=%d ...", len(RP_SEEDS), n_pca
    )
    rp_records = []
    for seed in RP_SEEDS:
        rp, _ = fit_rp(X_train, n_components=n_pca, seed=seed)
        err = rp_reconstruction_error(rp, X_train)
        rp_records.append(
            {"seed": seed, "n_components": n_pca, "reconstruction_error": err}
        )
        log.info("    seed=%d  recon_error=%.6f", seed, err)
    rp_df = pd.DataFrame(rp_records)
    rp_path = OUTPUT_DIR / f"{name}_rp_stability.csv"
    rp_df.to_csv(rp_path, index=False)
    log.info("  RP  Saved  → %s", rp_path)

    n_rp = n_pca  # RP uses same target dim as PCA (geometry-preserving compression)
    log.info(
        "  RP  frozen n_components=%d (same as PCA; mean_recon_error=%.6f ± %.6f)",
        n_rp,
        rp_df["reconstruction_error"].mean(),
        rp_df["reconstruction_error"].std(),
    )

    pc1_var_pct = round(float(pca.explained_variance_ratio_[0]) * 100, 1)
    cumvar_at_n_pct = round(float(cumvar[n_pca - 1]) * 100, 1)
    comp_ratio_pct = round(n_pca / n_features * 100)
    comp_ratio_x = round(n_features / n_pca, 1)

    return {
        "pca": n_pca,
        "ica": n_ica,
        "rp": n_rp,
        "pc1_var_pct": pc1_var_pct,
        "cumvar_at_n_pct": cumvar_at_n_pct,
        "comp_ratio_pct": comp_ratio_pct,
        "comp_ratio_x": comp_ratio_x,
        "n_features": n_features,
    }


def main() -> None:
    run_id = "phase3"
    log = configure_logger(run_id)
    log.info("Phase 3 start — run_id=%s", run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Reconstructing Adult feature names ...")
    adult_feature_names = get_adult_feature_names()
    log.info("Adult feature count: %d", len(adult_feature_names))

    datasets = {
        "wine": (load_wine(seed=SEED_EXPLORE)[0], WINE_FEATURE_NAMES),
        "adult": (load_adult(seed=SEED_EXPLORE)[0], adult_feature_names),
    }

    frozen = {}
    for name, (X_train, feature_names) in tqdm(datasets.items(), desc="Datasets"):
        frozen[name] = run_dataset(name, X_train, feature_names, log)

    log.info("── Phase 3 complete. Frozen n_components:")
    for name, vals in frozen.items():
        log.info(
            "   %s: PCA=%d  ICA=%d  RP=%d", name, vals["pca"], vals["ica"], vals["rp"]
        )

    # ── Metadata JSON ──────────────────────────────────────────────────────────
    meta = {
        "frozen_n": {
            ds: {
                "pca": frozen[ds]["pca"],
                "ica": frozen[ds]["ica"],
                "rp": frozen[ds]["rp"],
            }
            for ds in frozen
        },
        "wine": {
            "n_features": frozen["wine"]["n_features"],
            "pca": {
                "pc1_var_pct": frozen["wine"]["pc1_var_pct"],
                "cumvar_at_n_pct": frozen["wine"]["cumvar_at_n_pct"],
                "comp_ratio_pct": frozen["wine"]["comp_ratio_pct"],
                "comp_ratio_x": frozen["wine"]["comp_ratio_x"],
            },
        },
        "adult": {
            "n_features": frozen["adult"]["n_features"],
            "pca": {
                "pc1_var_pct": frozen["adult"]["pc1_var_pct"],
                "comp_ratio_pct": frozen["adult"]["comp_ratio_pct"],
                "comp_ratio_x": frozen["adult"]["comp_ratio_x"],
            },
        },
    }
    meta_dir = ARTIFACTS_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "phase3.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Metadata → %s", meta_path)

    log.info("── Artifacts:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        log.info("   %s", p)
    for fig in visualize(meta_path):
        log.info("   %s", fig)


def visualize(checkpoint: Path) -> list[Path]:
    """Regenerate phase 3 figures from saved CSVs.

    PCA/ICA loadings are re-fitted with frozen n_components (fast, deterministic/seeded).
    No clustering or NN code is executed.
    """
    meta = json.loads(checkpoint.read_text())
    frozen_n = meta["frozen_n"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    adult_feature_names = get_adult_feature_names()
    ds_map = {
        "wine": (load_wine(seed=SEED_EXPLORE)[0], WINE_FEATURE_NAMES),
        "adult": (load_adult(seed=SEED_EXPLORE)[0], adult_feature_names),
    }

    figs: list[Path] = []
    for name, (X_train, feature_names) in ds_map.items():
        n_ica = frozen_n[name]["ica"]

        pca_df = pd.read_csv(OUTPUT_DIR / f"{name}_pca.csv")
        figs.append(plot_pca_variance(pca_df, name, FIGURES_DIR))

        pca, _ = fit_pca(X_train, n_components=None)
        out = ANALYSIS_FIGURES_DIR / f"{name}_pca_loadings.png"
        figs.append(
            plot_pca_loadings(
                pca.components_,
                feature_names,
                n_show=3,
                dataset_name=name.title(),
                out_path=out,
            )
        )

        ica_df = pd.read_csv(OUTPUT_DIR / f"{name}_ica.csv")
        figs.append(plot_ica_kurtosis(ica_df, name, FIGURES_DIR))

        if len(feature_names) <= 30:
            ica_final, _ = fit_ica(X_train, n_components=n_ica, seed=SEED_EXPLORE)
            out = ANALYSIS_FIGURES_DIR / f"{name}_ica_loadings.png"
            figs.append(
                plot_ica_loadings(
                    ica_final.mixing_,
                    feature_names,
                    dataset_name=name.title(),
                    out_path=out,
                )
            )

        rp_df = pd.read_csv(OUTPUT_DIR / f"{name}_rp_stability.csv")
        figs.append(plot_rp_stability(rp_df, name, FIGURES_DIR))

    return figs


if __name__ == "__main__":
    main()
