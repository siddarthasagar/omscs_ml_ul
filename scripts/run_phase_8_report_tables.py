"""
Phase 8: Generate LaTeX table bodies from experiment artifacts.

Produces 5 .tex files in artifacts/tables/ — each contains only a tabular
environment (no preamble) suitable for \\input{} in the Overleaf report.

Tables:
  tab_phase2_clustering.tex  — raw clustering metrics at frozen K
  tab_phase3_reduction.tex   — frozen n_components per dataset+method
  tab_phase4_silhouette.tex  — silhouette in reduced spaces vs raw baseline
  tab_phase5_nn.tex          — NN Macro-F1 on DR-reduced inputs
  tab_phase6_nn.tex          — NN Macro-F1 with cluster-derived features
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import ARTIFACTS_DIR
from src.utils.logger import configure_logger

METRICS = ARTIFACTS_DIR / "metrics"
OUT_DIR = ARTIFACTS_DIR / "tables"

FROZEN_K = {"wine": {"kmeans": 2, "gmm": 7}, "adult": {"kmeans": 8, "gmm": 7}}
FROZEN_N = {
    "wine": {"PCA": 8, "ICA": 4, "RP": 8},
    "adult": {"PCA": 22, "ICA": 11, "RP": 22},
}


# ── LaTeX helpers ──────────────────────────────────────────────────────────────


def _bf(val: str) -> str:
    return f"\\textbf{{{val}}}"


def _tex_table(header: str, rows: list[str], caption_label: str = "") -> str:
    """Wrap rows in a tabular. header is the column-format string e.g. 'llrrr'."""
    body = "\n".join(rows)
    return (
        f"\\begin{{tabular}}{{{header}}}\n"
        f"\\toprule\n"
        f"{body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
    )


def _save(tex: str, name: str, log) -> Path:
    path = OUT_DIR / name
    path.write_text(tex, encoding="utf-8")
    log.info("  → %s", path)
    return path


# ── Table 1: Phase 2 raw clustering ───────────────────────────────────────────


def emit_phase2_table(log) -> Path:
    rows = []
    # Header
    rows.append("Dataset & Algorithm & $K$ & Silhouette & CH / BIC & DB / AIC \\\\")
    rows.append("\\midrule")

    for dataset in ("wine", "adult"):
        # KMeans
        df = pd.read_csv(METRICS / "phase2_clustering" / f"{dataset}_kmeans.csv")
        k = FROZEN_K[dataset]["kmeans"]
        r = df[df["k"] == k].iloc[0]
        rows.append(
            f"{dataset.title()} & K-Means & {k} "
            f"& {r['silhouette']:.3f} "
            f"& {r['calinski_harabasz']:.0f} "
            f"& {r['davies_bouldin']:.3f} \\\\"
        )
        # GMM
        df = pd.read_csv(METRICS / "phase2_clustering" / f"{dataset}_gmm.csv")
        n = FROZEN_K[dataset]["gmm"]
        r = df[df["n_components"] == n].iloc[0]
        rows.append(
            f"{dataset.title()} & GMM (EM) & {n} "
            f"& {r['silhouette']:.3f} "
            f"& {r['bic']:.0f} "
            f"& {r['aic']:.0f} \\\\"
        )

    # Note row
    rows.append("\\midrule")
    rows.append(
        "\\multicolumn{6}{l}{\\footnotesize "
        "CH = Calinski-Harabasz (\\textuparrow); "
        "DB = Davies-Bouldin (\\textdownarrow); "
        "BIC/AIC (\\textdownarrow)} \\\\"
    )

    tex = _tex_table("llrrrr", rows)
    return _save(tex, "tab_phase2_clustering.tex", log)


# ── Table 2: Phase 3 frozen n_components ──────────────────────────────────────


def emit_phase3_table(log) -> Path:
    CRITERIA = {
        "wine": {
            "PCA": "Cumulative variance $\\geq 90\\%$",
            "ICA": "Above-median $|$kurtosis$|$ (floor 2)",
            "RP": "= PCA target dim",
        },
        "adult": {
            "PCA": "Cumulative variance $\\geq 90\\%$",
            "ICA": "Above-median $|$kurtosis$|$ (floor 2)",
            "RP": "= PCA target dim",
        },
    }

    rows = ["Dataset & Method & $d$ & Selection criterion \\\\", "\\midrule"]
    for dataset in ("wine", "adult"):
        for method in ("PCA", "ICA", "RP"):
            n = FROZEN_N[dataset][method]
            crit = CRITERIA[dataset][method]
            rows.append(f"{dataset.title()} & {method} & {n} & {crit} \\\\")

    tex = _tex_table("llrl", rows)
    return _save(tex, "tab_phase3_reduction.tex", log)


# ── Table 3: Phase 4 silhouette heatmap ───────────────────────────────────────


def emit_phase4_table(log) -> Path:
    p4 = pd.read_csv(METRICS / "phase4_clustering" / "summary_table.csv")

    # Build raw baselines from Phase 2
    raw_sil: dict[tuple, float] = {}
    for dataset in ("wine", "adult"):
        km_df = pd.read_csv(METRICS / "phase2_clustering" / f"{dataset}_kmeans.csv")
        k = FROZEN_K[dataset]["kmeans"]
        raw_sil[(dataset, "KMeans")] = float(
            km_df[km_df["k"] == k]["silhouette"].iloc[0]
        )
        gmm_df = pd.read_csv(METRICS / "phase2_clustering" / f"{dataset}_gmm.csv")
        n = FROZEN_K[dataset]["gmm"]
        raw_sil[(dataset, "GMM")] = float(
            gmm_df[gmm_df["n_components"] == n]["silhouette"].iloc[0]
        )

    combos = [
        ("wine", "KMeans"),
        ("wine", "GMM"),
        ("adult", "KMeans"),
        ("adult", "GMM"),
    ]
    dr_methods = ["PCA", "ICA", "RP"]

    rows = [
        "Dataset \\& Clusterer & Raw & PCA & ICA & RP \\\\",
        "\\midrule",
    ]
    for ds, cl in combos:
        vals: dict[str, float] = {"Raw": raw_sil[(ds, cl)]}
        for dr in dr_methods:
            mask = (
                (p4["dataset"] == ds)
                & (p4["clusterer"] == cl)
                & (p4["dr_method"] == dr)
            )
            row = p4[mask]
            vals[dr] = (
                float(row["silhouette"].iloc[0]) if not row.empty else float("nan")
            )

        best_key = max(vals, key=lambda k: vals[k])
        cells = []
        for key in ["Raw", "PCA", "ICA", "RP"]:
            fmt = f"{vals[key]:.3f}"
            cells.append(_bf(fmt) if key == best_key else fmt)

        label = f"{ds.title()} {cl}"
        rows.append(f"{label} & {' & '.join(cells)} \\\\")

    rows.append(
        "\\multicolumn{5}{l}{\\footnotesize Bold = best silhouette per row.} \\\\"
    )
    tex = _tex_table("lrrrr", rows)
    return _save(tex, "tab_phase4_silhouette.tex", log)


# ── Table 4: Phase 5 NN on DR-reduced inputs ──────────────────────────────────


def emit_phase5_table(log) -> Path:
    df = pd.read_csv(METRICS / "phase5_nn_reduced" / "comparison_table.csv")
    order = ["raw", "pca", "ica", "rp"]
    labels = {"raw": "Raw (12d)", "pca": "PCA (8d)", "ica": "ICA (4d)", "rp": "RP (8d)"}

    rows = [
        "Variant & $d_{\\mathrm{in}}$ & Mean F1 & Std & Min & Max \\\\",
        "\\midrule",
    ]
    summaries = (
        df.groupby("variant")["val_f1_final"]
        .agg(["mean", "std", "min", "max"])
        .reindex(order)
    )
    best_mean = summaries["mean"].max()

    for v in order:
        s = summaries.loc[v]
        dim = int(df[df["variant"] == v]["input_dim"].iloc[0])
        mean_str = f"{s['mean']:.4f}"
        row = (
            f"{labels[v]} & {dim} "
            f"& {_bf(mean_str) if s['mean'] == best_mean else mean_str} "
            f"& {s['std']:.4f} "
            f"& {s['min']:.4f} "
            f"& {s['max']:.4f} \\\\"
        )
        rows.append(row)

    rows.append(
        "\\multicolumn{6}{l}{\\footnotesize "
        "10 seeds (42--51). Bold = best mean F1.} \\\\"
    )
    tex = _tex_table("lrrrrl", rows)
    return _save(tex, "tab_phase5_nn.tex", log)


# ── Table 5: Phase 6 NN with cluster features ─────────────────────────────────


def emit_phase6_table(log) -> Path:
    p6 = pd.read_csv(METRICS / "phase6_nn_cluster" / "comparison_table.csv")
    p5 = pd.read_csv(METRICS / "phase5_nn_reduced" / "comparison_table.csv")

    raw_mean = float(p5[p5["variant"] == "raw"]["val_f1_final"].mean())
    raw_dim = int(p5[p5["variant"] == "raw"]["input_dim"].iloc[0])

    order = ["raw_baseline", "kmeans_onehot", "kmeans_dist", "gmm_posterior"]
    labels = {
        "raw_baseline": "Raw baseline (12d)",
        "kmeans_onehot": "KMeans one-hot (14d)",
        "kmeans_dist": "KMeans distances (14d)",
        "gmm_posterior": "GMM posterior (19d)",
    }

    rows = [
        "Variant & $d_{\\mathrm{in}}$ & Mean F1 & Std & Min & Max \\\\",
        "\\midrule",
    ]

    # Compute all means first so we can bold above-baseline
    all_means: dict[str, float] = {"raw_baseline": raw_mean}
    for v in ["kmeans_onehot", "kmeans_dist", "gmm_posterior"]:
        all_means[v] = float(p6[p6["variant"] == v]["val_f1_final"].mean())

    for v in order:
        if v == "raw_baseline":
            raw_std = float(p5[p5["variant"] == "raw"]["val_f1_final"].std())
            raw_min = float(p5[p5["variant"] == "raw"]["val_f1_final"].min())
            raw_max = float(p5[p5["variant"] == "raw"]["val_f1_final"].max())
            mean_str = f"{raw_mean:.4f}"
            rows.append(
                f"{labels[v]} & {raw_dim} "
                f"& {mean_str} "
                f"& {raw_std:.4f} "
                f"& {raw_min:.4f} "
                f"& {raw_max:.4f} \\\\"
            )
            rows.append("\\midrule")
        else:
            s = p6[p6["variant"] == v]["val_f1_final"].agg(
                ["mean", "std", "min", "max"]
            )
            dim = int(p6[p6["variant"] == v]["input_dim"].iloc[0])
            mean_str = f"{s['mean']:.4f}"
            beats = all_means[v] > raw_mean
            rows.append(
                f"{labels[v]} & {dim} "
                f"& {_bf(mean_str) if beats else mean_str} "
                f"& {s['std']:.4f} "
                f"& {s['min']:.4f} "
                f"& {s['max']:.4f} \\\\"
            )

    rows.append(
        "\\multicolumn{6}{l}{\\footnotesize "
        "10 seeds (42--51). Bold = beats raw baseline.} \\\\"
    )
    tex = _tex_table("lrrrrl", rows)
    return _save(tex, "tab_phase6_nn.tex", log)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    run_id = "phase8"
    log = configure_logger(run_id)
    log.info("Phase 8 start — generating report tables")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emit_phase2_table(log)
    emit_phase3_table(log)
    emit_phase4_table(log)
    emit_phase5_table(log)
    emit_phase6_table(log)

    log.info("Phase 8 complete — 5 LaTeX tables in %s", OUT_DIR)
    log.info("Usage in report: \\input{tab_phase2_clustering} etc.")


if __name__ == "__main__":
    main()
