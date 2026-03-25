"""
Phase 8: Generate LaTeX table bodies and prose number macros from experiment artifacts.

Produces 6 .tex files in artifacts/tables/:
  tab_phase2_clustering.tex  — raw clustering metrics at frozen K
  tab_phase3_reduction.tex   — frozen n_components per dataset+method
  tab_phase4_silhouette.tex  — silhouette in reduced spaces vs raw baseline
  tab_phase5_nn.tex          — NN Macro-F1 on DR-reduced inputs
  tab_phase6_nn.tex          — NN Macro-F1 with cluster-derived features
  report_numbers.tex         — \\newcommand macros for every number cited in prose

The report \\input{tables/report_numbers} in its preamble so all inline numbers
are computed from the same CSVs as the tables, eliminating copy-paste drift.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import ARTIFACTS_DIR
from src.utils.logger import configure_logger

METRICS = ARTIFACTS_DIR / "metrics"
OUT_DIR = ARTIFACTS_DIR / "tables"
METADATA = ARTIFACTS_DIR / "metadata"


def _load_metadata(n: int) -> dict:
    """Load artifacts/metadata/phaseN.json written by the phase script."""
    path = METADATA / f"phase{n}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run 'make phase{n}' first to generate metadata."
        )
    return json.loads(path.read_text())


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
    meta2 = _load_metadata(2)
    frozen_k = meta2["frozen_k"]

    rows = []
    # Header
    rows.append("Dataset & Algorithm & $K$ & Silhouette & CH / BIC & DB / AIC \\\\")
    rows.append("\\midrule")

    for dataset in ("wine", "adult"):
        km = meta2[dataset]["kmeans"]
        k = frozen_k[dataset]["kmeans"]
        rows.append(
            f"{dataset.title()} & K-Means & {k} "
            f"& {km['silhouette']:.3f} "
            f"& {km['calinski_harabasz']:.0f} "
            f"& {km['davies_bouldin']:.3f} \\\\"
        )
        gm = meta2[dataset]["gmm"]
        n = frozen_k[dataset]["gmm"]
        rows.append(
            f"{dataset.title()} & GMM (EM) & {n} "
            f"& {gm['silhouette']:.3f} "
            f"& {gm['bic']:.0f} "
            f"& {gm['aic']:.0f} \\\\"
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
    frozen_n = _load_metadata(3)["frozen_n"]
    CRITERIA = {
        "PCA": "Cumulative variance $\\geq 90\\%$",
        "ICA": "Above-median $|$kurtosis$|$ (floor 2)",
        "RP": "= PCA target dim",
    }

    rows = ["Dataset & Method & $d$ & Selection criterion \\\\", "\\midrule"]
    for dataset in ("wine", "adult"):
        for method in ("PCA", "ICA", "RP"):
            n = frozen_n[dataset][method.lower()]
            crit = CRITERIA[method]
            rows.append(f"{dataset.title()} & {method} & {n} & {crit} \\\\")

    tex = _tex_table("llrl", rows)
    return _save(tex, "tab_phase3_reduction.tex", log)


# ── Table 3: Phase 4 silhouette heatmap ───────────────────────────────────────


def emit_phase4_table(log) -> Path:
    meta2 = _load_metadata(2)
    meta4 = _load_metadata(4)

    # Raw baselines from phase2 metadata
    raw_sil: dict[tuple, float] = {}
    for dataset in ("wine", "adult"):
        raw_sil[(dataset, "KMeans")] = meta2[dataset]["kmeans"]["silhouette"]
        raw_sil[(dataset, "GMM")] = meta2[dataset]["gmm"]["silhouette"]

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
            key = f"{ds}_{cl.lower()}_{dr.lower()}"
            vals[dr] = meta4["silhouette"].get(key, float("nan"))

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


# ── Report number macros ───────────────────────────────────────────────────────


def emit_report_numbers(log) -> Path:
    """
    Compute every number cited in report prose from per-phase metadata JSONs.
    Writes artifacts/tables/report_numbers.tex with \\newcommand definitions.
    All values come from artifacts/metadata/phaseN.json written by each phase script.
    """
    import math

    meta2 = _load_metadata(2)
    meta3 = _load_metadata(3)
    meta4 = _load_metadata(4)
    meta5 = _load_metadata(5)
    meta6 = _load_metadata(6)

    lines = [
        "% AUTO-GENERATED by run_phase_8_report_tables.py — do NOT edit manually.",
        "% Re-run: make phase8",
        "% Every number cited in report prose is defined here as a \\newcommand.",
        "% Usage in text: $\\WineKMAriType$ instead of hardcoded 0.981",
        "%",
    ]

    def mac(name: str, value: str, comment: str = "") -> None:
        c = f"  % {comment}" if comment else ""
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}{c}")

    def pct(val: float, decimals: int = 1) -> str:
        return f"{val:.{decimals}f}"

    def sil(val: float) -> str:
        return f"{val:.3f}"

    def f1(val: float) -> str:
        return f"{val:.3f}"

    def _rhu(x: float) -> int:
        """Round-half-up: avoids Python banker's rounding (462.5 → 463, not 462)."""
        return int(math.floor(x + 0.5))

    # ── Phase 2 — raw clustering ───────────────────────────────────────────────
    lines.append("% Phase 2 — raw clustering")
    mac(
        "WineKMSilhouette",
        sil(meta2["wine"]["kmeans"]["silhouette"]),
        "wine KMeans K=2 silhouette",
    )
    mac(
        "WineKMCH",
        f"{meta2['wine']['kmeans']['calinski_harabasz']:.0f}",
        "wine KMeans K=2 CH",
    )
    mac(
        "WineGmmRawSil",
        sil(meta2["wine"]["gmm"]["silhouette"]),
        "wine GMM K=7 raw silhouette",
    )
    mac(
        "AdultGmmRawSil",
        sil(meta2["adult"]["gmm"]["silhouette"]),
        "adult GMM K=7 raw silhouette",
    )
    mac(
        "AdultKMSilhouette",
        sil(meta2["adult"]["kmeans"]["silhouette"]),
        "adult KMeans K=8 silhouette",
    )

    mac(
        "WineKMAriType",
        f"{meta2['wine']['ari']['kmeans_type']:.3f}",
        "wine KMeans ARI vs type",
    )
    mac(
        "WineKMAriClass",
        f"{meta2['wine']['ari']['kmeans_class']:.3f}",
        "wine KMeans ARI vs quality",
    )
    mac(
        "WineGmmAriClass",
        f"{meta2['wine']['ari']['gmm_class']:.3f}",
        "wine GMM ARI vs quality",
    )
    mac(
        "WineGmmAriType",
        f"{meta2['wine']['ari']['gmm_type']:.3f}",
        "wine GMM ARI vs type",
    )
    mac(
        "AdultKMAriClass",
        f"{meta2['adult']['ari']['kmeans_class']:.3f}",
        "adult KMeans ARI vs income",
    )
    mac(
        "AdultGmmAriClass",
        f"{meta2['adult']['ari']['gmm_class']:.3f}",
        "adult GMM ARI vs income",
    )

    bic = float(meta2["adult"]["gmm"]["bic"])
    exp = int(math.floor(math.log10(abs(bic))))
    mac("AdultGmmBicMantissa", f"{bic / (10**exp):.2f}", "mantissa of adult GMM BIC")
    mac("AdultGmmBicExp", str(exp), "exponent of adult GMM BIC")

    # ── Phase 3 — dimensionality reduction ────────────────────────────────────
    lines.append("% Phase 3 — dimensionality reduction")
    fn = meta3["frozen_n"]
    mac("WinePcaNComp", str(fn["wine"]["pca"]), "wine PCA frozen n_components")
    mac("WineIcaNComp", str(fn["wine"]["ica"]), "wine ICA frozen n_components")
    mac("AdultPcaNComp", str(fn["adult"]["pca"]), "adult PCA frozen n_components")
    mac("AdultIcaNComp", str(fn["adult"]["ica"]), "adult ICA frozen n_components")

    mac(
        "WinePcaVarOne",
        pct(meta3["wine"]["pca"]["pc1_var_pct"]),
        "wine PC1 explained variance %",
    )
    mac(
        "AdultPcaVarOne",
        pct(meta3["adult"]["pca"]["pc1_var_pct"]),
        "adult PC1 explained variance %",
    )
    mac(
        "WinePcaVarRetained",
        pct(meta3["wine"]["pca"]["cumvar_at_n_pct"]),
        "wine cumvar at n_pca",
    )
    mac(
        "WineCompRatioPct",
        str(meta3["wine"]["pca"]["comp_ratio_pct"]),
        "wine: % of dims retained by PCA",
    )
    mac(
        "AdultCompRatioPct",
        str(meta3["adult"]["pca"]["comp_ratio_pct"]),
        "adult: % of dims retained by PCA",
    )
    mac(
        "AdultCompRatioX",
        pct(meta3["adult"]["pca"]["comp_ratio_x"]),
        "adult compression ratio (fold)",
    )
    mac(
        "WineCompRatioX",
        pct(meta3["wine"]["pca"]["comp_ratio_x"]),
        "wine compression ratio (fold)",
    )

    # ── Phase 4 — clustering in reduced spaces ────────────────────────────────
    lines.append("% Phase 4 — clustering in reduced spaces")

    def p4sil(ds, cl, dr):
        return meta4["silhouette"][f"{ds}_{cl.lower()}_{dr.lower()}"]

    def p4ari(ds, cl, dr):
        return meta4["ari_raw_vs_reduced"][f"{ds}_{cl.lower()}_{dr.lower()}"]

    wkm_raw = meta2["wine"]["kmeans"]["silhouette"]
    akm_raw = meta2["adult"]["kmeans"]["silhouette"]
    wgm_raw = meta2["wine"]["gmm"]["silhouette"]
    agm_raw = meta2["adult"]["gmm"]["silhouette"]

    for macro, ds, cl, dr in [
        ("WineKMPcaSil", "wine", "KMeans", "PCA"),
        ("WineKMIcaSil", "wine", "KMeans", "ICA"),
        ("WineKMRpSil", "wine", "KMeans", "RP"),
        ("WineGmmPcaSil", "wine", "GMM", "PCA"),
        ("WineGmmIcaSil", "wine", "GMM", "ICA"),
        ("AdultKMPcaSil", "adult", "KMeans", "PCA"),
        ("AdultKMIcaSil", "adult", "KMeans", "ICA"),
        ("AdultKMRpSil", "adult", "KMeans", "RP"),
        ("AdultGmmPcaSil", "adult", "GMM", "PCA"),
        ("AdultGmmIcaSil", "adult", "GMM", "ICA"),
        ("AdultGmmRpSil", "adult", "GMM", "RP"),
    ]:
        mac(macro, sil(p4sil(ds, cl, dr)), f"{ds} {cl} {dr} silhouette")

    # Gains — computed from 3-decimal rounded values (table precision) with round-half-up
    def _gain(ds, cl, dr, base_raw):
        return (
            (round(p4sil(ds, cl, dr), 3) - round(base_raw, 3))
            / round(base_raw, 3)
            * 100
        )

    mac(
        "WineKMPcaGainPct",
        pct(_gain("wine", "KMeans", "PCA", wkm_raw), 1),
        "wine KMeans: % sil gain PCA vs raw",
    )
    mac(
        "AdultKMPcaGainPct",
        str(_rhu(_gain("adult", "KMeans", "PCA", akm_raw))),
        "adult KMeans: % sil gain PCA vs raw",
    )
    mac(
        "WineGmmIcaGainPct",
        str(_rhu(_gain("wine", "GMM", "ICA", wgm_raw))),
        "wine GMM: % sil gain ICA vs raw",
    )
    mac(
        "WineGmmPcaGainPct",
        str(_rhu(_gain("wine", "GMM", "PCA", wgm_raw))),
        "wine GMM: % sil gain PCA vs raw",
    )
    mac(
        "AdultGmmIcaGainPct",
        str(_rhu(_gain("adult", "GMM", "ICA", agm_raw))),
        "adult GMM: % sil gain ICA vs raw",
    )
    mac(
        "AdultGmmPcaGainPct",
        str(_rhu(_gain("adult", "GMM", "PCA", agm_raw))),
        "adult GMM: % sil gain PCA vs raw",
    )

    mac(
        "WineKMPcaAri",
        f"{p4ari('wine', 'KMeans', 'PCA'):.3f}",
        "wine KMeans+PCA raw-vs-reduced ARI",
    )
    mac(
        "WineKMIcaAri",
        f"{p4ari('wine', 'KMeans', 'ICA'):.3f}",
        "wine KMeans+ICA raw-vs-reduced ARI",
    )
    mac(
        "AdultKMPcaAri",
        f"{p4ari('adult', 'KMeans', 'PCA'):.3f}",
        "adult KMeans+PCA raw-vs-reduced ARI",
    )

    gmm_aris = [
        p4ari(ds, "GMM", dr) for ds in ("wine", "adult") for dr in ("PCA", "ICA", "RP")
    ]
    mac(
        "GmmAriMin",
        f"{min(gmm_aris):.2f}",
        "min GMM raw-vs-reduced ARI across all combos",
    )
    mac(
        "GmmAriMax",
        f"{max(gmm_aris):.2f}",
        "max GMM raw-vs-reduced ARI across all combos",
    )

    # ── Phase 5 — NN on reduced inputs ────────────────────────────────────────
    lines.append("% Phase 5 — NN on DR-reduced inputs")
    raw_f1 = meta5["mean_f1"]["raw"]
    mac("RawFscore", f1(raw_f1), "raw variant mean val F1")
    mac("PcaFscore", f1(meta5["mean_f1"]["pca"]), "PCA variant mean val F1")
    mac("IcaFscore", f1(meta5["mean_f1"]["ica"]), "ICA variant mean val F1")
    mac("RpFscore", f1(meta5["mean_f1"]["rp"]), "RP variant mean val F1")

    mac(
        "PcaDegradPct",
        pct((raw_f1 - meta5["mean_f1"]["pca"]) / raw_f1 * 100),
        "PCA F1 degradation vs raw %",
    )
    mac(
        "RpDegradPct",
        pct((raw_f1 - meta5["mean_f1"]["rp"]) / raw_f1 * 100),
        "RP F1 degradation vs raw %",
    )
    mac(
        "IcaDegradPct",
        pct((raw_f1 - meta5["mean_f1"]["ica"]) / raw_f1 * 100),
        "ICA F1 degradation vs raw %",
    )

    reduced_times = [meta5["mean_timing_s"][v] for v in ("pca", "ica", "rp")]
    mac(
        "RawTimingS",
        f"{meta5['mean_timing_s']['raw']:.2f}",
        "raw mean training time seconds",
    )
    mac(
        "ReducedTimingLo",
        f"{min(reduced_times):.2f}",
        "min reduced-variant mean training time",
    )
    mac(
        "ReducedTimingHi",
        f"{max(reduced_times):.2f}",
        "max reduced-variant mean training time",
    )

    # ── Phase 6 — NN with cluster features ────────────────────────────────────
    lines.append("% Phase 6 — NN with cluster-derived features")
    for name, variant in [
        ("KMeansOnehot", "kmeans_onehot"),
        ("KMeansDist", "kmeans_dist"),
        ("GmmPost", "gmm_posterior"),
    ]:
        m = meta6["mean_f1"][variant]
        mac(f"{name}Fscore", f1(m), f"{variant} mean F1")
        mac(f"{name}Dim", str(meta6["input_dim"][variant]), f"{variant} input dim")
        mac(
            f"{name}GainPct",
            pct((m - raw_f1) / raw_f1 * 100),
            f"{variant} gain vs raw %",
        )

    # ── Write file ─────────────────────────────────────────────────────────────
    path = OUT_DIR / "report_numbers.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(
        "  → %s  (%d macros)",
        path,
        sum(1 for line in lines if line.startswith("\\newcommand")),
    )
    return path


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
    emit_report_numbers(log)

    log.info("Phase 8 complete — 5 tables + report_numbers.tex in %s", OUT_DIR)
    log.info(
        "Usage: \\input{tables/report_numbers} in preamble, then use macros in prose."
    )


if __name__ == "__main__":
    main()
