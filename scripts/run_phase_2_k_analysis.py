"""
Phase 2 K-selection analysis and finalisation.

Step 1 — Reads sweep CSVs from run_phase_2_raw_cluster.py and prints a
         human-readable decision-support report to stdout and to
         artifacts/analysis/phase2_k_selection.md.

Step 2 — Using FROZEN_K defined below (set by human after reviewing the
         report), computes ARI at the chosen K values, writes
         ari_results.csv, and emits artifacts/metadata/phase2.json so
         that phases 3–8 can read frozen decisions without hardcoding.

Workflow:
  1. make phase2          → sweep CSVs + plots
  2. make analysis        → this script prints the report
  3. Review report + plots, update FROZEN_K below
  4. make analysis        → re-run to finalise ari_results.csv + phase2.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from src.config import ARTIFACTS_DIR, SEED_EXPLORE
from src.data.adult import load_adult
from src.data.wine import load_wine
from src.utils.logger import configure_logger

METRICS = ARTIFACTS_DIR / "metrics" / "phase2_clustering"
OUT_DIR = ARTIFACTS_DIR / "analysis"
METADATA = ARTIFACTS_DIR / "metadata"

# ── Human decision — set after reviewing 'make analysis' report ───────────────
#
# These values were chosen by inspecting the sweep metric plots and the
# analysis report produced by this script. Do not change without re-running
# 'make phase2' + 'make analysis' and reviewing the new report.
#
# Wine KMeans = 2
#   Silhouette, Calinski-Harabasz, Davies-Bouldin, and inertia elbow all peak
#   or trough at k=2. Four-metric consensus — the strongest possible signal.
#
# Wine GMM = 7
#   BIC reaches a local minimum at n=7 (60841), with a clear uptick at n=8
#   (+605). The global BIC minimum is n=13, but the curve is noisy beyond n=7
#   and the improvement is marginal. AIC excluded — it monotonically decreases
#   by design and always favours more components.
#
# Adult KMeans = 8
#   Silhouette peaks at k=8 (0.114). CH has a secondary local peak at k=8
#   (2583). DB continues improving past k=8 but without a clear elbow.
#   Adult silhouette scores are uniformly low (0.09–0.11) due to curse of
#   dimensionality in the 104-feature OHE space — k=8 is the best available
#   single-metric optimum and consistent with a reasonable cluster granularity.
#
# Adult GMM = 7
#   BIC reaches a local minimum at n=7 (-8,290,938) with an uptick at n=8
#   (+66,753). Consistent with Wine GMM choice. The global minimum (n=19) is
#   far beyond where the BIC curve has meaningfully flattened.
#
FROZEN_K = {
    "wine": {"kmeans": 2, "gmm": 7},
    "adult": {"kmeans": 8, "gmm": 7},
}


# ── Analysis helpers ──────────────────────────────────────────────────────────


def _local_minima(values: np.ndarray) -> list[int]:
    return [
        i
        for i in range(1, len(values) - 1)
        if values[i] < values[i - 1] and values[i] <= values[i + 1]
    ]


def _elbow_index(values: np.ndarray) -> int:
    n = len(values)
    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)
    x_n = (x - x[0]) / (x[-1] - x[0])
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return int(np.argmax(np.abs(y_n - x_n)))


def analyse_kmeans(ds: str) -> list[str]:
    df = pd.read_csv(METRICS / f"{ds}_kmeans.csv")
    lines = []
    best_sil = int(df.loc[df["silhouette"].idxmax(), "k"])
    best_ch = int(df.loc[df["calinski_harabasz"].idxmax(), "k"])
    best_db = int(df.loc[df["davies_bouldin"].idxmin(), "k"])
    elbow_k = int(df["k"].iloc[_elbow_index(df["inertia"].values)])

    votes: dict[int, list[str]] = {}
    for k, metric in [
        (best_sil, "silhouette↑"),
        (best_ch, "CH↑"),
        (best_db, "DB↓"),
        (elbow_k, "inertia-elbow"),
    ]:
        votes.setdefault(k, []).append(metric)
    consensus_k = max(votes, key=lambda k: len(votes[k]))

    lines.append(f"### {ds.capitalize()} — KMeans")
    lines.append("")
    lines.append(
        f"{'k':>3}  {'silhouette':>10}  {'CH':>10}  {'DB':>8}  {'inertia':>12}"
    )
    lines.append(f"{'─' * 3}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 12}")
    for _, row in df.iterrows():
        k = int(row["k"])
        markers = []
        if k == best_sil:
            markers.append("←sil")
        if k == best_ch:
            markers.append("←CH")
        if k == best_db:
            markers.append("←DB")
        if k == elbow_k:
            markers.append("←elbow")
        tag = "  " + " ".join(markers) if markers else ""
        lines.append(
            f"{k:>3}  {row['silhouette']:>10.4f}  {row['calinski_harabasz']:>10.1f}"
            f"  {row['davies_bouldin']:>8.4f}  {row['inertia']:>12.1f}{tag}"
        )
    lines += [
        "",
        "**Metric optima:**",
        f"  - Silhouette peak : k = {best_sil}",
        f"  - CH peak         : k = {best_ch}",
        f"  - DB trough       : k = {best_db}",
        f"  - Inertia elbow   : k = {elbow_k}",
        "",
        f"**Consensus candidate: k = {consensus_k}**",
        f"  Supported by: {', '.join(votes[consensus_k])}",
    ]
    if len(votes[consensus_k]) == 4:
        lines.append("  ✓ All four metrics agree — strong selection.")
    elif len(votes[consensus_k]) >= 2:
        lines.append("  ~ Partial agreement — review plot before deciding.")
    else:
        lines.append("  ✗ No consensus — manual inspection required.")
    lines.append("")
    return lines


def analyse_gmm(ds: str) -> list[str]:
    df = pd.read_csv(METRICS / f"{ds}_gmm.csv")
    lines = []
    bic = df["bic"].values
    ns = df["n_components"].values
    global_bic_n = int(ns[np.argmin(bic)])
    local_min_idxs = _local_minima(bic)
    first_local_n = int(ns[local_min_idxs[0]]) if local_min_idxs else global_bic_n
    elbow_n = int(ns[_elbow_index(bic)])

    lines.append(f"### {ds.capitalize()} — GMM")
    lines.append("")
    lines.append(
        f"{'n':>3}  {'BIC':>14}  {'ΔBIC':>10}  {'AIC':>14}  {'silhouette':>10}"
    )
    lines.append(f"{'─' * 3}  {'─' * 14}  {'─' * 10}  {'─' * 14}  {'─' * 10}")
    prev_bic = None
    for _, row in df.iterrows():
        n = int(row["n_components"])
        delta = (
            f"{row['bic'] - prev_bic:>+10.1f}" if prev_bic is not None else f"{'—':>10}"
        )
        prev_bic = row["bic"]
        markers = []
        if n == global_bic_n:
            markers.append("←BIC-min")
        if n == first_local_n != global_bic_n:
            markers.append("←BIC-local-min")
        if n == elbow_n:
            markers.append("←BIC-elbow")
        tag = "  " + " ".join(markers) if markers else ""
        lines.append(
            f"{n:>3}  {row['bic']:>14.1f}  {delta}  {row['aic']:>14.1f}"
            f"  {row['silhouette']:>10.4f}{tag}"
        )
    lines += [
        "",
        "**BIC analysis:**",
        f"  - Global BIC minimum : n = {global_bic_n}  (check ΔBIC — may over-fit)",
    ]
    for idx in local_min_idxs:
        lines.append(
            f"  - Local BIC minimum  : n = {int(ns[idx])}  (BIC = {bic[idx]:.1f})"
        )
    if not local_min_idxs:
        lines.append("  - No local minimum — curve is monotonically decreasing")
    lines += [
        f"  - BIC elbow          : n = {elbow_n}",
        "",
        "**Note:** AIC monotonically decreases by design — do not use as primary criterion.",
        "",
    ]
    if local_min_idxs:
        lines.append(
            f"**Recommended candidate: n = {first_local_n}** (first local BIC minimum)"
        )
    else:
        lines.append(
            f"**Recommended candidate: n = {elbow_n}** (BIC elbow — no local minimum found)"
        )
    lines.append(
        "  Manual review required: confirm the local minimum is stable, not noise."
    )
    lines.append("")
    return lines


# ── ARI + metadata emission ───────────────────────────────────────────────────


def compute_ari_and_emit(log) -> None:
    wine_splits = load_wine(seed=SEED_EXPLORE)
    adult_splits = load_adult(seed=SEED_EXPLORE)
    datasets = {
        "wine": (wine_splits[0], wine_splits[3]),
        "adult": (adult_splits[0], adult_splits[3]),
    }

    ari_records = []
    for name, (X_train, y_true) in datasets.items():
        k_km = FROZEN_K[name]["kmeans"]
        k_gm = FROZEN_K[name]["gmm"]

        km_labels = KMeans(
            n_clusters=k_km, random_state=SEED_EXPLORE, n_init="auto"
        ).fit_predict(X_train)
        gm_labels = GaussianMixture(
            n_components=k_gm,
            random_state=SEED_EXPLORE,
            covariance_type="diag",
            reg_covar=1e-3,
        ).fit_predict(X_train)

        for clusterer, labels, k in [
            ("KMeans", km_labels, k_km),
            ("GMM", gm_labels, k_gm),
        ]:
            ari = adjusted_rand_score(y_true, labels)
            log.info("  %s %s K=%d ARI vs class=%.4f", name.upper(), clusterer, k, ari)
            ari_records.append(
                {
                    "dataset": name,
                    "clusterer": clusterer,
                    "K": k,
                    "label": "class",
                    "ARI": round(ari, 4),
                }
            )

        if name == "wine":
            type_labels = (X_train[:, 11] > 0).astype(int)
            for clusterer, labels, k in [
                ("KMeans", km_labels, k_km),
                ("GMM", gm_labels, k_gm),
            ]:
                ari_type = adjusted_rand_score(type_labels, labels)
                log.info("  Wine %s K=%d ARI vs type=%.4f", clusterer, k, ari_type)
                ari_records.append(
                    {
                        "dataset": "wine",
                        "clusterer": clusterer,
                        "K": k,
                        "label": "type",
                        "ARI": round(ari_type, 4),
                    }
                )

    ari_df = pd.DataFrame(ari_records)
    ari_path = METRICS / "ari_results.csv"
    ari_df.to_csv(ari_path, index=False)
    log.info("ARI results → %s", ari_path)

    # ── phase2.json ────────────────────────────────────────────────────────────
    def _km_row(ds):
        df = pd.read_csv(METRICS / f"{ds}_kmeans.csv")
        r = df[df["k"] == FROZEN_K[ds]["kmeans"]].iloc[0]
        return {
            "silhouette": round(float(r["silhouette"]), 4),
            "calinski_harabasz": round(float(r["calinski_harabasz"]), 1),
            "davies_bouldin": round(float(r["davies_bouldin"]), 4),
        }

    def _gm_row(ds):
        df = pd.read_csv(METRICS / f"{ds}_gmm.csv")
        r = df[df["n_components"] == FROZEN_K[ds]["gmm"]].iloc[0]
        return {
            "silhouette": round(float(r["silhouette"]), 4),
            "bic": round(float(r["bic"]), 2),
            "aic": round(float(r["aic"]), 2),
        }

    def _ari(ds, cl, lbl):
        r = ari_df[
            (ari_df["dataset"] == ds)
            & (ari_df["clusterer"] == cl)
            & (ari_df["label"] == lbl)
        ]
        return round(float(r["ARI"].iloc[0]), 4) if not r.empty else None

    meta = {
        "frozen_k": FROZEN_K,
        "wine": {
            "kmeans": _km_row("wine"),
            "gmm": _gm_row("wine"),
            "ari": {
                "kmeans_type": _ari("wine", "KMeans", "type"),
                "kmeans_class": _ari("wine", "KMeans", "class"),
                "gmm_type": _ari("wine", "GMM", "type"),
                "gmm_class": _ari("wine", "GMM", "class"),
            },
        },
        "adult": {
            "kmeans": _km_row("adult"),
            "gmm": _gm_row("adult"),
            "ari": {
                "kmeans_class": _ari("adult", "KMeans", "class"),
                "gmm_class": _ari("adult", "GMM", "class"),
            },
        },
    }
    METADATA.mkdir(parents=True, exist_ok=True)
    meta_path = METADATA / "phase2.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Metadata → %s", meta_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    log = configure_logger("phase2_analysis")

    for csv in ["wine_kmeans.csv", "adult_kmeans.csv", "wine_gmm.csv", "adult_gmm.csv"]:
        if not (METRICS / csv).exists():
            print(f"ERROR: {METRICS / csv} not found — run 'make phase2' first.")
            sys.exit(1)

    report: list[str] = [
        "# Phase 2 K-Selection Analysis",
        "",
        "Decision support for K selection. Review alongside sweep plots in",
        "`artifacts/figures/phase2_clustering/` before committing to K values.",
        "",
        "---",
        "",
        "## KMeans",
        "",
    ]
    for ds in ("wine", "adult"):
        report.extend(analyse_kmeans(ds))

    report += ["---", "", "## GMM", ""]
    for ds in ("wine", "adult"):
        report.extend(analyse_gmm(ds))

    report += [
        "---",
        "",
        "## How to use this report",
        "",
        "1. **KMeans:** prefer k where the most metrics agree (consensus candidate).",
        "   Strong = all 4 agree. Weak = only 1–2 agree — inspect the plot.",
        "2. **GMM:** prefer the first local BIC minimum over the global minimum.",
        "   Global minimum often over-fits. Check ΔBIC: if small after the local",
        "   minimum the curve has flattened. AIC always decreases — ignore it.",
        "3. Update FROZEN_K in this file (run_phase_2_k_analysis.py).",
        "4. Record rationale as a comment above FROZEN_K in this file.",
        "5. Re-run 'make analysis' to finalise ari_results.csv + phase2.json.",
        "",
    ]

    text = "\n".join(report)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase2_k_selection.md"
    out_path.write_text(text)
    print(f"\n── Saved → {out_path}")

    log.info("── Computing ARI and emitting phase2.json at FROZEN_K=%s", FROZEN_K)
    compute_ari_and_emit(log)


if __name__ == "__main__":
    main()
