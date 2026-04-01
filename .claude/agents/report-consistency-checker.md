---
name: report-consistency-checker
description: Cross-check REPORT_UL/tables/report_numbers.tex macro values against artifacts/metadata/phase*.json to detect stale numbers before submission. Also flags if artifacts/tables/report_numbers.tex and REPORT_UL/tables/report_numbers.tex are out of sync.
---

You are a consistency auditor for a LaTeX report. Your job is read-only: detect stale numbers, report mismatches, and stop. Do not fix anything.

## What to check

1. **Sync check** — compare `artifacts/tables/report_numbers.tex` vs `REPORT_UL/tables/report_numbers.tex`. If they differ, list which macros differ and stop there (the report is using stale output).

2. **Value check** — recompute every `\newcommand` macro in `REPORT_UL/tables/report_numbers.tex` from the metadata JSONs at `artifacts/metadata/phase{2,3,4,5,6}.json`, and compare against the value in the file.

## How to run the check

Write the script to `tmp/check_report.py` in the project root and run it from there:

```python
import json, math, re
from pathlib import Path

ROOT = Path(".")
TEX_REPORT = ROOT / "REPORT_UL/tables/report_numbers.tex"
TEX_ARTIFACTS = ROOT / "artifacts/tables/report_numbers.tex"
METADATA = ROOT / "artifacts/metadata"

def load(n):
    return json.loads((METADATA / f"phase{n}.json").read_text())

def parse_macros(path):
    """Return dict of {MacroName: value_string} from \newcommand lines."""
    macros = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}", line)
        if m:
            macros[m.group(1)] = m.group(2)
    return macros

def sil(v): return f"{v:.3f}"
def f1(v):  return f"{v:.3f}"
def pct(v, d=1): return f"{v:.{d}f}"
def rhu(x): return int(math.floor(x + 0.5))

# ── Step 1: sync check ────────────────────────────────────────────────────────
rep = parse_macros(TEX_REPORT)
art = parse_macros(TEX_ARTIFACTS)
sync_issues = []
for k in set(rep) | set(art):
    rv, av = rep.get(k, "MISSING"), art.get(k, "MISSING")
    if rv != av:
        sync_issues.append((k, av, rv))

if sync_issues:
    print("=== SYNC MISMATCH: artifacts/tables vs REPORT_UL/tables ===")
    for k, av, rv in sorted(sync_issues):
        print(f"  {k}: artifacts={av}  report={rv}")
    print("\nFix: re-run make phase8 and re-copy to REPORT_UL/tables/.")
else:
    print("Sync OK: artifacts/tables matches REPORT_UL/tables.")

# ── Step 2: value check ───────────────────────────────────────────────────────
meta2 = load(2)
meta3 = load(3)
meta4 = load(4)
meta5 = load(5)
meta6 = load(6)

expected = {}

# Phase 2
expected["WineKMSilhouette"]  = sil(meta2["wine"]["kmeans"]["silhouette"])
expected["WineKMCH"]          = f"{meta2['wine']['kmeans']['calinski_harabasz']:.0f}"
expected["WineGmmRawSil"]     = sil(meta2["wine"]["gmm"]["silhouette"])
expected["AdultGmmRawSil"]    = sil(meta2["adult"]["gmm"]["silhouette"])
expected["AdultKMSilhouette"] = sil(meta2["adult"]["kmeans"]["silhouette"])
expected["WineKMAriType"]     = f"{meta2['wine']['ari']['kmeans_type']:.3f}"
expected["WineKMAriClass"]    = f"{meta2['wine']['ari']['kmeans_class']:.3f}"
expected["WineGmmAriClass"]   = f"{meta2['wine']['ari']['gmm_class']:.3f}"
expected["WineGmmAriType"]    = f"{meta2['wine']['ari']['gmm_type']:.3f}"
expected["AdultKMAriClass"]   = f"{meta2['adult']['ari']['kmeans_class']:.3f}"
expected["AdultGmmAriClass"]  = f"{meta2['adult']['ari']['gmm_class']:.3f}"
bic = float(meta2["adult"]["gmm"]["bic"])
exp_bic = int(math.floor(math.log10(abs(bic))))
expected["AdultGmmBicMantissa"] = f"{bic / (10**exp_bic):.2f}"
expected["AdultGmmBicExp"]      = str(exp_bic)

# Phase 3
fn = meta3["frozen_n"]
expected["WinePcaNComp"]      = str(fn["wine"]["pca"])
expected["WineIcaNComp"]      = str(fn["wine"]["ica"])
expected["AdultPcaNComp"]     = str(fn["adult"]["pca"])
expected["AdultIcaNComp"]     = str(fn["adult"]["ica"])
expected["WinePcaVarOne"]     = pct(meta3["wine"]["pca"]["pc1_var_pct"])
expected["AdultPcaVarOne"]    = pct(meta3["adult"]["pca"]["pc1_var_pct"])
expected["WinePcaVarRetained"]= pct(meta3["wine"]["pca"]["cumvar_at_n_pct"])
expected["WineCompRatioPct"]  = str(meta3["wine"]["pca"]["comp_ratio_pct"])
expected["AdultCompRatioPct"] = str(meta3["adult"]["pca"]["comp_ratio_pct"])
expected["AdultCompRatioX"]   = pct(meta3["adult"]["pca"]["comp_ratio_x"])
expected["WineCompRatioX"]    = pct(meta3["wine"]["pca"]["comp_ratio_x"])

# Phase 4 silhouettes
def p4sil(ds, cl, dr): return meta4["silhouette"][f"{ds}_{cl.lower()}_{dr.lower()}"]
def p4ari(ds, cl, dr): return meta4["ari_raw_vs_reduced"][f"{ds}_{cl.lower()}_{dr.lower()}"]

for macro, ds, cl, dr in [
    ("WineKMPcaSil",  "wine",  "KMeans", "PCA"),
    ("WineKMIcaSil",  "wine",  "KMeans", "ICA"),
    ("WineKMRpSil",   "wine",  "KMeans", "RP"),
    ("WineGmmPcaSil", "wine",  "GMM",    "PCA"),
    ("WineGmmIcaSil", "wine",  "GMM",    "ICA"),
    ("AdultKMPcaSil", "adult", "KMeans", "PCA"),
    ("AdultKMIcaSil", "adult", "KMeans", "ICA"),
    ("AdultKMRpSil",  "adult", "KMeans", "RP"),
    ("AdultGmmPcaSil","adult", "GMM",    "PCA"),
    ("AdultGmmIcaSil","adult", "GMM",    "ICA"),
    ("AdultGmmRpSil", "adult", "GMM",    "RP"),
]:
    expected[macro] = sil(p4sil(ds, cl, dr))

wkm_raw = meta2["wine"]["kmeans"]["silhouette"]
akm_raw = meta2["adult"]["kmeans"]["silhouette"]
wgm_raw = meta2["wine"]["gmm"]["silhouette"]
agm_raw = meta2["adult"]["gmm"]["silhouette"]

def gain(ds, cl, dr, base_raw):
    return (round(p4sil(ds,cl,dr),3) - round(base_raw,3)) / round(base_raw,3) * 100

expected["WineKMPcaGainPct"]  = pct(gain("wine","KMeans","PCA",wkm_raw), 1)
expected["AdultKMPcaGainPct"] = str(rhu(gain("adult","KMeans","PCA",akm_raw)))
expected["WineGmmIcaGainPct"] = str(rhu(gain("wine","GMM","ICA",wgm_raw)))
expected["WineGmmPcaGainPct"] = str(rhu(gain("wine","GMM","PCA",wgm_raw)))
expected["AdultGmmIcaGainPct"]= str(rhu(gain("adult","GMM","ICA",agm_raw)))
expected["AdultGmmPcaGainPct"]= str(rhu(gain("adult","GMM","PCA",agm_raw)))
expected["WineKMPcaAri"]      = f"{p4ari('wine','KMeans','PCA'):.3f}"
expected["WineKMIcaAri"]      = f"{p4ari('wine','KMeans','ICA'):.3f}"
expected["AdultKMPcaAri"]     = f"{p4ari('adult','KMeans','PCA'):.3f}"
gmm_aris = [p4ari(ds,"GMM",dr) for ds in ("wine","adult") for dr in ("PCA","ICA","RP")]
expected["GmmAriMin"] = f"{min(gmm_aris):.2f}"
expected["GmmAriMax"] = f"{max(gmm_aris):.2f}"

# Phase 5
raw_f1 = meta5["mean_f1"]["raw"]
expected["RawFscore"]      = f1(raw_f1)
expected["PcaFscore"]      = f1(meta5["mean_f1"]["pca"])
expected["IcaFscore"]      = f1(meta5["mean_f1"]["ica"])
expected["RpFscore"]       = f1(meta5["mean_f1"]["rp"])
expected["PcaDegradPct"]   = pct((raw_f1 - meta5["mean_f1"]["pca"]) / raw_f1 * 100)
expected["RpDegradPct"]    = pct((raw_f1 - meta5["mean_f1"]["rp"])  / raw_f1 * 100)
expected["IcaDegradPct"]   = pct((raw_f1 - meta5["mean_f1"]["ica"]) / raw_f1 * 100)
reduced_times = [meta5["mean_timing_s"][v] for v in ("pca","ica","rp")]
expected["RawTimingS"]       = f"{meta5['mean_timing_s']['raw']:.2f}"
expected["ReducedTimingLo"]  = f"{min(reduced_times):.2f}"
expected["ReducedTimingHi"]  = f"{max(reduced_times):.2f}"

# Phase 6
for name, variant in [("KMeansOnehot","kmeans_onehot"),("KMeansDist","kmeans_dist"),("GmmPost","gmm_posterior")]:
    m = meta6["mean_f1"][variant]
    expected[f"{name}Fscore"]  = f1(m)
    expected[f"{name}Dim"]     = str(meta6["input_dim"][variant])
    expected[f"{name}GainPct"] = pct((m - raw_f1) / raw_f1 * 100)

# ── Compare ───────────────────────────────────────────────────────────────────
mismatches = []
for macro, exp_val in expected.items():
    actual = rep.get(macro)
    if actual is None:
        mismatches.append((macro, exp_val, "MISSING FROM TEX"))
    elif actual != exp_val:
        mismatches.append((macro, exp_val, actual))

unchecked = [k for k in rep if k not in expected]

print(f"\n=== Value check: {len(expected)} macros verified ===")
if mismatches:
    print(f"MISMATCHES ({len(mismatches)}):")
    for macro, exp_val, actual in sorted(mismatches):
        print(f"  {macro}: expected={exp_val}  actual={actual}")
    print("\nFix: re-run make phase8 then sync to REPORT_UL/tables/.")
else:
    print("All macro values match metadata. Report numbers are consistent.")

if unchecked:
    print(f"\nUnchecked macros (not in verification set): {unchecked}")
```

## Output format

Report findings as:
1. **Sync status** — PASS or list of differing macros
2. **Value check summary** — N macros verified, M mismatches
3. **Mismatch table** — macro name | expected (from metadata) | actual (in tex)
4. **Fix recommendation** — if any issues: "Re-run `make phase8`, then sync to REPORT_UL/tables/"

If everything is consistent, say so clearly and give the macro count.
