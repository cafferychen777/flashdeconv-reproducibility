"""Build Supplementary Data 1 (benchmark results workbook) from final result CSVs.

Usage:
    .venv/bin/python validation/figures/supp/make_supplementary_data_1.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "rerun_final"
SP = RES / "benchmarks" / "spotless"
LI = RES / "benchmarks" / "li2023" / "expected"
OUT = ROOT / "paper" / "Supplementary_Data_1_final.xlsx"
CONFIG = "final_default"
MAX_ROWS = 200_000

sheets = []  # (name, description, sources, dataframe)
skipped = []


def rel(p):
    return str(Path(p).relative_to(ROOT))


def read(p):
    return pd.read_csv(p)


def final(df):
    if "config" in df.columns:
        df = df[df["config"] == CONFIG]
    return df.reset_index(drop=True)


def add(name, desc, sources, df):
    assert len(name) <= 31, name
    # TACCO was a reserve run and is not part of this paper's comparisons.
    if "method" in df.columns:
        df = df[~df["method"].astype(str).str.contains("tacco", case=False)].reset_index(drop=True)
    if len(df) > MAX_ROWS:
        skipped.append((name, f"{len(df)} rows", ", ".join(rel(s) for s in sources)))
        return
    sheets.append((name, desc, [rel(s) for s in sources], df))


# 1. Spotless silver per data set (13 methods x 54 data sets x 4 metrics)
f = SP / "silver_per_dataset_final_vs_competitors.csv"
df = read(f)
methods = [c for c in df.columns if c not in ("tissue", "pattern", "metric")]
methods = ["FlashDeconv"] + [m for m in methods if m != "FlashDeconv"]
df = df[["metric", "tissue", "pattern"] + methods].sort_values(["metric", "tissue", "pattern"])
add("Spotless_silver_per_dataset",
    "Spotless silver standard: per-data-set metric (corr = Pearson, rmse, jsd, aupr) for FlashDeconv "
    "(final_default) and 12 competing methods; one column per method, 54 data sets x 4 metrics.",
    [f], df.reset_index(drop=True))

# 2. Paired Wilcoxon tests
f = SP / "silver_paired_wilcoxon_final.csv"
add("Spotless_silver_wilcoxon",
    "Paired two-sided Wilcoxon signed-rank tests, FlashDeconv (final_default) vs each competitor across "
    "54 silver data sets, per metric.", [f], final(read(f)))

# 3. Per-cell-type silver
f = SP / "fd_per_celltype_silver.csv.gz"
df = final(read(f)).drop(columns=["version"], errors="ignore")
add("Spotless_silver_per_celltype",
    "FlashDeconv (final_default) per-cell-type metrics on each silver data set, with abundance category "
    "(rare / moderate / abundant).", [f], df)
f = SP / "silver_per_celltype_by_abundance.csv"
add("Spotless_silver_by_abundance",
    "FlashDeconv (final_default) per-cell-type metrics averaged by abundance category (n = cell-type x "
    "data-set pairs).", [f], final(read(f)))

# 4. Gold standard, FlashDeconv + competitors per field of view
f1, f2 = SP / "fd_aggregate_gold.csv", SP / "gold_competitors_recomputed.csv"
fd = final(read(f1))
fd["field_of_view"] = fd["tissue"].str.extract(r"(fov\d+)$")[0].fillna(fd["benchmark"])
fd["method"] = "FlashDeconv"
comp = read(f2).rename(columns={"tissue": "field_of_view"})
metric_cols = ["corr", "rmse", "jsd", "aupr", "precision", "sensitivity", "f1",
               "specificity", "accuracy", "balanced_accuracy"]
gold = pd.concat([fd[["benchmark", "field_of_view", "method"] + metric_cols],
                  comp[["benchmark", "field_of_view", "method"] + metric_cols]], ignore_index=True)
gold = gold.sort_values(["benchmark", "field_of_view", "method"]).reset_index(drop=True)
add("Spotless_gold",
    "Spotless gold standard (seqFISH+ cortex/SVZ and OB, 7 FOVs each; STARmap VISp): metrics per field of "
    "view and method, FlashDeconv (final_default) and 12 competitors recomputed with the same metric code.",
    [f1, f2], gold)

# 5. Liver
f = SP / "liver_case_study.csv"
add("Spotless_liver",
    "Liver case study (mouse Visium JB01-JB04): JSD and portal/central AUPR per slide, plus mean predicted "
    "proportion per cell type (prop_*), FlashDeconv final_default.", [f], final(read(f)))
f = SP / "liver_stability.csv"
add("Spotless_liver_stability",
    "Liver reference stability: JSD between FlashDeconv predictions obtained with two different "
    "single-cell references (exVivo / inVivo / nuclei) on the same slide.", [f], final(read(f)))

# 6. Melanoma
f = SP / "melanoma_fixed.csv"
add("Spotless_melanoma",
    "Melanoma case study: JSD to expected composition and predicted melanocytic / T-cell proportions per "
    "sample, FlashDeconv package defaults (final_default) and Pearson-residual preprocessing "
    "(final_default_pearson).", [f], read(f)[lambda d: d.config.isin(["final_default", "final_default_pearson"])])
f = SP / "melanoma_grid_expected.csv"
add("Spotless_melanoma_grid",
    "Melanoma parameter sensitivity grid (n_hvg, markers per type, lambda, rho, preprocessing, gene "
    "weighting = expected): JSD and melanocytic proportion.", [f], read(f))

# 7. Li et al. 2023
rows = []
srcs = []
for res in (100, 50, 20):
    fs = LI / f"merfish_{res}_suppdata1_vs_published.csv"
    fj = LI / f"merfish_{res}_section_jsd_vs_published.csv"
    a = read(fs)
    b = read(fj).rename(columns={"rank": "rank_section_JSD"})
    m = a.merge(b, on="method", how="outer")
    fp = LI / f"merfish_{res}_pooled_vs_published.csv"
    if fp.exists():
        p = read(fp).rename(columns={"JSD": "pooled_JSD", "total_RMSE": "pooled_total_RMSE"})
        m = m.merge(p, on="method", how="outer")
        srcs.append(fp)
    m.insert(0, "dataset", f"MERFISH {res} um")
    rows.append(m)
    srcs += [fs, fj]
fq = LI / "seqfish10000_vs_published.csv"
q = read(fq).rename(columns={"JSD": "pooled_JSD", "rank_JSD": "rank_pooled_JSD"})
q.insert(0, "dataset", "seqFISH+ 10000 genes")
rows.append(q)
srcs.append(fq)
li = pd.concat(rows, ignore_index=True)
add("Li2023_methods",
    "Li et al. 2023 benchmark: FlashDeconv vs published method results. MERFISH at 100/50/20 um "
    "(total_RMSE and mean_typeJSD as in Li et al. Supp. Data 1; mean section JSD; pooled JSD at 100 um) "
    "and seqFISH+ (pooled JSD, total RMSE). Ranks computed among all listed methods.", srcs, li)

rows = []
srcs = []
for res in (100, 50, 20):
    fb = LI / f"merfish_{res}_flashdeconv_by_section.csv"
    b = read(fb).rename(columns={"Unnamed: 0": "section_bregma"})
    b.insert(0, "resolution_um", res)
    rows.append(b)
    srcs.append(fb)
add("Li2023_FD_by_section",
    "FlashDeconv per-MERFISH-section metrics (section identified by Bregma coordinate) at 100/50/20 um.",
    srcs, pd.concat(rows, ignore_index=True))
f = LI / "merfish_flashdeconv_summary.csv"
add("Li2023_FD_summary",
    "FlashDeconv pooled MERFISH metrics per resolution, including per-type RMSE and JSD and runtime.",
    [f], read(f))

# 8. Xenium CRC pseudo-VHD
f = RES / "benchmarks" / "c2" / "c2_standard_metrics_final.csv"
add("Xenium_CRC_pseudoVHD",
    "Xenium CRC pseudo-Visium HD benchmark: metrics per resolution (2/8/16 um), method, mode and evaluation "
    "set (all bins, doublet/UMI subsets).", [f], read(f))
f = RES / "benchmarks" / "c2" / "c2_supp_table.csv"
if f.exists():
    add("Xenium_CRC_supp_table", "Xenium CRC pseudo-Visium HD summary table.", [f], read(f))
else:
    skipped.append(("Xenium_CRC_supp_table", "file absent at build time", rel(f)))

# 9. Runtime / memory
f = RES / "benchmarks" / "c1" / "c1_runtime_table_final.csv"
add("Runtime_memory",
    "Runtime and peak memory per method and number of spots (median fit time in s; peak RSS in GB).",
    [f], read(f))

# 10. Gene weighting
W = RES / "weighting"
add("Weighting_spotless_acc",
    "Gene-weighting ablation on Spotless silver: per-data-set accuracy for each weighting variant and "
    "depth fraction (frac).", [W / "spotless_acc.csv"], read(W / "spotless_acc.csv"))
add("Weighting_spotless_stats",
    "Gene-weighting ablation on Spotless silver: paired statistics (median difference, bootstrap CI, "
    "win fraction, rank-biserial, Wilcoxon p) for EXP_LEV vs other variants.",
    [W / "stats_spotless.csv"], read(W / "stats_spotless.csv"))
add("Weighting_xenium_acc",
    "Gene-weighting ablation on Xenium CRC pseudo-Visium HD: accuracy per resolution and weighting variant.",
    [W / "xenium_acc.csv"], read(W / "xenium_acc.csv"))
add("Weighting_xenium_stats",
    "Gene-weighting ablation on Xenium CRC: paired statistics across cell types for EXP_LEV vs other "
    "variants.", [W / "stats_xenium.csv"], read(W / "stats_xenium.csv"))

# 11. Laplacian ablation
L = RES / "benchmarks" / "laplacian" / "final_default"
add("Laplacian_ablation_paired",
    "Spatial Laplacian ablation (final_default, lambda auto vs none): paired per-cell-type metrics and deltas.",
    [L / "laplacian_ablation_paired.csv"], read(L / "laplacian_ablation_paired.csv"))
add("Laplacian_ablation_celltype",
    "Spatial Laplacian ablation: per-cell-type metrics per condition (auto lambda / no spatial).",
    [L / "laplacian_ablation_per_celltype.csv"], read(L / "laplacian_ablation_per_celltype.csv"))

# 12. Marker scoring
M = RES / "benchmarks" / "marker_scoring" / "final_default"
add("Marker_scoring",
    "FlashDeconv vs marker-based scoring baselines: per-cell-type metrics per tissue and method.",
    [M / "marker_scoring_comparison.csv"], read(M / "marker_scoring_comparison.csv"))

# 13. Reference diagnostic
R = RES / "refdiag"
add("Refdiag_removal_auroc",
    "Reference-completeness diagnostic on Spotless: after removing one cell type from the reference, "
    "AUROC/AUPRC of the diagnostic score for spots containing the removed type (thr = abundance threshold "
    "defining positive spots).", [R / "spotless_removal_auroc.csv"], read(R / "spotless_removal_auroc.csv"))
add("Refdiag_complete_ref_flags",
    "Reference diagnostic flag rate when the reference is complete (false-positive rate).",
    [R / "spotless_complete_reference_flag_rate.csv"],
    read(R / "spotless_complete_reference_flag_rate.csv"))
add("Refdiag_intestine_regions",
    "Reference diagnostic on intestine Visium HD: flag rate per histological region and reference "
    "(default settings).", [R / "intestine_region_flags_default.csv"],
    read(R / "intestine_region_flags_default.csv"))

# README
readme = pd.DataFrame(
    [{"sheet": n, "n_rows": len(d), "n_columns": d.shape[1], "content": desc, "source_files": "; ".join(s)}
     for n, desc, s, d in sheets]
    + [{"sheet": n, "n_rows": None, "n_columns": None, "content": f"NOT INCLUDED ({why})", "source_files": s}
       for n, why, s in skipped])
header = pd.DataFrame([{
    "sheet": "README",
    "content": "Supplementary Data 1. FlashDeconv benchmark results (package v0.2.0, configuration "
               "final_default unless stated). Source paths are relative to the repository root.",
}])
readme = pd.concat([header, readme], ignore_index=True)

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    readme.to_excel(xw, sheet_name="README", index=False)
    for n, _, _, d in sheets:
        d.to_excel(xw, sheet_name=n, index=False)
    for ws in xw.book.worksheets:
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:50])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60 if ws.title != "README" else 100)

print(f"Wrote {OUT}")
