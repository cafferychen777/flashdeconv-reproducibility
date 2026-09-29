"""Build Supplementary Data 1 (benchmark results workbook) from final result CSVs.

Usage:
    .venv/bin/python validation/figures/supp/make_supplementary_data_1.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "rerun_final"
SP = RES / "benchmarks" / "spotless"
LI = RES / "benchmarks" / "li2023" / "expected"
OUT = ROOT / "paper" / "Supplementary_Data_1.xlsx"
CONFIG = "final_default"
# Spotless silver standards use lambda=0 (the pseudo-spots have no spatial layout)
SILVER_CONFIG = "final_default_lam0"
ER = ROOT / "results" / "editor_revision"
MAX_ROWS = 200_000

sheets = []  # (name, description, sources, dataframe)
skipped = []


def rel(p):
    return str(Path(p).relative_to(ROOT))


def read(p):
    return pd.read_csv(p)


def read_li(p):
    d = read(p)
    d["method"] = d["method"].map(method_name)
    return d


def final(df):
    if "config" in df.columns:
        df = df[df["config"] == CONFIG]
    return df.reset_index(drop=True)


def silver(df):
    if "config" in df.columns:
        df = df[df["config"] == SILVER_CONFIG]
    return df.reset_index(drop=True)


# Readable labels for internal configuration names
LABELS = {
    "final_default_lam0_realxy": "lambda=0, measured coordinates",
    "final_default_realxy": "default, measured coordinates",
    "final_default_pearson": "Pearson residuals",
    "final_default_lam0": "lambda=0",
    "final_default_auto": "default",
    "final_default_l0": "lambda=0",
    "final_default": "default",
    "lam0": "lambda=0",
    "flashdeconv_final_coldcache": "FlashDeconv, first call (JIT compilation)",
    "flashdeconv_final": "FlashDeconv",
    "card": "CARD", "cell2location": "Cell2location", "rctd": "RCTD",
}


# Canonical method names (as spelled in the manuscript)
METHODS = {
    "flashdeconv": "FlashDeconv", "card": "CARD", "cell2location": "Cell2location", "rctd": "RCTD",
    "destvi": "DestVI", "dstg": "DSTG", "music": "MuSiC", "nnls": "NNLS", "seurat": "Seurat",
    "spatialdwls": "SpatialDWLS", "spotlight": "SPOTlight", "stereoscope": "Stereoscope",
    "stride": "STRIDE", "tangram": "Tangram", "spaotsc": "SpaOTsc", "markerscoring": "Marker scoring",
    "markerscoring_maxgap": "Marker scoring (max-gap markers)",
    "markerscoring_wilcoxon": "Marker scoring (Wilcoxon markers)",
}
METHOD_COLS = ("method", "comparator")
# Gene-weighting schemes (internal codes -> names used in the manuscript)
WEIGHTS = {"EXP_LEV": "leverage", "UNIFORM": "equal", "VAR_REF": "variance"}
# Columns that are internal bookkeeping and carry no result
DROP_COLS = ("version", "maxdiff_vs_package", "jsd_paper")
RENAME_COLS = {
    # Pearson correlation computed on all reference types, before restricting the prediction
    # to the ground-truth types and renormalising (the 'corr' column is after renormalisation)
    "corr_unrenormalised": "corr_before_renormalization",
    "mean_lev": "mean_leverage", "flash_mean": "flashdeconv_mean", "flash_better": "flashdeconv_better",
}


def method_name(v):
    if not isinstance(v, str):
        return v
    return METHODS.get(v.lower(), v)


def weight_name(v):
    if not isinstance(v, str):
        return v
    for k, w in WEIGHTS.items():
        v = v.replace(k, w)
    return v


def relabel(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: LABELS.get(v, v) if isinstance(v, str) else v)
    return df


def add(name, desc, sources, df):
    assert len(name) <= 31, name
    df = relabel(df)
    for c in METHOD_COLS:
        if c in df.columns:
            df[c] = df[c].map(method_name)
    for c in ("variant", "comparison"):
        if c in df.columns:
            df[c] = df[c].map(weight_name)
    df = df.rename(columns={c: method_name(c) for c in df.columns if c.lower() in METHODS})
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).rename(columns=RENAME_COLS)
    for k, w in WEIGHTS.items():
        desc = desc.replace(k, w)
    for k in sorted(LABELS, key=len, reverse=True):
        desc = desc.replace(f"({k}: ", "(").replace(f"({k})", f"({LABELS[k]})").replace(f" {k}.", ".")
    # TACCO was a reserve run and is not part of this paper's comparisons.
    if "method" in df.columns:
        df = df[~df["method"].astype(str).str.contains("tacco", case=False)].reset_index(drop=True)
    if len(df) > MAX_ROWS:
        skipped.append((name, f"{len(df)} rows", ", ".join(rel(s) for s in sources)))
        return
    sheets.append((name, desc, [rel(s) for s in sources], df))


# 1. Spotless silver per data set (13 methods x 54 data sets x 4 metrics)
f = ER / "silver_per_dataset_lam0_vs_competitors.csv"
df = read(f)
methods = [c for c in df.columns if c not in ("tissue", "pattern", "metric")]
methods = ["FlashDeconv"] + [m for m in methods if m != "FlashDeconv"]
df = df[["metric", "tissue", "pattern"] + methods].sort_values(["metric", "tissue", "pattern"])
add("Spotless_silver_per_dataset",
    "Spotless silver standard: per-data-set metric (corr = Pearson, rmse, jsd, aupr) for FlashDeconv "
    "(final_default_lam0: package defaults with lambda = 0) and 12 competing methods; one column per "
    "method, 54 data sets x 4 metrics.",
    [f], df.reset_index(drop=True))
f = ROOT / "results" / "controls_editor" / "c1_fd_aggregate_settings.csv"
add("Spotless_silver_layouts",
    "Spotless silver standard, FlashDeconv per data set under three spot layouts: default automatic lambda "
    "with spots on a square lattice in file order (lattice), default automatic lambda at randomly permuted "
    "lattice positions (perm0-perm4), and lambda = 0 (lam0).", [f], read(f))

# 2. Paired Wilcoxon tests
f = SP / "silver_paired_wilcoxon_final.csv"
add("Spotless_silver_wilcoxon",
    "Paired two-sided Wilcoxon signed-rank tests, FlashDeconv (final_default_lam0) vs each competitor "
    "across 54 silver data sets, per metric.", [f], silver(read(f)))

# 3. Per-cell-type silver
f = SP / "fd_per_celltype_silver.csv.gz"
df = silver(read(f)).drop(columns=["version"], errors="ignore")
add("Spotless_silver_per_celltype",
    "FlashDeconv (final_default_lam0) per-cell-type metrics on each silver data set, with abundance category "
    "(rare / moderate / abundant).", [f], df)
f = SP / "silver_per_celltype_by_abundance.csv"
add("Spotless_silver_by_abundance",
    "FlashDeconv (final_default_lam0) per-cell-type metrics averaged by abundance category (n = "
    "cell-type x data-set pairs).", [f], silver(read(f)))

# 4. Gold standard, FlashDeconv + competitors per field of view
f1, f2 = ER / "gold_realxy" / "fd_aggregate_gold.csv", SP / "gold_competitors_recomputed.csv"
fd = read(f1)
fd = fd[fd["config"] == "final_default_realxy"].reset_index(drop=True)  # measured spot coordinates
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
    "view and method, FlashDeconv (final_default_realxy: package defaults on the measured spot coordinates) and 12 competitors recomputed with the same metric code.",
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
    a = read_li(fs)
    b = read_li(fj).rename(columns={"rank": "rank_section_JSD"})
    m = a.merge(b, on="method", how="outer")
    fp = LI / f"merfish_{res}_pooled_vs_published.csv"
    if fp.exists():
        p = read_li(fp).rename(columns={"JSD": "pooled_JSD", "total_RMSE": "pooled_total_RMSE"})
        m = m.merge(p, on="method", how="outer")
        srcs.append(fp)
    m.insert(0, "dataset", f"MERFISH {res} um")
    rows.append(m)
    srcs += [fs, fj]
fq = LI / "seqfish10000_vs_published.csv"
q = read_li(fq).rename(columns={"JSD": "pooled_JSD", "rank_JSD": "rank_pooled_JSD"})
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
df = read(f)
# FlashDeconv rows of the reported version only (default and lambda = 0)
df = df[(df.method != "FlashDeconv") | df["mode"].isin(["final_default_auto", "final_default_l0"])]
df = df.drop(columns=["source", "reserve", "r_legacy", "auprc_legacy"])
add("Xenium_CRC_pseudoVHD",
    "Xenium CRC pseudo-Visium HD benchmark: metrics per bin size (2, 4, 8, 16 and 32 um; resolution_um), "
    "method, mode (FlashDeconv: default or lambda = 0; RCTD: full or doublet) and evaluation set "
    "('all' = all bins predicted by that method; 'common_<RCTD mode>_umi<k>' = the bins retained by RCTD in "
    "that mode with minimum-UMI filter k, on which every method is evaluated). umi_min: RCTD minimum-UMI "
    "filter; coverage: fraction of all bins predicted by the method. pearson_flat / ap_flat: Pearson correlation / "
    "average precision over all bin x cell-type entries; type_r / ap_type_mean: mean over cell types.", [f], df)
f = RES / "benchmarks" / "c2" / "c2_supp_table.csv"
if f.exists():
    add("Xenium_CRC_supp_table", "Xenium CRC pseudo-Visium HD summary table per bin size (2, 4, 8, 16 and "
        "32 um), method and evaluation set (all predicted bins, or bins shared with RCTD full / doublet).", [f], read(f))
else:
    skipped.append(("Xenium_CRC_supp_table", "file absent at build time", rel(f)))

# 9. Runtime / memory
f = RES / "benchmarks" / "c1" / "c1_runtime_table_final.csv"
df = read(f)
# FlashDeconv rows of the reported version and settings only
df = df[~df.method.str.startswith("flashdeconv") | (df.method.isin(["flashdeconv_final", "flashdeconv_final_coldcache"])
                                                     & (df["mode"] == "default"))]
df = df[df.status != "NOT_RUN"]


def cgroup_peak(notes):
    m = re.search(r"cgroup_peak_gb=([0-9.]+)", str(notes))
    return float(m.group(1)) if m else float("nan")


def runtime_memory(r):
    """Peak memory (GB) and its source. For failed runs the per-process sample misses the final
    allocation burst, so the job's cgroup peak (or SLURM MaxRSS) is reported instead."""
    if r.status == "OOM":
        cg = cgroup_peak(r.notes)
        if cg == cg:
            return cg, "cgroup peak of the SLURM job"
        if "MaxRSS" in str(r.notes):
            return r.peak_rss_gb, "SLURM MaxRSS of the batch step"
        return float("nan"), ""
    if r.status == "TIMEOUT":
        return float("nan"), ""
    return r.peak_rss_gb, "sampled peak RSS of the process tree"


def runtime_note(r):
    if r.status == "OOM":
        limit = "470" if r.method == "cell2location" else "500"
        m = re.search(r"cannot allocate vector of size ([0-9.]+) Gb", str(r.notes))
        if m:
            return (f"out of memory: requested a {float(m.group(1)):,.0f}-GB allocation, exceeding the "
                    f"{limit}-GB memory limit")
        return f"out of memory: exceeded the {limit}-GB memory limit"
    if r.status == "TIMEOUT":
        return "did not finish within the 24-h time limit"
    if r["mode"] == "ref_train":
        return "one-time reference model training"
    return ""


mem = df.apply(runtime_memory, axis=1, result_type="expand")
df = df.assign(peak_rss_gb=mem[0], memory_source=mem[1], note=df.apply(runtime_note, axis=1))
df = df.drop(columns=["hosts", "notes"]).rename(columns={"peak_rss_gb": "peak_memory_gb",
                                                         "scale": "n_spots"})
add("Runtime_memory",
    "Runtime and peak memory per method, mode and number of spots (n_spots; median_fit_s = median fit "
    "time in s over n_reps repeats; peak_memory_gb = peak memory in GB, source given in memory_source). "
    "Failed runs (status OOM = out of memory, TIMEOUT = 24-h limit) have no fit time; for OOM runs the "
    "peak is the memory reached before the job was killed or the allocation failed. Cell2location ran on "
    "a GPU (gpu_model); ref_train is its one-time reference training (n_spots = 0).",
    [f], df)

# 10. Gene weighting
W = RES / "weighting"
WL = ER / "weighting_lam0"
WDEF = ("Gene weightings (variant): leverage = expected leverage weights (package default); equal = "
        "uniform scores, every gene weighted equally; variance = scores equal to the between-type "
        "variance of the log-normalized reference signature. Only the per-gene scores differ.")
add("Weighting_spotless_acc",
    "Gene-weighting ablation on Spotless silver (lambda = 0): per-data-set accuracy for each weighting "
    "scheme (variant) and depth fraction (frac = fraction of counts retained by binomial thinning; median_umi = "
    "resulting median UMIs per spot). " + WDEF,
    [WL / "spotless_acc_lam0.csv"], read(WL / "spotless_acc_lam0.csv"))
add("Weighting_spotless_stats",
    "Gene-weighting ablation on Spotless silver (lambda = 0): paired statistics of leverage vs equal and "
    "leverage vs variance weighting per depth fraction (mean_leverage / mean_other = mean metric; "
    "median_diff with bootstrap CI ci_lo-ci_hi and mean_diff = leverage minus other; win_frac = fraction of "
    "pairs in which leverage is better; rank-biserial correlation; Wilcoxon signed-rank p). " + WDEF,
    [WL / "stats_spotless_lam0.csv"], read(WL / "stats_spotless_lam0.csv"))
add("Weighting_xenium_acc",
    "Gene-weighting ablation on Xenium CRC pseudo-Visium HD (2, 4 and 8 um bins): accuracy per bin size "
    "and weighting scheme (variant); weight_cv = coefficient of variation of the gene weights. " + WDEF,
    [W / "xenium_acc.csv"], read(W / "xenium_acc.csv").drop(columns=["r_legacy"]))
add("Weighting_xenium_stats",
    "Gene-weighting ablation on Xenium CRC pseudo-Visium HD: paired statistics of leverage vs equal and "
    "leverage vs variance weighting across cell types (all or rare types) or for the single global fit, "
    "per bin size; columns as in Weighting_spotless_stats. " + WDEF,
    [W / "stats_xenium.csv"], read(W / "stats_xenium.csv"))

# 11. Laplacian ablation
f = ER / "gold_realxy" / "fd_aggregate_gold.csv"
add("Spatial_ablation_gold",
    "Spatial-penalty ablation on the gold standards with real spot coordinates: FlashDeconv per field of "
    "view with automatic lambda (final_default_realxy) and lambda = 0 (final_default_lam0_realxy). "
    "Metrics are computed after restricting predictions to the ground-truth cell types and renormalising "
    "to sum to one; corr_before_renormalization is the Pearson correlation before this step.",
    [f], read(f)[lambda d: d.config.isin(["final_default_realxy", "final_default_lam0_realxy"])]
    .reset_index(drop=True))

# 12. Marker scoring
M = ER / "marker_scoring_lam0"
add("Marker_scoring",
    "Spotless silver: FlashDeconv (lambda = 0) vs marker-based scoring baselines (markers selected by "
    "maximum expression gap or by Wilcoxon rank-sum test): per-cell-type metrics per tissue and method.",
    [M / "marker_scoring_comparison.csv"], read(M / "marker_scoring_comparison.csv"))

# 13. Reference diagnostic
R = RES / "refdiag"
TISSUES = {1: "brain_cortex", 2: "cerebellum_cell", 3: "cerebellum_nucleus", 4: "hippocampus",
           5: "kidney", 6: "scc_p5"}
PATTERNS = {2: "artificial_diverse_distinct", 4: "artificial_diverse_overlap",
            7: "artificial_dominant_rare_celltype_diverse", 8: "artificial_regional_rare_celltype_diverse"}
NULLS = {"left_half": "left-half", "central": "central matching", "auto": "automatic (default)"}


def spotless_ids(d):
    d = d.copy()
    ds, pat = d.pop("ds").astype(int), d.pop("pattern").astype(int)
    d.insert(0, "tissue", ds.map(TISSUES))
    d.insert(1, "pattern", pat.map(PATTERNS))
    return d


f = R / "spotless_removal_auroc.csv"
d = spotless_ids(read(f)).rename(columns={"class": "abundance_category", "distinct": "distinctness",
                                          "flag_rate_pos": "flag_rate_pos_left_half",
                                          "flag_rate_neg": "flag_rate_neg_left_half"})
add("Refdiag_removal_auroc",
    "Reference-completeness diagnostic on Spotless silver (24 data sets: 6 tissues x 4 patterns; unpooled "
    "score): after removing "
    "one cell type (removed) from the reference, AUROC/AUPRC of the diagnostic score for spots in which "
    "the removed type's true proportion exceeds thr (positives, n_pos) vs spots in which it is below 0.01 "
    "(negatives, n_neg). AUROC and AUPRC use the calibrated score, whose ranking does not depend on the "
    "empirical null; the flag rates among positives and negatives use the left-half null. distinctness = "
    "1 - uncentred R^2 of the removed type's profile regressed (NNLS) on the remaining profiles.",
    [f], d)
f = R / "complete_reference_null_comparison.csv"  # Xenium CRC rows
f0 = ER / "refdiag_lam0" / "spotless_null_comparison.csv"  # Spotless rows, fits with lambda = 0
c = pd.concat([read(f0), read(f).query("set != 'spotless'")], ignore_index=True)
c["null_selected_by_auto"] = c["null_used"].map(NULLS)
auto_sel = c[c["null"] == "auto"].set_index(["set", "ds", "pattern", "key"])["null_selected_by_auto"]
wide = c.pivot_table(index=["set", "ds", "pattern", "key", "n_bins"], columns="null", values="flag_rate",
                     dropna=False).reset_index()
wide = wide.dropna(subset=["left_half"])
wide = wide.rename(columns={"left_half": "flag_rate_left_half", "central": "flag_rate_central_matching",
                            "auto": "flag_rate_automatic_default"})
wide["null_selected_by_auto"] = [auto_sel[(r.set, r.ds, r.pattern, r.key)]
                                 for r in wide[["set", "ds", "pattern", "key"]].itertuples()]
wide.columns.name = None
sp = wide[wide.set == "spotless"]
sp = spotless_ids(sp.drop(columns=["set"]))
sp.insert(0, "data_set", "Spotless silver (lambda = 0)")
xe = wide[wide.set != "spotless"].drop(columns=["ds", "pattern"])
xe.insert(0, "data_set", xe.pop("set").str.replace(r"c2_(\d+)um", r"Xenium CRC pseudo-Visium HD, \1 um",
                                                     regex=True))
cref = pd.concat([sp, xe], ignore_index=True)
cref["score"] = cref.pop("key").map({"score": "unpooled", "score_pooled": "pooled"})
cref = cref[["data_set", "tissue", "pattern", "score", "n_bins", "flag_rate_left_half",
             "flag_rate_central_matching", "flag_rate_automatic_default", "null_selected_by_auto"]]
add("Refdiag_complete_ref_flags",
    "Reference diagnostic with a complete reference (every bin is explained, so flags are false "
    "positives; nominal rate 0.05): fraction of bins flagged under each empirical null - left-half "
    "(median and 16th percentile), central matching, and the automatic default (null = 'auto', the "
    "estimator with the smaller scale; null_selected_by_auto names the estimator it chose). Spotless "
    "silver data sets (fits with lambda = 0) use the unpooled score; Xenium CRC bins report both the unpooled and the pooled "
    "score. The main-text calibration uses the automatic default.",
    [f0, f], cref)
f = R / "intestine_region_flags_default.csv"
add("Refdiag_intestine_regions",
    "Reference diagnostic on mouse small intestine Visium HD (8 um bins), package defaults, with an "
    "epithelium-only (haber) or composite reference: fraction of bins flagged per region with the "
    "unpooled (flag_rate) and pooled (flag_rate_pooled) score, and median pooled score per region. Flags "
    "use the left-half null, which is also the estimator the automatic default selects for both "
    "references. Regions are defined from raw marker counts, not from deconvolution, and may overlap: "
    "follicle = B-cell follicle mask (Cd79a, Ms4a1, Cd19, Cr2) dilated by 2 bins (16 um); muscle = "
    "5 x 5-bin (40 um) smoothed smooth-muscle marker expression (Acta2, Myh11, Des, Cnn1, Tagln) >= 100 "
    "counts per 10,000 UMIs and above the epithelial marker expression; epi_low = bins with low "
    "epithelial marker expression, i.e. 5 x 5-bin smoothed Epcam, Vil1, Krt8, Krt19, Cldn7, Cdh1, Elf3 "
    "and Krt20 < 30 counts per 10,000 UMIs; epithelium = bins outside all three masks; all = every bin.",
    [f], read(f).rename(columns={"flag": "flag_rate", "flag_pooled": "flag_rate_pooled"}))

# README
readme = pd.DataFrame(
    [{"sheet": n, "n_rows": len(d), "n_columns": d.shape[1], "content": desc}
     for n, desc, s, d in sheets]
)
assert not skipped, skipped
header = pd.DataFrame([{
    "sheet": "README",
    "content": "Supplementary Data 1. FlashDeconv benchmark results (package v0.2.0, default settings "
               "unless stated; Spotless silver-standard results use lambda = 0). Method names follow the "
               "manuscript; gene weightings are named leverage (default), equal and variance; metric "
               "abbreviations: corr/pearson = Pearson correlation, rmse = root-mean-square error, jsd = "
               "Jensen-Shannon divergence, aupr/auprc/ap = area under the precision-recall curve / "
               "average precision.",
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
