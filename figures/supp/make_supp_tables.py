"""Generate the LaTeX tables of the Supplementary Information from final result files.

Writes paper/supp_tables/*.tex (each file is a complete table environment).
Run: .venv/bin/python validation/figures/supp/make_supp_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
R = PROJ / "results"
RF = R / "rerun_final"
SP = RF / "benchmarks" / "spotless"
V = PROJ / "validation"
OUT = PROJ / "paper" / "supp_tables"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {
    "flashdeconv": "FlashDeconv", "FlashDeconv": "FlashDeconv", "rctd": "RCTD", "cell2location": "Cell2location",
    "spatialdwls": "SpatialDWLS", "stereoscope": "Stereoscope", "music": "MuSiC", "nnls": "NNLS",
    "seurat": "Seurat", "destvi": "DestVI", "spotlight": "SPOTlight", "stride": "STRIDE",
    "tangram": "Tangram", "dstg": "DSTG",
}
LOW = {"corr": False, "rmse": True, "jsd": True, "aupr": False}
METS = ["corr", "rmse", "jsd", "aupr"]


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        m = p / 10 ** e
        return f"${m:.1f}\\times10^{{{e}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def bold_best(vals, low):
    """Return formatted strings with the best value in bold."""
    v = np.array([np.nan if x is None else x for x in vals], float)
    best = np.nanmin(v) if low else np.nanmax(v)
    return [("\\textbf{" + fmt(x) + "}") if np.isfinite(x) and np.isclose(x, best) else fmt(x) for x in v]


def write(name, text):
    # keep wide tables within the text width
    text = text.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}")
    text = text.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")
    (OUT / f"{name}.tex").write_text(text)
    print("wrote", OUT / f"{name}.tex")


# ---------------------------------------------------------------------------
# Spotless silver: all 13 methods, 4 metrics, paired Wilcoxon vs FlashDeconv
# ---------------------------------------------------------------------------
w = pd.read_csv(SP / "silver_paired_wilcoxon_final.csv")
w = w[w.config == "final_default"]
rows = {}
for m in METS:
    sub = w[w.metric == m]
    rows.setdefault("FlashDeconv", {})[m] = sub.flash_mean.iloc[0]
    for _, r in sub.iterrows():
        rows.setdefault(NAMES[r.comparator], {})[m] = r.comparator_mean
        rows[NAMES[r.comparator]][m + "_p"] = r.p_value
        rows[NAMES[r.comparator]][m + "_n"] = f"{int(r.flash_better)}/{int(r.n)}"
df = pd.DataFrame(rows).T
ranks = {m: df[m].astype(float).rank(ascending=LOW[m], method="min") for m in METS}
df = df.sort_values("corr", ascending=False)
lines = []
for meth, r in df.iterrows():
    cells = [meth]
    for m in METS:
        val = f"{float(r[m]):.3f} ({int(ranks[m][meth])})"
        if meth == "FlashDeconv":
            cells += ["\\textbf{" + val + "}", "--"]
        else:
            cells += [val, f"{r[m + '_n']}; {fmt_p(float(r[m + '_p']))}"]
    lines.append(" & ".join(cells) + " \\\\")
write("tab_spotless_silver", r"""\begin{table}[H]
\centering
\caption{\textbf{Spotless silver standards: all methods and metrics.} Mean over 54 data sets (rank among 13 methods in parentheses). For each competitor, the number of data sets in which FlashDeconv is better and the two-sided paired Wilcoxon signed-rank $P$ value are given. SpatialDWLS returned no JSD for 3 data sets (JSD comparison on 51 data sets). FlashDeconv: package defaults.}
\label{tab:spotless_silver}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lllllllll@{}}
\toprule
 & \multicolumn{2}{c}{Pearson $r$} & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{JSD} & \multicolumn{2}{c}{AUPR} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & Mean (rank) & FD better; $P$ & Mean (rank) & FD better; $P$ & Mean (rank) & FD better; $P$ & Mean (rank) & FD better; $P$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Unified table: FlashDeconv rank and best competitor on every Spotless benchmark
# ---------------------------------------------------------------------------
cs = pd.read_csv(SP / "cases_summary_ranks.csv")
ss = pd.read_csv(SP / "silver_summary_ranks.csv")
MLAB = {"corr": "Pearson $r$", "rmse": "RMSE", "jsd": "JSD", "aupr": "AUPR"}
BLAB = {"silver": "Silver standards (54 data sets)", "seqfish_cortex_svz": "seqFISH+ cortex/SVZ (7 FOVs)",
        "seqfish_ob": "seqFISH+ olfactory bulb (7 FOVs)", "starmap": "STARmap (108 spots)",
        "liver": "Liver (4 Visium sections)", "liver_ref_stability": "Liver, reference stability",
        "melanoma": "Melanoma (3 Visium sections)"}
lines = []
blocks = [("silver", ss[ss.config == "final_default"].assign(benchmark="silver", flash=lambda d: d.flash_mean,
                                                            best_value=lambda d: d.best_competitor_mean))]
for b in ["seqfish_cortex_svz", "seqfish_ob", "starmap", "liver", "liver_ref_stability"]:
    blocks.append((b, cs[(cs.benchmark == b) & (cs.config == "final_default")]))
for b, g in blocks:
    first = True
    for _, r in g.iterrows():
        lab = BLAB[b] if first else ""
        met = MLAB[r.metric] + (" (between references)" if b == "liver_ref_stability" else "")
        lines.append(f"{lab} & {met} & {fmt(r.flash)} & {int(r.rank_of_13)} & {NAMES.get(r.best_competitor, r.best_competitor)} & {fmt(r.best_value)} \\\\")
        first = False
    lines.append("\\addlinespace")
mel = cs[cs.benchmark == "melanoma"].set_index("config")
for cfg, lab in [("final_default", "Melanoma (3 Visium sections)"), ("final_default_pearson", "")]:
    r = mel.loc[cfg]
    met = "JSD" + (" (Pearson residuals)" if cfg.endswith("pearson") else " (default)")
    lines.append(f"{lab} & {met} & {fmt(r.flash)} & {int(r.rank_of_13)} & {NAMES.get(r.best_competitor)} & {fmt(r.best_value)} \\\\")
write("tab_unified_metrics", r"""\begin{table}[H]
\centering
\caption{\textbf{FlashDeconv across all Spotless benchmarks.} FlashDeconv value, rank among 13 methods (1 = best) and the best competing method. Silver-standard values are means over 54 data sets; gold-standard values are means over fields of view (FOVs) or the single STARmap section, with all competitors rescored from their published predictions with the same metric code and ground truth; liver JSD and AUPR (mean of portal- and central-vein endothelial AUPR) and melanoma JSD are compared with the values reported by Spotless; reference stability is the mean JSD between the tissue compositions estimated with the three liver references (lower is more stable). FlashDeconv used package defaults; the melanoma row labelled Pearson residuals uses \texttt{preprocess="pearson"} with otherwise default parameters.}
\label{tab:unified_metrics}
\small
\begin{tabular}{@{}llrrlr@{}}
\toprule
Benchmark & Metric & FlashDeconv & Rank & Best competitor & Value \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Gold standards: all methods, all metrics
# ---------------------------------------------------------------------------
fd = pd.read_csv(SP / "fd_aggregate_gold.csv")
fd = fd[fd.config == "final_default"].assign(method="FlashDeconv")
cp = pd.read_csv(SP / "gold_competitors_recomputed.csv")
allg = pd.concat([fd[["benchmark", "method"] + METS], cp[["benchmark", "method"] + METS]])
allg["method"] = allg.method.map(lambda s: NAMES.get(s, s))
mean = allg.groupby(["benchmark", "method"])[METS].mean()
bench = ["seqfish_cortex_svz", "seqfish_ob", "starmap"]
methods = mean.loc["starmap"].sort_values("corr", ascending=False).index.tolist()
cols = {}
for b in bench:
    for m in METS:
        vals = [mean.loc[(b, meth), m] if (b, meth) in mean.index else np.nan for meth in methods]
        cols[(b, m)] = bold_best(vals, LOW[m])
lines = []
for i, meth in enumerate(methods):
    lab = "\\textbf{FlashDeconv}" if meth == "FlashDeconv" else meth
    lines.append(lab + " & " + " & ".join(cols[(b, m)][i] for b in bench for m in METS) + " \\\\")
write("tab_gold", r"""\begin{table}[H]
\centering
\caption{\textbf{Spotless gold standards: all methods and metrics.} Means over the seven fields of view of each seqFISH+ data set and over the single STARmap section. Competitor predictions from the published Spotless results were clipped at zero, matched by cell-type name, restricted to the cell types in the ground truth, renormalized and scored with the same code as FlashDeconv. Bold, best value per column. $r$, Pearson correlation.}
\label{tab:gold}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrrrrrrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{seqFISH+ cortex/SVZ} & \multicolumn{4}{c}{seqFISH+ olfactory bulb} & \multicolumn{4}{c}{STARmap} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
Method & $r$ & RMSE & JSD & AUPR & $r$ & RMSE & JSD & AUPR & $r$ & RMSE & JSD & AUPR \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Liver and melanoma: all methods
# ---------------------------------------------------------------------------
lv = pd.read_csv(SP / "liver_case_study.csv")
lv = lv[lv.config == "final_default"][["jsd", "aupr_portal", "aupr_central", "aupr_mean"]].mean()
sp = pd.read_csv(V / "spotless_liver_all_methods.csv")
sp = sp[sp.digest == "all"].pivot(index="method", columns="metric", values="value")
st = pd.read_csv(SP / "liver_stability.csv")
st = st[st.config == "final_default"].jsd.mean()
sst = pd.read_csv(V / "spotless_liver_ref_sensitivity.csv").groupby("method").jsd.mean()
ms = pd.read_csv(V / "melanoma_analysis" / "spotless_melanoma_jsd.csv").set_index("method").jsd
mf = pd.read_csv(SP / "melanoma_fixed.csv").groupby("config")[["jsd", "melanocytic", "tcell"]].mean()
tab = pd.DataFrame({"liver_jsd": sp["jsd"], "liver_aupr": sp["aupr"], "stab": sst, "mel": ms})
tab.index = [NAMES.get(i, i) for i in tab.index]
tab.loc["FlashDeconv"] = [lv.jsd, lv.aupr_mean, st, mf.loc["final_default", "jsd"]]
tab = tab.sort_values("liver_jsd")
colv = {c: bold_best(tab[c].tolist(), c != "liver_aupr") for c in tab.columns}
rk = {c: tab[c].rank(ascending=(c != "liver_aupr"), method="min") for c in tab.columns}
lines = []
for i, meth in enumerate(tab.index):
    lab = "\\textbf{FlashDeconv}" if meth == "FlashDeconv" else meth
    cells = []
    for c in tab.columns:
        cells.append(colv[c][i] + (f" ({int(rk[c][meth])})" if np.isfinite(tab.loc[meth, c]) else ""))
    lines.append(lab + " & " + " & ".join(cells) + " \\\\")
pr = mf.loc["final_default_pearson"]
dr = mf.loc["final_default"]
write("tab_liver_melanoma", r"""\begin{table}[H]
\centering
\caption{\textbf{Spotless liver and melanoma case studies: all methods.} Liver: JSD between the predicted tissue composition and single-nucleus cell-type frequencies and AUPR for portal- and central-vein endothelial cells (mean of both), averaged over four Visium sections; reference stability, mean JSD between the compositions estimated with the ex vivo scRNA-seq, in vivo scRNA-seq and snRNA-seq references. Melanoma: JSD between the tissue composition and Molecular Cartography frequencies over seven classes, averaged over three sections. Values of other methods are those reported by Spotless; rank among 13 methods in parentheses; bold, best. With \texttt{preprocess="pearson"}, FlashDeconv reaches a melanoma JSD of """ + f"{pr.jsd:.4f}" + r""" (rank 4; melanocytic fraction """ + f"{pr.melanocytic:.3f}" + r""", T/NK fraction """ + f"{pr.tcell:.3f}" + r"""; default: """ + f"{dr.melanocytic:.3f}" + r""" and """ + f"{dr.tcell:.4f}" + r"""; ground truth 0.848 and 0.047).}
\label{tab:liver_melanoma}
\small
\begin{tabular}{@{}lllll@{}}
\toprule
Method & Liver JSD & Liver AUPR & Liver reference stability & Melanoma JSD \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Li et al. 2023: all 19 methods
# ---------------------------------------------------------------------------
LI = RF / "benchmarks" / "li2023" / "expected"
merf = {res: pd.read_csv(LI / f"merfish_{res}_suppdata1_vs_published.csv").set_index("method") for res in [100, 50, 20]}
sq = pd.read_csv(LI / "seqfish10000_vs_published.csv").set_index("method")
canon = {"stereoscope": "Stereoscope", "SpaOtsc": "SpaOTsc"}
for d in list(merf.values()) + [sq]:
    d.rename(index=lambda s: canon.get(s, s), inplace=True)
order = merf[20].sort_values("total_RMSE").index.tolist()
lines = []
for meth in order:
    cells = []
    for res in [100, 50, 20]:
        r = merf[res].loc[meth]
        cells += [f"{r.total_RMSE:.3f} ({int(r.rank_RMSE)})", f"{r.mean_typeJSD:.3f} ({int(r.rank_typeJSD)})"]
    r = sq.loc[meth]
    cells += [f"{r.total_RMSE:.3f} ({int(r.rank_RMSE)})", f"{r.JSD:.3f} ({int(r.rank_JSD)})"]
    lab = "\\textbf{FlashDeconv}" if meth == "FlashDeconv" else meth
    lines.append(lab + " & " + " & ".join(cells) + " \\\\")
write("tab_li2023", r"""\begin{table}[H]
\centering
\caption{\textbf{Independent benchmark of Li et al.: all 19 methods.} MERFISH mouse hypothalamus binned at 100, 50 and 20~$\mu$m (3,067, 13,375 and 44,679 spots): RMSE over all spot--cell-type entries and mean per-cell-type JSD; seqFISH+ (71 spots, 10,000 genes): RMSE and median per-spot JSD (base 2). Rank among 19 methods in parentheses. Values of the 18 published methods are from the source data of Li et al.\ (MERFISH) or were recomputed from the released predictions with our metric code (seqFISH+), which reproduced the published values. Methods are ordered by RMSE at 20~$\mu$m.}
\label{tab:li2023}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lllllllll@{}}
\toprule
 & \multicolumn{2}{c}{MERFISH 100~$\mu$m} & \multicolumn{2}{c}{MERFISH 50~$\mu$m} & \multicolumn{2}{c}{MERFISH 20~$\mu$m} & \multicolumn{2}{c}{seqFISH+} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & RMSE & Type JSD & RMSE & Type JSD & RMSE & Type JSD & RMSE & JSD \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Marker scoring
# ---------------------------------------------------------------------------
from scipy.stats import wilcoxon  # noqa: E402

ms_ = pd.read_csv(RF / "benchmarks" / "marker_scoring" / "final_default" / "marker_scoring_comparison.csv")
ML = {"FlashDeconv": "FlashDeconv", "MarkerScoring_maxgap": "Marker scoring (max-gap markers)",
      "MarkerScoring_wilcoxon": "Marker scoring (Wilcoxon markers)"}
lines = []
for meth in ML:
    d = ms_[ms_.method == meth]
    cells = [ML[meth]]
    for cat in ["rare", "moderate", "abundant", None]:
        g = d if cat is None else d[d.category == cat]
        cells += [f"{g.pearson.mean():.3f}", f"{g.auprc.mean():.3f}", f"{g.rmse.mean():.3f}"]
    lines.append(" & ".join(cells) + " \\\\")
piv = ms_.pivot_table(index=["tissue", "cell_type"], columns="method", values=["pearson", "auprc"])
foot = []
for meth in ["MarkerScoring_maxgap", "MarkerScoring_wilcoxon"]:
    bp = int((piv[("pearson", "FlashDeconv")] > piv[("pearson", meth)]).sum())
    ba = int((piv[("auprc", "FlashDeconv")] > piv[("auprc", meth)]).sum())
    wa = int((piv[("auprc", "FlashDeconv")] < piv[("auprc", meth)]).sum())
    pp = wilcoxon(piv[("pearson", "FlashDeconv")], piv[("pearson", meth)]).pvalue
    pa = wilcoxon(piv[("auprc", "FlashDeconv")], piv[("auprc", meth)]).pvalue
    foot.append(f"{ML[meth]}: FlashDeconv higher Pearson for {bp} of 76 cell types ($P = {fmt_p(pp).strip('$')}$) and higher AUPR for {ba} (lower for {wa}; $P = {fmt_p(pa).strip('$')}$)")
write("tab_marker_scoring", r"""\begin{table}[H]
\centering
\caption{\textbf{FlashDeconv versus marker-gene scoring on the Spotless silver standards.} Per-cell-type Pearson correlation, AUPR and RMSE, averaged over the 76 cell types of the first abundance pattern of each of the six tissues, by mean abundance (rare, $<5\%$, $n = 26$; moderate, 5--15\%, $n = 42$; abundant, $>15\%$, $n = 8$). Paired two-sided Wilcoxon tests over the 76 cell types: """ + "; ".join(foot) + r""".}
\label{tab:marker_scoring}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{Rare} & \multicolumn{3}{c}{Moderate} & \multicolumn{3}{c}{Abundant} & \multicolumn{3}{c}{All} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
Method & $r$ & AUPR & RMSE & $r$ & AUPR & RMSE & $r$ & AUPR & RMSE & $r$ & AUPR & RMSE \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Runtime and memory
# ---------------------------------------------------------------------------
rt = pd.read_csv(RF / "benchmarks" / "c1" / "c1_runtime_table_final.csv")
ROWS = [("flashdeconv_final", "default", "FlashDeconv"),
        ("flashdeconv_final_coldcache", "default", "FlashDeconv, first call (JIT compilation)"),
        ("rctd", "doublet", "RCTD, doublet mode"), ("rctd", "full", "RCTD, full mode"),
        ("card", "default", "CARD"), ("cell2location", "fullbatch", "Cell2location (GPU)")]
SCALES = [10000, 100000, 300000, 1000000]


def cell(r):
    if r is None:
        return "--", "--"
    s = str(r.status)
    if s == "OK":
        t = r.median_fit_s
        ts = f"{t:.1f}~s" if t < 60 else (f"{t / 60:.1f}~min" if t < 3600 else f"{t / 3600:.1f}~h")
        return ts, f"{r.peak_rss_gb:.1f}"
    if s == "OOM":
        return "OOM", "--"
    return "\\TBD{running}" if s.startswith("RUNNING") else "\\TBD{pending}", "--"


lines = []
for meth, mode, lab in ROWS:
    cells = [lab]
    for sc in SCALES:
        g = rt[(rt.method == meth) & (rt["mode"] == mode) & (rt.scale == sc)]
        cells += list(cell(g.iloc[0] if len(g) else None))
    lines.append(" & ".join(cells) + " \\\\")
write("tab_runtime", r"""\begin{table}[H]
\centering
\caption{\textbf{Runtime and peak memory on pooled CRC Visium HD 8-$\mu$m bins.} Nested random subsets of $10^4$ to $10^6$ bins (18,082 genes, 38 cell types); every CPU run used 32 threads on identical nodes (AMD EPYC 7763), Cell2location one NVIDIA A30 GPU. Wall time of the fitting call, excluding data loading, and peak memory (GB, summed proportional set size of all processes). FlashDeconv values are medians of three repetitions at $10^4$ and $10^5$ bins; all other entries are single runs. FlashDeconv converged in 153, 148, 159 and 170 iterations. The first-call row includes Numba compilation in a fresh process. OOM, the run exceeded the 500-GB memory limit (CARD at $10^6$ bins failed when allocating its dense spatial kernel). RCTD doublet mode at $10^6$ bins and Cell2location were still running or queued when this version was compiled.}
\label{tab:runtime}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lllllllll@{}}
\toprule
 & \multicolumn{2}{c}{$10^4$ bins} & \multicolumn{2}{c}{$10^5$ bins} & \multicolumn{2}{c}{$3\times10^5$ bins} & \multicolumn{2}{c}{$10^6$ bins} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & Time & GB & Time & GB & Time & GB & Time & GB \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Xenium CRC pseudo-Visium HD: all metrics, all bin sizes, all methods
# ---------------------------------------------------------------------------
c2 = pd.read_csv(RF / "benchmarks" / "c2" / "c2_supp_table.csv")
print(c2.method.unique(), c2.eval_set.unique())
MLAB2 = {"FlashDeconv": "FlashDeconv", "FlashDeconv (lambda=0)": "FlashDeconv ($\\lambda=0$)",
         "RCTD (doublet)": "RCTD doublet", "RCTD (full)": "RCTD full", "NNLS": "NNLS",
         "Marker scoring": "Marker scoring"}
EVAL = {"all predicted bins": "all", "common bins with RCTD doublet": "doublet", "common bins with RCTD full": "full"}
# TACCO (reserve run, results/benchmarks/c2/c2_reserve_tacco.csv) is not part of this paper's comparisons.
allc = c2[~c2.method.str.contains("TACCO", case=False)].copy()
MCOL = ["pearson_flat", "pearson_type_mean", "rmse", "jsd", "ap_flat", "ap_type_mean"]
print(allc.method.unique())


def c2_table(evalset, label, caption):
    lines = []
    for b in [2, 4, 8, 16, 32]:
        g = allc[(allc.bin_size_um == b) & (allc.eval_set == evalset)].copy()
        if evalset != "all predicted bins":
            g = g[~g.method.str.startswith("RCTD") | (g.method == ("RCTD (doublet)" if "doublet" in evalset else "RCTD (full)"))]
        order = [m for m in MLAB2 if m in g.method.values]
        g = g.set_index("method").loc[order]
        best = {c: (g[c].min() if c in ("rmse", "jsd") else g[c].max()) for c in MCOL}
        first = True
        for m, r in g.iterrows():
            cells = [f"{b}~$\\mu$m" if first else "", MLAB2[m], f"{int(r.n_bins):,}", f"{100 * r.coverage:.1f}"]
            for c in MCOL:
                v = f"{r[c]:.3f}"
                cells.append("\\textbf{" + v + "}" if np.isclose(r[c], best[c]) else v)
            lines.append(" & ".join(cells) + " \\\\")
            first = False
        lines.append("\\midrule" if b != 32 else "")
    return (r"""\begin{table}[H]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrrrr@{}}
\toprule
Bin & Method & Bins & Scored (\\%) & $r$ (all) & $r$ (type) & RMSE & JSD & AP (all) & AP (type) \\\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


write("tab_pseudo_vhd_all", c2_table(
    "all predicted bins", "tab:pseudo_vhd_all",
    r"\textbf{Xenium-derived CRC pseudo-Visium HD benchmark: all metrics on all bins each method predicted.} Patient P1, 38 cell types, bins of 2--32~$\mu$m. $r$ (all), Pearson correlation over all bin--cell-type entries; $r$ (type), mean per-cell-type Pearson correlation; RMSE and JSD per bin, averaged; AP (all), average precision over all entries; AP (type), mean per-type average precision (presence, true proportion $>0.01$). Scored, percentage of bins for which the method returned estimates; RCTD values refer only to the bins it scored (default UMI threshold of 100). FlashDeconv, package defaults (automatic $\lambda$). Bold, best per bin size."))
write("tab_pseudo_vhd_common", c2_table(
    "common bins with RCTD doublet", "tab:pseudo_vhd_doublet",
    r"\textbf{Xenium-derived CRC pseudo-Visium HD benchmark on the bins scored by RCTD doublet mode.} All methods evaluated on the same bins; metrics as in Supplementary Table~\ref{tab:pseudo_vhd_all}. Bold, best per bin size."))
write("tab_pseudo_vhd_full", c2_table(
    "common bins with RCTD full", "tab:pseudo_vhd_full",
    r"\textbf{Xenium-derived CRC pseudo-Visium HD benchmark on the bins scored by RCTD full mode.} All methods evaluated on the same bins; metrics as in Supplementary Table~\ref{tab:pseudo_vhd_all}. Bold, best per bin size."))

# ---------------------------------------------------------------------------
# Gene weighting (Spotless): body only (caption lives in supplementary.tex)
# ---------------------------------------------------------------------------
acc = pd.read_csv(RF / "weighting" / "spotless_acc.csv")
stw = pd.read_csv(RF / "weighting" / "stats_spotless.csv")
stw = stw[(stw.unit == "dataset") & (stw.comparison == "EXP_LEV vs UNIFORM")]
means = acc.groupby(["frac", "variant"])[METS if False else ["pearson", "rmse", "jsd", "aupr"]].mean()
WM = [("pearson", "Pearson $r$"), ("rmse", "RMSE"), ("jsd", "JSD"), ("aupr", "AUPR")]
lines = []
for frac in [1.0, 0.25, 0.10, 0.05]:
    mu = int(acc[(acc.frac == frac)].groupby("dataset").median_umi.first().median())
    cells = [f"{int(frac * 100)}\\%"]
    for m, _ in WM:
        lev, eq, var = (means.loc[(frac, v), m] for v in ["EXP_LEV", "UNIFORM", "VAR_REF"])
        p = stw[(np.isclose(stw.depth, frac)) & (stw.metric == m)].p_value.iloc[0]
        cells += [f"{lev:.3f}", f"{eq:.3f}", f"{var:.3f}", fmt_p(p)]
    lines.append(" & ".join(cells) + " \\\\")
write("tab_weighting_body", r"""\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrlrrrlrrrlrrrl@{}}
\toprule
 & \multicolumn{4}{c}{Pearson $r$} & \multicolumn{4}{c}{RMSE} & \multicolumn{4}{c}{JSD} & \multicolumn{4}{c}{AUPR} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}
Depth & Lev. & Eq. & Var. & $P$ & Lev. & Eq. & Var. & $P$ & Lev. & Eq. & Var. & $P$ & Lev. & Eq. & Var. & $P$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
# CRC per-patient statistics (from the consolidated final claims file)
# ---------------------------------------------------------------------------
cl = pd.read_csv(RF / "crc" / "claims_final.csv").set_index("claim").FINAL
tm = pd.read_csv(RF / "crc" / "timing_final.csv").set_index("sample")


def trip(key):
    return [x.strip() for x in str(cl[key]).split("/")]


hot = trip("hotspot bins P1/P2/P5 (5295/8305/3227)")
selfx = trip("Neutrophil self-enrichment x (16.6/22.8/56.2)")
rank = trip("Neutrophil self-enrichment rank among types (1=top)")
rank = [r.split(" ")[0] for r in rank]
agg = str(cl["aggregates per patient SR/TP"]).replace("P1:", "").replace("P2:", "").replace("P5:", "").split()
mreg = trip("SR median log2 mRegDC P1/P2/P5 (SN9 +1.4..+2.1)")
mac = trip("SR median log2 Macrophage P1/P2/P5 (SN9 +0.8..+1.7)")
lamp = str(cl["LAMP3 neighborhood fold (1.40; 1.13-1.67)"]).split("(")[1].rstrip(")").split("/")
lampp = trip("LAMP3 MWU p per patient")
rows = [("Bins (8~$\\mu$m)", [f"{int(tm.loc[s, 'n_bins']):,}" for s in ["P1_CRC", "P2_CRC", "P5_CRC"]]),
        ("Fitting time (s)", [f"{tm.loc[s, 'fit_seconds']:.1f}" for s in ["P1_CRC", "P2_CRC", "P5_CRC"]]),
        ("Iterations", [str(int(tm.loc[s, 'n_iterations'])) for s in ["P1_CRC", "P2_CRC", "P5_CRC"]]),
        ("Neutrophil hotspot bins", [f"{int(h):,}" for h in hot]),
        ("Neutrophil self-enrichment (fold; rank of 38)", [f"{a} ({b})" for a, b in zip(selfx, rank)]),
        ("Aggregates, stromal-resident/tumour-proximal", agg),
        ("mRegDC $\\log_2$ enrichment, stromal-resident (median)", mreg),
        ("Macrophage $\\log_2$ enrichment, stromal-resident (median)", mac),
        ("\\textit{LAMP3} fold in hotspot neighbourhoods ($P$)", [f"{a} ({fmt_p(float(b))})" for a, b in zip(lamp, lampp)])]
lines = [f"{lab} & " + " & ".join(v) + " \\\\" for lab, v in rows]
write("tab_crc_patients", r"""\begin{table}[H]
\centering
\caption{\textbf{Colorectal cancer cohort: per-patient statistics.} Visium HD sections of patients P1, P2 and P5 at 8~$\mu$m, 38-type Flex reference. Fitting time on 32 threads (AMD EPYC 7763). Hotspot bins, neutrophil proportion $\ge 0.1$. Self-enrichment, mean neutrophil proportion among the 30 nearest bins of hotspot bins relative to the section mean. Aggregates, DBSCAN clusters of hotspot bins with at least 50 neighbourhood bins. $\log_2$ enrichment, mean proportion within 120~$\mu$m of an aggregate relative to the section mean. \textit{LAMP3} fold, normalized expression within 100~$\mu$m of hotspot bins relative to random background bins (one-sided Mann--Whitney $P$). Source: \texttt{results/rerun\_final/crc/claims\_final.csv}, \texttt{timing\_final.csv}.}
\label{tab:crc_patients}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
 & P1 & P2 & P5 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Tumour-stroma interface: per-section summary
# ---------------------------------------------------------------------------
B1 = R / "b1_pilot"
ss1 = pd.concat([pd.read_csv(B1 / "section_summary.csv"), pd.read_csv(B1 / "section_summary_ext.csv")])
ss1 = ss1[~ss1.section.str.contains("sens") & (ss1.section != "OV10X_v1")].set_index("section")
SEC = [("CRC_P1", "CRC P1", "Colorectal", "Xenium, serial"), ("CRC_P2", "CRC P2", "Colorectal", "Xenium, serial"),
       ("CRC_P5", "CRC P5", "Colorectal", "Xenium, serial"), ("SPATCH_COAD", "COAD", "Colon adenocarcinoma", "CODEX, adjacent"),
       ("SPATCH_OV", "OV-1", "Ovarian", "CODEX, adjacent"), ("SPATCH_HCC", "HCC", "Hepatocellular", "CODEX, adjacent"),
       ("LUNG_X1", "Lung 1", "Lung", "Xenium v1, same section"), ("LUNG_X5K", "Lung 2", "Lung", "Xenium 5K, same section"),
       ("OV10X", "OV-2", "Ovarian", "Xenium 5K, adjacent")]
lines = []
for key, lab, cancer, orth in SEC:
    r = ss1.loc[key]
    lines.append(f"{lab} & {cancer} & {int(r.n_hd_bins):,} & {r.t_deconvolve_s:.1f} & {int(r.fd_n_iter)} & {orth} & {int(r.n_orth_cells):,} & {int(r.n_included)} & {r.median_r:.3f} \\\\")
tot_bins = int(sum(ss1.loc[k].n_hd_bins for k, *_ in SEC))
tot_t = sum(ss1.loc[k].t_deconvolve_s for k, *_ in SEC)
write("tab_interface", r"""\begin{table}[H]
\centering
\caption{\textbf{Tumour--stroma interface analysis: sections, fitting and concordance.} Bins, 8-$\mu$m Visium HD bins with at least one count; time, FlashDeconv fitting time with package defaults (total """ + f"{tot_bins:,}" + r""" bins in """ + f"{tot_t / 60:.2f}" + r"""~min); orthogonal units, assigned Xenium cells or CODEX cells; lineages, lineages included in the concordance (mean orthogonal fraction $\ge 0.5\%$, epithelial excluded); median $r$, median over included lineages of the Pearson correlation across 20 distance bands between FlashDeconv and orthogonal gradients (pre-specified criterion, $> 0.7$). Primary sections: CRC P1, P2, P5, COAD and OV-1.}
\label{tab:interface}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrlrrr@{}}
\toprule
Section & Cancer & Bins & Time (s) & Iterations & Orthogonal data & Units & Lineages & Median $r$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
