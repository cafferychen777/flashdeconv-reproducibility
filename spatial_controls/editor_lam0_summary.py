"""Spotless silver-standard results at lambda=0 (no spatial information) for the editor
revision: derived tables for the figures and the numbers the text needs.

Inputs
  results/rerun_final/benchmarks/spotless/{fd_aggregate_silver.csv, fd_per_celltype_silver.csv.gz,
      silver_paired_wilcoxon_final.csv, silver_summary_ranks.csv}   (config final_default_lam0)
  results/editor_revision/weighting_lam0/spotless_{acc,pertype}_lam0.csv (weighting_spotless_lam0.py)
  results/editor_revision/marker_scoring_lam0/marker_scoring_comparison.csv (marker_scoring_lam0.py)
  results/controls_editor/c1_*.csv                                   (lattice / permuted layouts)
  results/rerun_final/benchmarks/c2/c2_final_summary_long.csv        (RCTD full UMI>=20)
Outputs (results/editor_revision/)
  silver_per_dataset_lam0_vs_competitors.csv, weighting_lam0/stats_spotless_lam0.csv,
  spotless_lam0_numbers.csv, rctd_umi20_numbers.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final/weighting")
from analyze_weighting_final import paired  # noqa: E402

P = Path("/Users/apple/Research/FlashDeconv")
SP = P / "results/rerun_final/benchmarks/spotless"
CE = P / "validation/results/comprehensive_evaluation"
OUT = P / "results/editor_revision"
WL = OUT / "weighting_lam0"
CFG = "final_default_lam0"
KEY = ["tissue", "pattern"]
METS = ["corr", "rmse", "jsd", "aupr"]
rows = []


def add(metric, value, source):
    rows.append(dict(metric=metric, value=value, source=source))


# ---------------------------------------------------------------- silver, 13 methods
fd = pd.read_csv(SP / "fd_aggregate_silver.csv")
fd = fd[fd.config == CFG]
assert len(fd) == 54 and (fd.lambda_used == 0).all()
comp = pd.read_csv(CE / "aggregate_all_benchmarks.csv")
comp = comp[(comp.benchmark == "silver_standard") & (comp.method != "FlashDeconv")]
wide = []
for met in METS:
    c = comp.pivot_table(index=KEY, columns="method", values=met)
    c["FlashDeconv"] = fd.set_index(KEY)[met]
    c["metric"] = met
    wide.append(c.reset_index())
wide = pd.concat(wide)
wide.to_csv(OUT / "silver_per_dataset_lam0_vs_competitors.csv", index=False)

src = "results/rerun_final/benchmarks/spotless/silver_summary_ranks.csv (config final_default_lam0)"
ss = pd.read_csv(SP / "silver_summary_ranks.csv")
for _, r in ss[ss.config == CFG].iterrows():
    add(f"silver_{r.metric}_mean_flashdeconv_lam0", r.flash_mean, src)
    add(f"silver_{r.metric}_rank_of_13", r.rank_of_13, src)
    add(f"silver_{r.metric}_best_competitor_{r.best_competitor}", r.best_competitor_mean, src)
for m in ["cell2location", "music", "spatialdwls"]:
    for met in METS:
        add(f"silver_{met}_mean_{m}", comp[comp.method == m][met].mean(),
            "validation/results/comprehensive_evaluation/aggregate_all_benchmarks.csv")

src = "results/rerun_final/benchmarks/spotless/silver_paired_wilcoxon_final.csv (config final_default_lam0)"
w = pd.read_csv(SP / "silver_paired_wilcoxon_final.csv")
w = w[w.config == CFG]
for _, r in w.iterrows():
    add(f"wilcoxon_{r.metric}_vs_{r.comparator}_p", r.p_value, src)
    add(f"wilcoxon_{r.metric}_vs_{r.comparator}_fd_better_of_{int(r.n)}", r.flash_better, src)
    add(f"wilcoxon_{r.metric}_vs_{r.comparator}_median_improvement", r.median_improvement, src)
add("wilcoxon_max_p_excluding_rctd_c2l", w[~w.comparator.isin(["rctd", "cell2location"])].p_value.max(), src)

# per tissue (Pearson / JSD, FlashDeconv vs best competitor per tissue)
src = "results/editor_revision/silver_per_dataset_lam0_vs_competitors.csv"
for met in ["corr", "jsd"]:
    t = wide[wide.metric == met].drop(columns=["pattern", "metric"]).groupby("tissue").mean()
    for tis, r in t.iterrows():
        add(f"tissue_{tis}_{met}_flashdeconv_lam0", r["FlashDeconv"], src)
        low = met == "jsd"
        rk = int(r.rank(ascending=low, method="min")["FlashDeconv"])
        add(f"tissue_{tis}_{met}_rank_of_13", rk, src)
        o = r.drop("FlashDeconv")
        add(f"tissue_{tis}_{met}_best_competitor_{o.idxmin() if low else o.idxmax()}",
            o.min() if low else o.max(), src)

# per-cell-type strata
src = "results/rerun_final/benchmarks/spotless/fd_per_celltype_silver.csv.gz (config final_default_lam0)"
pc = pd.read_csv(SP / "fd_per_celltype_silver.csv.gz")
pc = pc[pc.config == CFG]
for cat, g in pc.groupby("category"):
    add(f"percelltype_{cat}_n", len(g), src)
    for m in ["pearson", "auprc"]:
        add(f"percelltype_{cat}_{m}_mean", g[m].mean(), src)
        add(f"percelltype_{cat}_{m}_median", g[m].median(), src)

# ---------------------------------------------------------------- lattice / permuted layouts
src = "results/controls_editor/c1_ranks.csv"
cr = pd.read_csv(P / "results/controls_editor/c1_ranks.csv")
for s, lab in [("lattice", "default_penalty_file_order_lattice")]:
    for _, r in cr[cr.setting == s].iterrows():
        add(f"silver_{r.metric}_mean_{lab}", r.fd_mean, src)
perm = cr[cr.setting.str.startswith("perm")].groupby("metric").fd_mean.agg(["mean", "min", "max"])
for met, r in perm.iterrows():
    add(f"silver_{met}_mean_default_penalty_permuted_positions_mean5", r["mean"], src)
src = "results/controls_editor/c1_penalty_effect_vs_lam0.csv"
pe = pd.read_csv(P / "results/controls_editor/c1_penalty_effect_vs_lam0.csv")
for _, r in pe.iterrows():
    add(f"penalty_effect_{r.setting}_{r.metric}_mean_delta_vs_lam0", r.mean_delta, src)
    add(f"penalty_effect_{r.setting}_{r.metric}_n_improved", r.n_improved, src)
    add(f"penalty_effect_{r.setting}_{r.metric}_p", r.p_value, src)

# ---------------------------------------------------------------- gene weighting
acc = pd.read_csv(WL / "spotless_acc_lam0.csv")
pt = pd.read_csv(WL / "spotless_pertype_lam0.csv")
assert len(acc) == 648 and (acc.lambda_used == 0).all() and acc.converged.all()
st = []
for frac, g in acc.groupby("frac"):
    wv = g.pivot(index="dataset", columns="variant")
    for other in ("UNIFORM", "VAR_REF"):
        for m in ("pearson", "rmse", "jsd", "aupr", "mean_type_pearson", "mean_type_auprc",
                  "rare_pearson", "rare_auprc"):
            st.append(dict(benchmark="spotless", depth=frac, unit="dataset",
                           comparison=f"EXP_LEV vs {other}", metric=m,
                           mean_lev=wv[m]["EXP_LEV"].mean(), mean_other=wv[m][other].mean(),
                           **paired(wv[m]["EXP_LEV"], wv[m][other], m)))
ptr = pt[pt.category == "rare"]
for frac, g in ptr.groupby("frac"):
    wv = g.pivot_table(index=["dataset", "cell_type"], columns="variant", values=["pearson", "auprc"])
    for other in ("UNIFORM", "VAR_REF"):
        for m in ("pearson", "auprc"):
            st.append(dict(benchmark="spotless", depth=frac, unit="rare dataset x type",
                           comparison=f"EXP_LEV vs {other}", metric=f"rare_type_{m}",
                           mean_lev=wv[m]["EXP_LEV"].mean(), mean_other=wv[m][other].mean(),
                           **paired(wv[m]["EXP_LEV"], wv[m][other], m)))
st = pd.DataFrame(st)
st.to_csv(WL / "stats_spotless_lam0.csv", index=False)
src = "results/editor_revision/weighting_lam0/stats_spotless_lam0.csv"
for _, r in st[st.metric.isin(["pearson", "jsd", "aupr", "rmse", "rare_auprc", "rare_type_auprc"])].iterrows():
    tag = f"weighting_{int(round(r.depth * 100))}pct_{r.metric}_{r.comparison.replace(' ', '_')}"
    add(f"{tag}_mean_lev", r.mean_lev, src)
    add(f"{tag}_mean_other", r.mean_other, src)
    add(f"{tag}_win_frac", r.win_frac, src)
    add(f"{tag}_p", r.p_value, src)
for frac, g in acc.groupby("frac"):
    add(f"weighting_{int(round(frac * 100))}pct_median_umi", g.groupby("dataset").median_umi.first().median(),
        "results/editor_revision/weighting_lam0/spotless_acc_lam0.csv")

# ---------------------------------------------------------------- marker scoring
ms = pd.read_csv(OUT / "marker_scoring_lam0/marker_scoring_comparison.csv")
src = "results/editor_revision/marker_scoring_lam0/marker_scoring_comparison.csv"
for meth, g in ms.groupby("method"):
    for cat in ["rare", "moderate", "abundant", "all"]:
        gg = g if cat == "all" else g[g.category == cat]
        for m in ["pearson", "auprc", "rmse"]:
            add(f"marker_{meth}_{cat}_{m}", gg[m].mean(), src)
piv = ms.pivot_table(index=["tissue", "cell_type"], columns="method", values=["pearson", "auprc"])
for meth in ["MarkerScoring_maxgap", "MarkerScoring_wilcoxon"]:
    for m in ["pearson", "auprc"]:
        a, b = piv[(m, "FlashDeconv")], piv[(m, meth)]
        add(f"marker_vs_{meth}_{m}_fd_better_of_{len(a)}", int((a > b).sum()), src)
        add(f"marker_vs_{meth}_{m}_fd_worse", int((a < b).sum()), src)
        add(f"marker_vs_{meth}_{m}_p", wilcoxon(a, b).pvalue, src)
    rare = ms[ms.category == "rare"].pivot_table(index=["tissue", "cell_type"], columns="method",
                                                  values=["pearson", "auprc"])
    for m in ["pearson", "auprc"]:
        add(f"marker_rare_vs_{meth}_{m}_p", wilcoxon(rare[(m, "FlashDeconv")], rare[(m, meth)]).pvalue, src)

# ---------------------------------------------------------------- reference diagnostic (Spotless removal)
for tag, d in [("lam0", OUT / "refdiag_lam0"), ("default_lattice", P / "results/rerun_final/refdiag")]:
    rm = pd.read_csv(d / "spotless_removal_auroc.csv")
    for thr, g in rm.groupby("thr"):
        add(f"refdiag_{tag}_thr{thr}_auroc_median", g.auroc.median(), str(d.relative_to(P)) + "/spotless_removal_auroc.csv")
        add(f"refdiag_{tag}_thr{thr}_auroc_mean", g.auroc.mean(), str(d.relative_to(P)) + "/spotless_removal_auroc.csv")
    fr = pd.read_csv(d / "spotless_complete_reference_flag_rate.csv")
    add(f"refdiag_{tag}_complete_ref_flag_rate_mean", fr.flag_rate_complete_ref.mean(),
        str(d.relative_to(P)) + "/spotless_complete_reference_flag_rate.csv")

# ---------------------------------------------------------------- gold, real coordinates (penalty evidence)
gr = pd.read_csv(OUT / "gold_realxy/fd_aggregate_gold.csv")
gw = gr.pivot_table(index=["benchmark", "tissue"], columns="config", values=METS)
for met in METS:
    a_, b_ = gw[(met, "final_default_realxy")], gw[(met, "final_default_lam0_realxy")]
    better = (a_ > b_) if met in ("corr", "aupr") else (a_ < b_)
    src = "results/editor_revision/gold_realxy/fd_aggregate_gold.csv"
    add(f"gold_realxy_{met}_auto_mean", a_.mean(), src)
    add(f"gold_realxy_{met}_lam0_mean", b_.mean(), src)
    add(f"gold_realxy_{met}_auto_better_of_15", int(better.sum()), src)
    add(f"gold_realxy_{met}_p", wilcoxon(a_, b_).pvalue, src)

pd.DataFrame(rows).to_csv(OUT / "spotless_lam0_numbers.csv", index=False)

# ---------------------------------------------------------------- RCTD full UMI>=20
c2 = pd.read_csv(P / "results/rerun_final/benchmarks/c2/c2_final_summary_long.csv")
src = "results/rerun_final/benchmarks/c2/c2_final_summary_long.csv"
fdm = (c2.method == "FlashDeconv") & (c2["mode"] == "final_default_auto")
r20 = c2[(c2.method == "RCTD") & (c2["mode"] == "full") & (c2.umi_min == 20) & (c2.eval_set == "all")]
out = []
for _, r in r20.sort_values("resolution_um").iterrows():
    f = c2[fdm & (c2.eval_set == "common_full_umi20") & (c2.resolution_um == r.resolution_um)].iloc[0]
    fa = c2[fdm & (c2.eval_set == "all") & (c2.resolution_um == r.resolution_um)].iloc[0]
    b = int(r.resolution_um)
    for k, v in [("n_bins_scored", r.n_bins), ("coverage", r.coverage),
                 ("rctd_pearson", r.pearson_flat), ("fd_pearson", f.pearson_flat),
                 ("rctd_jsd", r.jsd), ("fd_jsd", f.jsd), ("rctd_rmse", r.rmse), ("fd_rmse", f.rmse),
                 ("rctd_type_r", r.type_r), ("fd_type_r", f.type_r),
                 ("fd_all_bins_pearson", fa.pearson_flat), ("fd_all_bins_jsd", fa.jsd)]:
        out.append(dict(metric=f"{b}um_full_umi20_{k}", value=v,
                        source=src + " (RCTD full umi_min=20 eval_set=all; FlashDeconv final_default_auto "
                        "eval_set=common_full_umi20 or all)"))
pd.DataFrame(out).to_csv(OUT / "rctd_umi20_numbers.csv", index=False)

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False, max_rows=None))
print(pd.DataFrame(out).drop(columns="source").to_string(index=False))
