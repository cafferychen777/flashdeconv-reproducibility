"""Silver Standard summary for the final-package rerun: means, ranks among the 13 methods,
paired two-sided Wilcoxon signed-rank tests (54 datasets) vs every Spotless
competitor, per-cell-type means by abundance stratum, and the Laplacian ablation
(lambda auto vs 0) on the same datasets. Competitor rows are the ones behind the
manuscript (aggregate_all_benchmarks.csv / per_celltype_all_benchmarks.csv, same
metric code)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless"
CE = "/Users/apple/Research/FlashDeconv/validation/results/comprehensive_evaluation"
METRICS = {"corr": False, "rmse": True, "jsd": True, "aupr": False}  # lower-is-better flag

fd = pd.read_csv(f"{R}/fd_aggregate_silver.csv")
comp = pd.read_csv(f"{CE}/aggregate_all_benchmarks.csv")
comp = comp[(comp.benchmark == "silver_standard") & (comp.method != "FlashDeconv")]
key = ["tissue", "pattern"]

summ, tests = [], []
for cfg, g in fd.groupby("config"):
    for met, low in METRICS.items():
        means = comp.groupby("method")[met].mean()
        means["FlashDeconv"] = g[met].mean()
        rank = int(means.rank(ascending=low, method="min")["FlashDeconv"])
        summ.append(dict(config=cfg, metric=met, flash_mean=g[met].mean(), rank_of_13=rank,
                         best_competitor=means.drop("FlashDeconv").idxmin() if low else means.drop("FlashDeconv").idxmax(),
                         best_competitor_mean=means.drop("FlashDeconv").min() if low else means.drop("FlashDeconv").max()))
        for m, cg in comp.groupby("method"):
            x = g[key + [met]].merge(cg[key + [met]], on=key, suffixes=("_fd", "_c")).dropna()
            diff = x[f"{met}_fd"] - x[f"{met}_c"]
            if low:
                diff = -diff
            res = wilcoxon(x[f"{met}_fd"], x[f"{met}_c"], alternative="two-sided")
            tests.append(dict(config=cfg, metric=met, comparator=m, n=len(x),
                              flash_mean=x[f"{met}_fd"].mean(), comparator_mean=x[f"{met}_c"].mean(),
                              median_improvement=diff.median(), flash_better=int((diff > 0).sum()),
                              W=res.statistic, p_value=res.pvalue))
summ = pd.DataFrame(summ)
tests = pd.DataFrame(tests)
summ.to_csv(f"{R}/silver_summary_ranks.csv", index=False)
tests.to_csv(f"{R}/silver_paired_wilcoxon_final.csv", index=False)

# final vs v0.2.0 rerun package defaults (paired, per dataset)
V020 = "/Users/apple/Research/FlashDeconv/results/rerun_v020/benchmarks/spotless"
old = pd.read_csv(f"{V020}/fd_aggregate_silver.csv")
chg = []
for met, low in METRICS.items():
    a = fd[fd.config == "final_default"].set_index(key)[met]
    b = old[old.config == "v020_default"].set_index(key)[met].loc[a.index]
    d = (b - a) if low else (a - b)
    chg.append(dict(config="final_default", metric=met, new=a.mean(), v020=b.mean(), n_better=int((d > 0).sum()),
                    n_worse=int((d < 0).sum()), max_abs_diff=float((a - b).abs().max()),
                    p_value=wilcoxon(a, b).pvalue if (a != b).any() else np.nan))
pd.DataFrame(chg).to_csv(f"{R}/silver_final_vs_v020.csv", index=False)

# per-dataset table (FlashDeconv final + every competitor), for the Fig. 3 panels
wide = []
for met in METRICS:
    c = comp.pivot_table(index=key, columns="method", values=met)
    c["FlashDeconv"] = fd[fd.config == "final_default"].set_index(key)[met]
    c["metric"] = met
    wide.append(c.reset_index())
pd.concat(wide).to_csv(f"{R}/silver_per_dataset_final_vs_competitors.csv", index=False)

# per-cell-type by abundance stratum
pc = pd.read_csv(f"{R}/fd_per_celltype_silver.csv.gz")
cat = (pc.groupby(["config", "category"])[["pearson", "auprc", "precision", "recall", "f1"]]
       .mean().round(4))
n = pc.groupby(["config", "category"]).size().rename("n")
cat = cat.join(n)
cat.to_csv(f"{R}/silver_per_celltype_by_abundance.csv")

# Laplacian ablation (package defaults, lambda auto vs 0), per cell type, paired
a = pc[pc.config == "final_default"].set_index(key + ["cell_type"])
b = pc[pc.config == "final_default_lam0"].set_index(key + ["cell_type"]).loc[a.index]
lap = []
for c in ["rare", "moderate", "abundant", "all"]:
    idx = a.index if c == "all" else a.index[a["category"] == c]
    row = dict(category=c, n=len(idx))
    for met in ["pearson", "auprc", "precision", "recall", "f1"]:
        d = (a.loc[idx, met] - b.loc[idx, met]).dropna()
        row[f"d_{met}"] = d.mean()
        row[f"p_{met}"] = wilcoxon(d).pvalue if (d != 0).sum() > 0 else np.nan
    lap.append(row)
pd.DataFrame(lap).to_csv(f"{R}/silver_laplacian_ablation_percelltype.csv", index=False)

pd.set_option("display.width", 200)
print(summ.pivot(index="config", columns="metric", values=["flash_mean", "rank_of_13"]).round(4))
print(tests[tests.config == "final_default"].pivot(index="comparator", columns="metric", values="p_value"))
print(tests[(tests.config == "final_default") & (tests.metric == "corr")].round(4).to_string(index=False))
print(pd.DataFrame(chg).round(4).to_string(index=False))
print(cat)
print(pd.DataFrame(lap).round(4).to_string(index=False))
