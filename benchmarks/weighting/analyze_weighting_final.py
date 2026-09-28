"""Paired statistics for the final gene-weighting comparison.

Reads results/rerun_final/weighting/{spotless,xenium}_{acc,pertype}.csv and writes
stats_spotless.csv, stats_xenium.csv (EXP_LEV vs UNIFORM / VAR_REF).

diff = EXP_LEV - other (raw metric units). 'lev_better' orients by metric direction
(higher is better for Pearson/AP/AUPR, lower for RMSE/JSD). Rank-biserial is the
matched-pairs effect size (W+ - W-) / (W+ + W-) over non-zero |diff| ranks, signed
so that positive = EXP_LEV better.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

R = Path("/Users/apple/Research/FlashDeconv/results/rerun_final/weighting")
LOWER = {"rmse", "jsd"}
rng = np.random.default_rng(0)


def paired(a, b, metric, n_boot=10000):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    s = -1.0 if metric in LOWER else 1.0
    good = s * d  # positive = EXP_LEV better
    nz = good[good != 0]
    rk = rankdata(np.abs(nz))
    rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if len(nz) else np.nan
    p = wilcoxon(d).pvalue if len(nz) > 0 else np.nan
    bs = np.median(d[rng.integers(0, len(d), (n_boot, len(d)))], axis=1)
    return dict(n=len(d), median_diff=float(np.median(d)),
                ci_lo=float(np.quantile(bs, 0.025)), ci_hi=float(np.quantile(bs, 0.975)),
                mean_diff=float(np.mean(d)), win_frac=float(np.mean(good > 0)),
                rank_biserial=float(rb), p_value=float(p))


def main():
    rows = []
    acc = pd.read_csv(R / "spotless_acc.csv")
    for frac, g in acc.groupby("frac"):
        w = g.pivot(index="dataset", columns="variant")
        for other in ("UNIFORM", "VAR_REF"):
            for m in ("pearson", "rmse", "jsd", "aupr", "mean_type_pearson",
                      "mean_type_auprc", "rare_pearson", "rare_auprc"):
                rows.append(dict(benchmark="spotless", depth=frac, unit="dataset",
                                 comparison=f"EXP_LEV vs {other}", metric=m,
                                 mean_lev=w[m]["EXP_LEV"].mean(), mean_other=w[m][other].mean(),
                                 **paired(w[m]["EXP_LEV"], w[m][other], m)))
    # Rare types pooled across datasets (unit = dataset x cell type, category == rare)
    pt = pd.read_csv(R / "spotless_pertype.csv")
    pt = pt[pt["category"] == "rare"]
    for frac, g in pt.groupby("frac"):
        w = g.pivot_table(index=["dataset", "cell_type"], columns="variant",
                          values=["pearson", "auprc"])
        for other in ("UNIFORM", "VAR_REF"):
            for m in ("pearson", "auprc"):
                rows.append(dict(benchmark="spotless", depth=frac, unit="rare dataset x type",
                                 comparison=f"EXP_LEV vs {other}", metric=f"rare_type_{m}",
                                 mean_lev=w[m]["EXP_LEV"].mean(), mean_other=w[m][other].mean(),
                                 **paired(w[m]["EXP_LEV"], w[m][other], m)))
    pd.DataFrame(rows).to_csv(R / "stats_spotless.csv", index=False)

    rows = []
    xa = pd.read_csv(R / "xenium_acc.csv")
    xp = pd.read_csv(R / "xenium_pertype.csv")
    for res, g in xp.groupby("res_um"):
        a = xa[xa["res_um"] == res].set_index("variant")
        for sub, gg in (("all types", g), ("rare types", g[g["category"] == "rare"])):
            w = gg.pivot(index="cell_type", columns="variant")
            for other in ("UNIFORM", "VAR_REF"):
                for m in ("pearson", "ap", "rmse"):
                    rows.append(dict(benchmark="xenium", res_um=res, unit=f"cell type ({sub})",
                                     comparison=f"EXP_LEV vs {other}", metric=f"type_{m}",
                                     mean_lev=w[m]["EXP_LEV"].mean(),
                                     mean_other=w[m][other].mean(),
                                     **paired(w[m]["EXP_LEV"], w[m][other], m)))
        for other in ("UNIFORM", "VAR_REF"):
            for m in ("pearson", "mean_type_pearson", "rmse", "jsd", "ap", "mean_type_ap",
                      "auprc_trapz", "rare_pearson", "rare_ap"):
                rows.append(dict(benchmark="xenium", res_um=res, unit="global (single fit)",
                                 comparison=f"EXP_LEV vs {other}", metric=m,
                                 mean_lev=a.loc["EXP_LEV", m], mean_other=a.loc[other, m],
                                 n=1, median_diff=a.loc["EXP_LEV", m] - a.loc[other, m]))
    pd.DataFrame(rows).to_csv(R / "stats_xenium.csv", index=False)


if __name__ == "__main__":
    main()
