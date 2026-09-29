"""Control 1 summary: rank among the 13 Spotless methods and paired two-sided Wilcoxon
(54 data sets) vs RCTD and Cell2location (and all competitors) for each layout setting.
Competitor rows and ranking rule as in validation/rerun_final/benchmarks/spotless/analyze_silver.py."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = "/Users/apple/Research/FlashDeconv/results/controls_editor"
CE = "/Users/apple/Research/FlashDeconv/validation/results/comprehensive_evaluation"
METRICS = {"corr": False, "rmse": True, "jsd": True, "aupr": False}
key = ["tissue", "pattern"]

fd = pd.read_csv(f"{R}/c1_fd_aggregate_settings.csv")
perm = fd[fd.setting.str.startswith("perm")].groupby(key)[list(METRICS)].mean().reset_index()
fd = pd.concat([fd, perm.assign(setting="perm_mean")], ignore_index=True)
comp = pd.read_csv(f"{CE}/aggregate_all_benchmarks.csv")
comp = comp[(comp.benchmark == "silver_standard") & (comp.method != "FlashDeconv")]

rows, tests = [], []
for s, g in fd.groupby("setting", sort=False):
    for met, low in METRICS.items():
        means = comp.groupby("method")[met].mean()
        means["FlashDeconv"] = g[met].mean()
        rk = means.rank(ascending=low, method="min")
        rows.append(dict(setting=s, metric=met, fd_mean=g[met].mean(), rank_of_13=int(rk["FlashDeconv"]),
                         rctd_mean=means["rctd"], c2l_mean=means["cell2location"]))
        for m, cg in comp.groupby("method"):
            x = g[key + [met]].merge(cg[key + [met]], on=key, suffixes=("_fd", "_c")).dropna()
            d = x[f"{met}_fd"] - x[f"{met}_c"]
            d = -d if low else d
            tests.append(dict(setting=s, metric=met, comparator=m, n=len(x), fd_better=int((d > 0).sum()),
                              median_improvement=d.median(),
                              p_value=wilcoxon(x[f"{met}_fd"], x[f"{met}_c"]).pvalue))
S, T = pd.DataFrame(rows), pd.DataFrame(tests)
S.to_csv(f"{R}/c1_ranks.csv", index=False)
T.to_csv(f"{R}/c1_wilcoxon.csv", index=False)

# penalty effect vs lambda=0 under each layout (data sets improved, paired P)
base = fd[fd.setting == "lam0"].set_index(key)
eff = []
for s in ["lattice"] + [f"perm{i}" for i in range(5)] + ["perm_mean"]:
    a = fd[fd.setting == s].set_index(key).loc[base.index]
    for met, low in METRICS.items():
        d = (base[met] - a[met]) if low else (a[met] - base[met])
        eff.append(dict(setting=s, metric=met, mean_delta=float(d.mean()), n_improved=int((d > 0).sum()),
                        n_worse=int((d < 0).sum()), p_value=wilcoxon(a[met], base[met]).pvalue))
E = pd.DataFrame(eff)
E.to_csv(f"{R}/c1_penalty_effect_vs_lam0.csv", index=False)

pd.set_option("display.width", 220)
print(S.pivot(index="setting", columns="metric", values=["fd_mean", "rank_of_13"]).round(4).to_string())
t = T[T.comparator.isin(["rctd", "cell2location"])]
print(t.pivot_table(index=["setting"], columns=["comparator", "metric"], values="p_value").to_string(float_format=lambda v: f"{v:.2g}"))
print(t.pivot_table(index=["setting"], columns=["comparator", "metric"], values="fd_better").to_string())
print(E.pivot(index="setting", columns="metric", values=["n_improved", "p_value"]).to_string(float_format=lambda v: f"{v:.2g}"))
print(S[S.setting == "lattice"][["metric", "rctd_mean", "c2l_mean"]].to_string(index=False))
