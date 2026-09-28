"""Gold / liver / stability / melanoma summaries with ranks among the 12 Spotless
competitors (13 methods incl. FlashDeconv)."""
import numpy as np
import pandas as pd

R = "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless"
V = "/Users/apple/Research/FlashDeconv/validation"
LOW = {"corr": False, "rmse": True, "jsd": True, "aupr": False}


def rank(value, others, low):
    s = pd.concat([others, pd.Series({"FlashDeconv": value})])
    return int(s.rank(ascending=low, method="min")["FlashDeconv"])


rows = []
# Gold: competitors recomputed with the same metric code / GT / type matching
fd = pd.read_csv(f"{R}/fd_aggregate_gold.csv")
cp = pd.read_csv(f"{R}/gold_competitors_recomputed.csv")
for b in ["seqfish_cortex_svz", "seqfish_ob", "starmap"]:
    cm = cp[cp.benchmark == b].groupby("method")[list(LOW)].mean()
    for cfg, g in fd[fd.benchmark == b].groupby("config"):
        for met, low in LOW.items():
            rows.append(dict(benchmark=b, config=cfg, metric=met, flash=g[met].mean(),
                             rank_of_13=rank(g[met].mean(), cm[met], low),
                             best_competitor=cm[met].idxmin() if low else cm[met].idxmax(),
                             best_value=cm[met].min() if low else cm[met].max()))
# Liver: Spotless 'all'-digest values
lv = pd.read_csv(f"{R}/liver_case_study.csv").groupby("config")[["jsd", "aupr_mean"]].mean()
sp = pd.read_csv(f"{V}/spotless_liver_all_methods.csv")
sp = sp[sp.digest == "all"].pivot(index="method", columns="metric", values="value")
for cfg, r in lv.iterrows():
    for met, col, low in [("jsd", "jsd", True), ("aupr", "aupr_mean", False)]:
        rows.append(dict(benchmark="liver", config=cfg, metric=met, flash=r[col],
                         rank_of_13=rank(r[col], sp[met], low),
                         best_competitor=sp[met].idxmin() if low else sp[met].idxmax(),
                         best_value=sp[met].min() if low else sp[met].max()))
st = pd.read_csv(f"{R}/liver_stability.csv").groupby("config").jsd.mean()
ss = pd.read_csv(f"{V}/spotless_liver_ref_sensitivity.csv").groupby("method").jsd.mean()
for cfg, v in st.items():
    rows.append(dict(benchmark="liver_ref_stability", config=cfg, metric="jsd", flash=v,
                     rank_of_13=rank(v, ss, True), best_competitor=ss.idxmin(), best_value=ss.min()))
mel = pd.read_csv(f"{R}/melanoma_fixed.csv").groupby("config")[["jsd", "melanocytic", "tcell"]].mean()
ms = pd.read_csv(f"{V}/melanoma_analysis/spotless_melanoma_jsd.csv").set_index("method").jsd
for cfg, r in mel.iterrows():
    rows.append(dict(benchmark="melanoma", config=cfg, metric="jsd", flash=r.jsd,
                     rank_of_13=rank(r.jsd, ms, True), best_competitor=ms.idxmin(), best_value=ms.min(),
                     note=f"melanocytic={r.melanocytic:.3f}; T={r.tcell:.3f}"))
try:
    grid = pd.read_csv(f"{R}/melanoma_grid_expected.csv")
    b = grid.loc[grid.jsd.idxmin()]
    rows.append(dict(benchmark="melanoma", config="final_best_of_same_108_grid", metric="jsd", flash=b.jsd,
                     rank_of_13=rank(b.jsd, ms, True), best_competitor=ms.idxmin(), best_value=ms.min(),
                     note=str(b[["n_hvg", "n_markers_per_type", "lambda_spatial", "rho_sparsity", "preprocess", "melanocytic"]].to_dict())))
except FileNotFoundError:
    pass
out = pd.DataFrame(rows)
out.to_csv(f"{R}/cases_summary_ranks.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
