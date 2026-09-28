"""Liver case study: endothelial signature collinearity and per-type JSD contributions.

Uses the liver reference signature of the Spotless case study and the final FlashDeconv
predictions (results/rerun_final/benchmarks/spotless/liver_case_study.csv, final_default).
Output: results/rerun_final/benchmarks/spotless/liver_collinearity_{cosine,jsd_contrib}.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation")
import benchmark_liver as bl  # noqa: E402

OUT = "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless"
z = np.load(f"{bl.DATA_DIR}/liver_signature.npz", allow_pickle=True)
types = [str(t) for t in z["celltypes"]]
X = np.asarray(z["signature"], float)
Xl = np.log1p(1e4 * X / X.sum(1, keepdims=True))
rows = []
for space, M in [("counts", X), ("lognorm", Xl)]:
    U = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = U @ U.T
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            rows.append(dict(space=space, type1=types[i], type2=types[j], cosine=C[i, j]))
cos = pd.DataFrame(rows).sort_values(["space", "cosine"], ascending=[True, False])
cos.to_csv(f"{OUT}/liver_collinearity_cosine.csv", index=False)

cs = pd.read_csv(f"{OUT}/liver_case_study.csv")
cs = cs[cs.config == "final_default"]
pcols = [c for c in cs.columns if c.startswith("prop_")]
names = [c[5:] for c in pcols]
gt = np.array([bl.SNRNASEQ_PROPORTIONS.get(n, 0) for n in names], float)
recs = []
for _, r in cs.iterrows():
    q = r[pcols].to_numpy(float)
    m = (gt + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 0.5 * (np.where(gt > 0, gt * np.log(gt / m), 0) + np.where(q > 0, q * np.log(q / m), 0))
    for n, g_, q_, c_ in zip(names, gt, q, c):
        recs.append(dict(sample=r["sample"], cell_type=n, truth=g_, predicted=q_, jsd_contribution=c_))
jd = pd.DataFrame(recs)
summ = jd.groupby("cell_type")[["truth", "predicted", "jsd_contribution"]].mean()
summ.to_csv(f"{OUT}/liver_collinearity_jsd_contrib.csv")
print(cos.head(8).to_string(index=False))
print(summ.round(4).to_string())
print("total JSD (mean over slides, natural log):", jd.groupby("sample").jsd_contribution.sum().mean())
print("aupr portal/central mean:", cs.aupr_portal.mean(), cs.aupr_central.mean(), cs.aupr_mean.mean(), "jsd", cs.jsd.mean())
