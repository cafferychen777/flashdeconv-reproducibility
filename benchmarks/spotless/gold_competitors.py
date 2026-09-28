"""Spotless competitor metrics on the Gold Standard, recomputed with the same metric
code (comprehensive_per_celltype_evaluation) and the same ground truth / type
matching used for the FlashDeconv rows (matched columns, renormalised)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation")
import comprehensive_per_celltype_evaluation as ce  # noqa: E402

D = ce.DATA_DIR
P = ce.SPOTLESS_DIR
OUT = "/Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless"
GOLD_NPZ = "/Users/apple/Research/FlashDeconv/results/rerun_v020/benchmarks/_inputs/gold"


def read_pred(f):
    L = open(f).read().strip().split("\n")
    return np.array([[float(x) for x in l.split("\t")] for l in L[1:]]), L[0].split("\t")


def evaluate(pred, ptypes, gt):
    m = ce.align_celltype_names(ptypes, list(gt.columns))
    cols = [j for j, t in enumerate(ptypes) if t in m]
    tt = [m[ptypes[j]] for j in cols]
    pa = np.clip(pred[:, cols], 0, None)
    rs = pa.sum(1, keepdims=True); rs[rs == 0] = 1
    a = ce.compute_aggregate_metrics(pa / rs, gt[tt], tt)
    a.pop("jsd_contributions")
    return a, len(tt)


rows = []
for region in ["cortex_svz", "ob"]:
    for fov in range(7):
        gt = pd.read_csv(f"{D}/Eng2019_{region}_fov{fov}_proportions.csv", index_col=0).drop(columns="spot_no")
        for meth in ce.SPOTLESS_METHODS:
            f = f"{P}/Eng2019_{region}/proportions_{meth}_Eng2019_{region}_fov{fov}"
            if not os.path.exists(f):
                continue
            pred, pt = read_pred(f)
            if pred.shape[0] != len(gt):
                print("row mismatch", f); continue
            a, n = evaluate(pred, pt, gt)
            rows.append({**a, "method": meth, "benchmark": f"seqfish_{region}", "tissue": f"fov{fov}", "n_types_eval": n})
z = np.load(f"{GOLD_NPZ}/gold_Wang2018_visp_rep0410.npz", allow_pickle=True)
gt = pd.DataFrame(z["gt"], columns=np.asarray(z["gt_cols"]).astype(str))
for meth in ce.SPOTLESS_METHODS:
    f = f"{P}/Wang2018_visp/proportions_{meth}_Wang2018_visp_rep0410_12celltypes"
    if not os.path.exists(f):
        continue
    pred, pt = read_pred(f)
    if pred.shape[0] != len(gt):
        print("row mismatch", f); continue
    a, n = evaluate(pred, pt, gt)
    rows.append({**a, "method": meth, "benchmark": "starmap", "tissue": "starmap", "n_types_eval": n})
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/gold_competitors_recomputed.csv", index=False)
print(df.groupby(["benchmark", "method"])[["corr", "rmse", "jsd", "aupr", "n_types_eval"]].mean().round(3))
