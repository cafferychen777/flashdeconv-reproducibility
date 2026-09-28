"""Re-validation with the updated package function (null='auto' vs 'left_half'), package defaults.
  intestine : Visium HD mouse small intestine 8 um, Haber vs composite reference (V2)
  spotless  : complete-reference flag rates per depth decile (V3) and controlled removal (V1)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import flashdeconv  # noqa: E402
import nullcal as nc  # noqa: E402

OUT = HERE.parents[1] / "results/reference_diagnostic_v2"
METHODS = ("auto", "left_half")


def intestine():
    rows, perbin = [], {}
    for ref in ("haber", "composite"):
        Y, X, crd, ct, genes, reg, sp = data.intestine(ref)
        m = data.fit(Y, X, crd, ct)
        clean = reg["epithelium"]
        for meth in METHODS:
            s = flashdeconv.reference_fit_scores(m, null=meth)
            perbin[f"{ref}_{meth}_score_pooled"] = s["score_pooled"].astype(np.float32)
            perbin[f"{ref}_{meth}_flag_pooled"] = s["flag_pooled"]
            for key in ("score", "score_pooled"):
                f = s["flag" + key[5:]]
                r = {"ref": ref, "null": meth, "key": key, "null_used": s["null"][key]["method"],
                     "flag_all": f.mean()}
                for rn in ("follicle", "muscle", "epi_low", "epithelium"):
                    r[f"flag_{rn}"] = f[reg[rn]].mean()
                for rn in ("follicle", "muscle"):
                    pos = reg[rn]
                    r[f"auroc_{rn}"] = roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(clean.sum())],
                                                     np.r_[s[key][pos], s[key][clean]])
                rows.append(r)
        perbin["n_umi_" + ref] = s["n_umi"].astype(np.float32)
        print(pd.DataFrame(rows).round(3).to_string(), flush=True)
    df = pd.DataFrame(perbin)
    df["x_px"], df["y_px"] = sp[:, 0].astype(np.float32), sp[:, 1].astype(np.float32)
    for rn in ("follicle", "muscle", "epi_low", "epithelium"):
        df[rn] = reg[rn]
    df.to_parquet(OUT / "pkg_intestine_perbin.parquet", index=False)
    pd.DataFrame(rows).to_csv(OUT / "pkg_intestine_region_flags.csv", index=False)


def spotless():
    v3, v1 = [], []
    for ds in range(1, 7):
        for pat in data.PATTERNS:
            d = data.spotless(ds, pat)
            if d is None:
                continue
            Yc, Xf, crd, cts, T = d
            m = data.fit(Yc, Xf, crd, cts)
            for meth in METHODS:
                s = flashdeconv.reference_fit_scores(m, pool=False, null=meth)
                for r in nc.summarize_by_depth(s["score"], s["n_umi"]):
                    r.update({"ds": ds, "pattern": pat, "null": meth})
                    v3.append(r)
            mean_prop = T.mean(0)
            for k, ct in enumerate(cts):
                if T[:, k].max() < 0.3:
                    continue
                keep = [j for j in range(len(cts)) if j != k]
                mm = data.fit(Yc, Xf[keep], crd, [cts[j] for j in keep])
                pos, neg = T[:, k] > 0.3, T[:, k] < 0.01
                if pos.sum() < 5 or neg.sum() < 5:
                    continue
                for meth in METHODS:
                    s = flashdeconv.reference_fit_scores(mm, pool=False, null=meth)
                    v1.append({"ds": ds, "pattern": pat, "removed": ct, "null": meth,
                               "mean_prop": mean_prop[k],
                               "auroc": roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())],
                                                      np.r_[s["score"][pos], s["score"][neg]]),
                               "flag_pos": s["flag"][pos].mean(), "flag_neg": s["flag"][neg].mean()})
            print(ds, pat, flush=True)
    pd.DataFrame(v3).to_csv(OUT / "pkg_spotless_v3_deciles.csv", index=False)
    pd.DataFrame(v1).to_csv(OUT / "pkg_spotless_v1.csv", index=False)


def c2():
    rows = []
    for res in (8, 16):
        Y, X, C, cts, T = data.c2(res)
        m = data.fit(Y, X, C, cts)
        for meth in METHODS:
            s = flashdeconv.reference_fit_scores(m, null=meth)
            for key, depth in (("score", s["n_umi"]), ("score_pooled", None)):
                if depth is None:
                    A = nc.pool_matrix(m)
                    depth = A @ s["n_umi"]
                for r in nc.summarize_by_depth(s[key], depth):
                    r.update({"set": f"c2_{res}um", "key": key, "null": meth,
                              "null_used": s["null"][key]["method"]})
                    rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "pkg_c2_v3_deciles.csv", index=False)


if __name__ == "__main__":
    {"intestine": intestine, "spotless": spotless, "c2": c2}[sys.argv[1]]()
