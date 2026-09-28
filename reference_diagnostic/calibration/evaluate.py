"""Steps 2-3: score the pre-registered candidate calibrations (PROTOCOL.md) on V3, V1, V2.

Usage: python evaluate.py v3 | v1 | v2
Writes results/reference_diagnostic_v2/eval_<mode>*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import nullcal as nc  # noqa: E402

OUT = HERE.parents[1] / "results/reference_diagnostic_v2"
Z = nc.Z95


def arrays_from_model(m, sim=True):
    Y, Xbar, P0 = nc.prep(m)
    o10 = nc.core(Y, Xbar, P0, n_em=10)
    o0 = nc.core(Y, Xbar, P0, n_em=0)
    s10 = nc.simulate(Xbar, P0, o10["P"], o10["n"], n_em=10, seed=1) if sim else None
    return o10, o0, s10


def arrays_from_npz(path):
    d = np.load(path)
    n = d["n"].astype(np.float64)
    o10 = {"dev": d["dev_em10"].astype(np.float64), "nV": d["nV_em10"].astype(np.float64), "n": n}
    o0 = {"dev": d["dev_em0"].astype(np.float64), "nV": d["nV_em0"].astype(np.float64), "n": n}
    s10 = {"dev": d["sdev_em10"].astype(np.float64), "nV": d["snV_em10"].astype(np.float64), "n": n}
    A = None
    if "A_indptr" in d.files:
        ip, ix = d["A_indptr"], d["A_indices"]
        A = sparse.csr_matrix((np.ones(len(ix)), ix, ip), shape=(len(n), len(n)))
    return o10, o0, s10, A


def decile_rows(scores, depth, meta):
    rows = []
    for name, s in scores.items():
        for r in nc.summarize_by_depth(s, depth):
            r.update(meta)
            r["cand"] = name
            rows.append(r)
    return rows


def v3():
    rows = []
    for ds in range(1, 7):
        for pat in data.PATTERNS:
            d = data.spotless(ds, pat)
            if d is None:
                continue
            Yc, Xf, crd, cts, T = d
            m = data.fit(Yc, Xf, crd, cts)
            o10, o0, s10 = arrays_from_model(m)
            sc = nc.candidates(o10, o0, s10)
            rows += decile_rows(sc, o10["n"], {"set": "spotless", "fit": f"{ds}_{pat}", "pooled": False})
    for res in (8, 16):
        o10, o0, s10, A = arrays_from_npz(OUT / f"diag_c2_{res}um.npz")
        for pooled in (False, True):
            AA = A if pooled else None
            sc = nc.candidates(o10, o0, s10, AA)
            depth = o10["n"] if not pooled else A @ o10["n"]
            rows += decile_rows(sc, depth, {"set": f"c2_{res}um", "fit": "full", "pooled": pooled})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eval_v3_deciles.csv", index=False)
    # Spotless: pool the 24 fits per decile (bin-weighted)
    df["n_flag"] = df.flag_rate * df.n_bins
    g = df.groupby(["set", "pooled", "cand", "decile"]).agg(n_bins=("n_bins", "sum"), n_flag=("n_flag", "sum"),
                                                          median_depth=("median_depth", "median")).reset_index()
    g["fpr"] = g.n_flag / g.n_bins
    g.to_csv(OUT / "eval_v3_fpr_by_decile.csv", index=False)
    s = g.groupby(["set", "pooled", "cand"]).agg(fpr_all=("n_flag", "sum"), nb=("n_bins", "sum"),
                                                fpr_min=("fpr", "min"), fpr_max=("fpr", "max"),
                                                maxdev=("fpr", lambda x: np.max(np.abs(x - 0.05)))).reset_index()
    s["fpr_all"] = s.fpr_all / s.nb
    s.drop(columns="nb").to_csv(OUT / "eval_v3_summary.csv", index=False)
    print(s.round(4).to_string())


def v1():
    rows = []
    for ds in range(1, 7):
        for pat in data.PATTERNS:
            d = data.spotless(ds, pat)
            if d is None:
                continue
            Yc, Xf, crd, cts, T = d
            mean_prop = T.mean(0)
            for k, ct in enumerate(cts):
                if T[:, k].max() < 0.3:
                    continue
                keep = [j for j in range(len(cts)) if j != k]
                m = data.fit(Yc, Xf[keep], crd, [cts[j] for j in keep])
                o10, o0, s10 = arrays_from_model(m)
                sc = nc.candidates(o10, o0, s10)
                neg = T[:, k] < 0.01
                cls = "rare" if mean_prop[k] < 0.05 else ("moderate" if mean_prop[k] <= 0.15 else "abundant")
                for thr in (0.1, 0.3, 0.5):
                    pos = T[:, k] > thr
                    ok = pos.sum() >= 5 and neg.sum() >= 5
                    for name, s in sc.items():
                        yv = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
                        sv = np.r_[s[pos], s[neg]]
                        rows.append({"ds": ds, "pattern": pat, "removed": ct, "class": cls,
                                     "mean_prop": mean_prop[k], "thr": thr, "cand": name,
                                     "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                                     "auroc": roc_auc_score(yv, sv) if ok else np.nan,
                                     "auprc": average_precision_score(yv, sv) if ok else np.nan,
                                     "flag_pos": (s[pos] > Z).mean() if pos.any() else np.nan,
                                     "flag_neg": (s[neg] > Z).mean() if neg.any() else np.nan})
            print(ds, pat, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eval_v1_spotless.csv", index=False)
    print(df[df.thr == 0.3].groupby("cand")[["auroc", "flag_pos", "flag_neg"]].median().round(3))


def v2():
    rows = []
    for ref in ("haber", "composite"):
        o10, o0, s10, A = arrays_from_npz(OUT / f"diag_intestine_{ref}.npz")
        if ref == "haber":
            _, _, _, _, _, reg, _ = data.intestine(ref)
        for pooled in (False, True):
            sc = nc.candidates(o10, o0, s10, A if pooled else None)
            clean = reg["epithelium"]
            for name, s in sc.items():
                r = {"ref": ref, "pooled": pooled, "cand": name, "flag_all": (s > Z).mean()}
                for rn in ("follicle", "muscle", "epi_low", "epithelium"):
                    r[f"flag_{rn}"] = (s[reg[rn]] > Z).mean()
                for rn in ("follicle", "muscle", "mask_full"):
                    pos = reg[rn]
                    yv = np.r_[np.ones(pos.sum()), np.zeros(clean.sum())]
                    r[f"auroc_{rn}"] = roc_auc_score(yv, np.r_[s[pos], s[clean]])
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eval_v2_intestine.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    {"v3": v3, "v1": v1, "v2": v2}[sys.argv[1]]()
