"""Final-package rerun of the Spotless controlled-removal validation of the incomplete-reference
diagnostic (package flashdeconv.reference_fit_scores, defaults).

Spotless silver datasets 1-6, patterns 2/4/7/8, rep1 (as validation/reference_diagnostic/
v1_spotless.py). For each type with max true proportion >= 0.3, refit without it and score bins.
Positives: removed-type true proportion > thr (0.1/0.3/0.5); negatives: < 0.01.
Grid coordinates are synthetic, so the unpooled score is used. Distinctness of each removed type
as in validation/reference_diagnostic/distinctness.py (1 - uncentred R^2 of NNLS of its
log1p-CP10k profile onto the remaining profiles).
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.optimize import nnls
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/Users/apple/Research/FlashDeconv")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/rerun_final"))
import fdfinal  # noqa: E402,F401
import flashdeconv  # noqa: E402
from flashdeconv import FlashDeconv  # noqa: E402

CONV = ROOT / "validation/benchmark_data/converted"
CACHE = ROOT / "results/reference_diagnostic/cache"  # per-dataset mean reference signatures
OUT = ROOT / "results/rerun_final/refdiag"
PATTERNS = [2, 4, 7, 8]
THRS = [0.1, 0.3, 0.5]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def distinct(X, k):
    L = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
    keep = [j for j in range(len(X)) if j != k]
    _, r = nnls(L[keep].T, L[k])
    return r ** 2 / (L[k] @ L[k])


def fit(Y, X, coords, names, ctx):
    os.environ["FD_CONTEXT"] = ctx
    m = FlashDeconv(verbose=False)
    m.fit(Y, X, coords, cell_type_names=np.array(names))
    return m


def run_ds(ds):
    d = np.load(CACHE / f"sig_{ds}.npz", allow_pickle=True)
    Xref, rtypes, rgenes = d["X"], list(d["types"]), list(d["genes"])
    rmap = {norm(t): i for i, t in enumerate(rtypes)}
    rows, v3 = [], []
    for pat in PATTERNS:
        pre = CONV / f"silver_{ds}_{pat}"
        if not Path(f"{pre}_counts.mtx").exists():
            continue
        Y = mmread(f"{pre}_counts.mtx").T.tocsr().astype(np.float64)
        genes = pd.read_csv(f"{pre}_genes.txt", header=None)[0].astype(str).values
        props = pd.read_csv(f"{pre}_proportions.csv", index_col=0).select_dtypes(include=[np.number])
        n = Y.shape[0]
        g = int(np.ceil(np.sqrt(n)))
        coords = np.array([[i % g, i // g] for i in range(n)], dtype=float)
        cts = [c for c in props.columns if norm(c) in rmap]
        idx = [rmap[norm(c)] for c in cts]
        rg = set(rgenes)
        common = [x for x in genes if x in rg]
        Yc = Y[:, pd.Index(genes).get_indexer(common)].tocsr()
        Xfull = Xref[np.ix_(idx, pd.Index(rgenes).get_indexer(common))]
        T = props[cts].to_numpy()
        mean_prop = T.mean(0)
        m = fit(Yc, Xfull, coords, cts, f"spotless_ds{ds}_p{pat}_full")
        s = flashdeconv.reference_fit_scores(m, pool=False)
        v3.append({"ds": ds, "pattern": pat, "n_bins": n, "K": len(cts),
                   "flag_rate_complete_ref": s["flag"].mean(),
                   "n_iterations": m.info_.get("n_iterations"), "converged": m.info_.get("converged")})
        for k, ct in enumerate(cts):
            if T[:, k].max() < 0.3:
                continue
            keep = [j for j in range(len(cts)) if j != k]
            m = fit(Yc, Xfull[keep], coords, [cts[j] for j in keep], f"spotless_ds{ds}_p{pat}_minus_{ct}")
            s = flashdeconv.reference_fit_scores(m, pool=False)
            neg = T[:, k] < 0.01
            cls = "rare" if mean_prop[k] < 0.05 else ("moderate" if mean_prop[k] <= 0.15 else "abundant")
            for thr in THRS:
                pos = T[:, k] > thr
                ok = pos.sum() >= 5 and neg.sum() >= 5
                yv = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
                sv = np.r_[s["score"][pos], s["score"][neg]]
                rows.append({"ds": ds, "pattern": pat, "removed": ct, "class": cls,
                             "mean_prop": mean_prop[k], "distinct": distinct(Xfull, k), "thr": thr,
                             "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                             "auroc": roc_auc_score(yv, sv) if ok else np.nan,
                             "auprc": average_precision_score(yv, sv) if ok else np.nan,
                             "flag_rate_pos": s["flag"][pos].mean() if pos.any() else np.nan,
                             "flag_rate_neg": s["flag"][neg].mean() if neg.any() else np.nan,
                             "n_iterations": m.info_.get("n_iterations"),
                             "converged": m.info_.get("converged")})
        print(f"  ds{ds} p{pat}: {len(cts)} types, {Yc.shape}", flush=True)
    return rows, v3


if __name__ == "__main__":
    rows, v3 = [], []
    for ds in range(1, 7):
        t = time.time()
        r, v = run_ds(ds)
        rows += r
        v3 += v
        print(f"ds{ds} finished in {time.time() - t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "spotless_removal_auroc.csv", index=False)
    pd.DataFrame(v3).to_csv(OUT / "spotless_complete_reference_flag_rate.csv", index=False)
