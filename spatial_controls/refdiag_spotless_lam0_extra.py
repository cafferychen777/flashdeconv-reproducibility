"""Spotless reference-diagnostic complements at lambda=0 (silver-standard pseudo-spots have no spatial
layout). Same loops as validation/reference_diagnostic_v2/pkg_local.py::spotless (depth deciles) and
validation/rerun_final/refdiag_extra/naming_nulls.py::spotless (null comparison + removed-type naming),
with FlashDeconv(lambda_spatial=0). Outputs to results/editor_revision/refdiag_lam0:
  spotless_v3_deciles.csv          per fit x null x depth decile (complete reference)
  spotless_fpr_by_depth_decile.csv pooled over the 24 fits (bin-weighted), as summarize.py
  spotless_null_comparison.csv     complete-reference flag rate under left_half / central / auto
  spotless_removal_naming.csv      rank of removed type among the full reference's types
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/apple/Research/FlashDeconv")
sys.path.insert(0, str(ROOT / "validation/reference_diagnostic_v2"))
import data  # noqa: E402
import nullcal as nc  # noqa: E402
import flashdeconv  # noqa: E402
from flashdeconv.core.refcheck import unexplained_genes, suggest_missing_types  # noqa: E402

OUT = ROOT / "results/editor_revision/refdiag_lam0"
NULLS = ("left_half", "central", "auto")


def fit(*a):
    return data.fit(*a, lambda_spatial=0.0)


def main():
    dec, cref, name = [], [], []
    for ds in range(1, 7):
        for pat in data.PATTERNS:
            d = data.spotless(ds, pat)
            if d is None:
                continue
            Yc, Xf, crd, cts, T = d
            gnames = np.array([f"g{j}" for j in range(Yc.shape[1])])
            m = fit(Yc, Xf, crd, cts)
            assert m.lambda_used_ == 0 if hasattr(m, "lambda_used_") else True
            for nl in NULLS:
                s = flashdeconv.reference_fit_scores(m, pool=False, null=nl)
                cref.append({"set": "spotless", "ds": ds, "pattern": pat, "key": "score", "null": nl,
                             "null_used": s["null"]["score"]["method"], "n_bins": len(s["flag"]),
                             "flag_rate": s["flag"].mean()})
                for r in nc.summarize_by_depth(s["score"], s["n_umi"]):
                    r.update({"ds": ds, "pattern": pat, "null": nl})
                    dec.append(r)
            for k, ct in enumerate(cts):
                if T[:, k].max() < 0.3:
                    continue
                pos, neg = T[:, k] > 0.3, T[:, k] < 0.01
                if pos.sum() < 5 or neg.sum() < 5:
                    continue
                keep = [j for j in range(len(cts)) if j != k]
                mm = fit(Yc, Xf[keep], crd, [cts[j] for j in keep])
                for nl in ("left_half", "auto"):
                    s = flashdeconv.reference_fit_scores(mm, pool=False, null=nl)
                    fl = s["flag"]
                    r = {"ds": ds, "pattern": pat, "removed": ct, "null": nl, "K": len(cts),
                         "n_flag": int(fl.sum()), "rank": np.nan}
                    if 0 < fl.sum() < len(fl):
                        g = unexplained_genes(mm, fl, gene_names=gnames)
                        rk = suggest_missing_types(g, Xf, gnames, cts)
                        r["rank"] = int(np.where(rk["type"] == ct)[0][0]) + 1
                        r["top_type"] = rk["type"][0]
                    name.append(r)
            print(ds, pat, flush=True)
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT / "spotless_v3_deciles.csv", index=False)
    dec["n_flag"] = dec.flag_rate * dec.n_bins
    g = dec.groupby(["null", "decile"]).agg(n_bins=("n_bins", "sum"), n_flag=("n_flag", "sum"),
                                            median_depth=("median_depth", "median")).reset_index()
    g["fpr"] = g.n_flag / g.n_bins
    g["set"], g["key"] = "spotless", "score"
    g[["set", "key", "null", "decile", "median_depth", "n_bins", "fpr"]].to_csv(
        OUT / "spotless_fpr_by_depth_decile.csv", index=False)
    pd.DataFrame(cref).to_csv(OUT / "spotless_null_comparison.csv", index=False)
    name = pd.DataFrame(name)
    name.to_csv(OUT / "spotless_removal_naming.csv", index=False)
    print(g.pivot_table(index="null", columns="decile", values="fpr").round(4).to_string())
    print(name.groupby("null")["rank"].agg(n="size", top1=lambda x: (x == 1).mean(),
                                           top3=lambda x: (x <= 3).mean()))


if __name__ == "__main__":
    main()
