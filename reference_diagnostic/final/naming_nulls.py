"""Final-package (v0.2.0) complement for the supplementary reference-diagnostic figure.
spotless: complete-reference flag rate under left_half / central / auto; controlled removal
          naming (rank of removed type among the full reference's types by suggest_missing_types).
c2      : complete-reference flag rate (pooled + unpooled) under the three nulls.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/apple/Research/FlashDeconv")
sys.path.insert(0, str(ROOT / "validation/reference_diagnostic_v2"))
import data  # noqa
import flashdeconv
from flashdeconv.core.refcheck import unexplained_genes, suggest_missing_types

OUT = ROOT / "results/rerun_final/refdiag"
NULLS = ("left_half", "central", "auto")


def spotless():
    cref, name = [], []
    for ds in range(1, 7):
        for pat in data.PATTERNS:
            d = data.spotless(ds, pat)
            if d is None:
                continue
            Yc, Xf, crd, cts, T = d
            G = Yc.shape[1]
            gnames = np.array([f"g{j}" for j in range(G)])
            m = data.fit(Yc, Xf, crd, cts)
            for nl in NULLS:
                s = flashdeconv.reference_fit_scores(m, pool=False, null=nl)
                cref.append({"set": "spotless", "ds": ds, "pattern": pat, "key": "score", "null": nl,
                             "null_used": s["null"]["score"]["method"], "n_bins": len(s["flag"]),
                             "flag_rate": s["flag"].mean()})
            for k, ct in enumerate(cts):
                if T[:, k].max() < 0.3:
                    continue
                pos, neg = T[:, k] > 0.3, T[:, k] < 0.01
                if pos.sum() < 5 or neg.sum() < 5:
                    continue
                keep = [j for j in range(len(cts)) if j != k]
                mm = data.fit(Yc, Xf[keep], crd, [cts[j] for j in keep])
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
    return pd.DataFrame(cref), pd.DataFrame(name)


def c2():
    rows = []
    for res in (8, 16):
        Y, X, C, cts, T = data.c2(res)
        m = data.fit(Y, X, C, cts)
        for nl in NULLS:
            s = flashdeconv.reference_fit_scores(m, null=nl)
            for key in ("score", "score_pooled"):
                f = s["flag" + key[5:]]
                rows.append({"set": f"c2_{res}um", "ds": np.nan, "pattern": np.nan, "key": key,
                             "null": nl, "null_used": s["null"][key]["method"], "n_bins": len(f),
                             "flag_rate": f.mean()})
        print("c2", res, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cref, name = spotless()
    name.to_csv(OUT / "spotless_removal_naming.csv", index=False)
    cref.to_csv(OUT / "complete_reference_null_comparison.csv", index=False)
    c = c2()
    pd.concat([cref, c]).to_csv(OUT / "complete_reference_null_comparison.csv", index=False)
    print(name.groupby("null")["rank"].apply(lambda x: (x == 1).mean()))
