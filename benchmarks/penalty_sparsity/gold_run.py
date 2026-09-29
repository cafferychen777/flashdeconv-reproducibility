"""Spotless gold standards (seqFISH+ cortex/SVZ and OB, 7 FOVs each; STARmap): lambda auto vs 0.

Inputs and evaluation exactly as validation/rerun_final/benchmarks/spotless/run_silver_gold.py
(all reference types fitted, evaluated on ground-truth types after renormalisation), package
defaults. Coordinates: 'real' = FOV coordinates stored in the gold npz (primary); 'grid' = the
index grid used by the benchmark harness (checks the earlier 14/15 statement).
Bootstrap blocks = FOVs (each FOV is one block).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/spotless")
import run_silver_gold as rsg  # noqa: E402  (imports the fdfinal hook; FD_FITLOG unset -> no log)
from flashdeconv import FlashDeconv, __version__  # noqa: E402
from ps_common import analyze, jsd_rows, normalize  # noqa: E402

OUT = Path("/Users/apple/Research/FlashDeconv/results/penalty_sparsity/gold")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = {c: dict(Pa=[], P0=[], G=[], umi=[], fov=[]) for c in ("real", "grid")}
    fov_rows = []
    for d in rsg.iter_datasets("gold"):
        Ya, Xa, _ = rsg.ce.align_genes(d["Y"], d["genes"], d["X"], d["rgenes"])
        true_cols = [c for c in d["types"] if c in d["props"].columns]
        G = d["props"][true_cols].to_numpy(float)
        idx = [list(d["types"]).index(c) for c in true_cols]
        umi = Ya.sum(1)
        for cname, coords in (("real", d["real_coords"]), ("grid", rsg.grid(Ya.shape[0]))):
            P, lam = {}, {}
            for arm, l in (("auto", "auto"), ("l0", 0.0)):
                m = FlashDeconv(lambda_spatial=l)
                P[arm] = normalize(m.fit_transform(Ya, Xa, coords)[:, idx])
                lam[arm] = m.lambda_used_
            Gn = G / np.maximum(G.sum(1, keepdims=True), 1e-12)
            ja, j0 = jsd_rows(P["auto"], Gn), jsd_rows(P["l0"], Gn)
            ca = np.corrcoef(P["auto"].ravel(), Gn.ravel())[0, 1]
            c0 = np.corrcoef(P["l0"].ravel(), Gn.ravel())[0, 1]
            fov_rows.append(dict(coords=cname, benchmark=d["benchmark"], fov=d["tissue"],
                                 n_spots=len(umi), median_umi=float(np.median(umi)),
                                 lambda_auto=lam["auto"], jsd_auto=ja.mean(), jsd_l0=j0.mean(),
                                 d_jsd=ja.mean() - j0.mean(), pearson_auto=ca, pearson_l0=c0,
                                 d_pearson=ca - c0, version=__version__))
            r = recs[cname]
            for k, M in (("Pa", P["auto"]), ("P0", P["l0"]), ("G", Gn)):
                r[k].append(pd.DataFrame(M, columns=true_cols))
            r["umi"].append(umi); r["fov"] += [d["tissue"]] * len(umi)
        print(fov_rows[-2]["fov"], {k: round(fov_rows[-2][k], 4) for k in ("d_jsd", "d_pearson")},
              flush=True)
    fov = pd.DataFrame(fov_rows)
    fov.to_csv(OUT / "gold_per_fov.csv", index=False)
    rows, trends = [], []
    for cname, r in recs.items():
        f = fov[fov.coords == cname]
        print(f"[{cname}] FOVs improved (JSD): {(f.d_jsd < 0).sum()}/{len(f)}, "
              f"Wilcoxon p={wilcoxon(f.jsd_auto, f.jsd_l0).pvalue:.3g}; "
              f"improved (Pearson): {(f.d_pearson > 0).sum()}/{len(f)}, "
              f"p={wilcoxon(f.pearson_auto, f.pearson_l0).pvalue:.3g}", flush=True)
        # FOVs differ in their ground-truth types: pool on the union (absent types = 0;
        # JSD is unchanged by zero columns, flattened Pearson then includes those zeros)
        Pa, P0, G = (pd.concat(r[k], ignore_index=True).fillna(0.0).to_numpy()
                     for k in ("Pa", "P0", "G"))
        umi = np.concatenate(r["umi"])
        # max observed type count in any FOV is < 20: rare-type AUPR is not estimable per stratum
        s, pb, tr = analyze("Spotless gold", f"{cname} coords", Pa, P0, G, umi,
                            np.zeros((len(umi), 2)), section=np.array(r["fov"]),
                            presence_thr=0.01, n_boot_ap=0,
                            lambda_auto=float(f.lambda_auto.median()), extra=dict(coords=cname))
        for row in s:
            row["fovs_improved"] = int((f.d_jsd < 0).sum())
            row["fov_wilcoxon_p"] = float(wilcoxon(f.jsd_auto, f.jsd_l0).pvalue)
        rows += s
        trends.append(tr)
        pb.to_csv(OUT / f"gold_{cname}_per_spot.csv.gz", float_format="%.6g")
    pd.DataFrame(rows).to_csv(OUT / "gold_summary.csv", index=False)
    pd.DataFrame(trends).to_csv(OUT / "gold_trend.csv", index=False)
    print(pd.DataFrame(rows)[["coords", "stratum", "n_bins", "d_jsd", "d_jsd_lo", "d_jsd_hi",
                              "frac_improved", "d_pearson"]].to_string())


if __name__ == "__main__":
    main()
