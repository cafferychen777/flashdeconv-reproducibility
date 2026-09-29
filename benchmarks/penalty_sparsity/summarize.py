"""Combine per-dataset outputs into results/penalty_sparsity summary tables.

Inputs (copied from arseven fd_final/results/penalty_sparsity and the local gold run):
  s1/s1_<scale>_p<p>_{summary,trend}.csv, *_per_bin.csv.gz, *_fit.json
  c2/c2_<res>um_{summary,trend}.csv, *_fit.json
  gold/gold_{summary,trend,per_fov}.csv
Outputs:
  summary_by_stratum.csv  dataset x arm x depth stratum (mean dJSD with block-bootstrap CI, fraction
                          improved, dPearson, dRMSE, rare-type dAUPR, Wilcoxon P)
  trend_tests.csv         low- vs high-depth stratum difference per dataset arm
  thinning_paired.csv     S1: same bins, dJSD(p) - dJSD(p=1), block bootstrap (primary causal test)
  lambda_by_dataset.csv   auto lambda and median depth per dataset arm
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ps_common import LABELS, N_BOOT, boot_counts, ci

R = Path("/Users/apple/Research/FlashDeconv/results/penalty_sparsity")


def thinning_paired(scale):
    base = pd.read_csv(R / "s1" / f"s1_{scale}_p1_per_bin.csv.gz", index_col=0)
    d1 = base.jsd_auto - base.jsd_l0
    blk = base.block.to_numpy()
    nblk = blk.max() + 1
    rng = np.random.default_rng(0)
    C = boot_counts(nblk, rng, N_BOOT)
    rows = []
    for p in (0.5, 0.25):
        f = R / "s1" / f"s1_{scale}_p{p:g}_per_bin.csv.gz"
        if not f.exists():
            continue
        t = pd.read_csv(f, index_col=0).loc[base.index]
        assert (t.block.to_numpy() == blk).all()
        dp = t.jsd_auto - t.jsd_l0
        # stratify by the ORIGINAL (p=1) depth so each stratum holds the same bins in both arms
        for lab in LABELS + ["all"]:
            m = np.ones(len(base), bool) if lab == "all" else (base.stratum == lab).to_numpy()
            if m.sum() == 0:
                continue
            diff = (dp - d1).to_numpy()
            cnt = np.bincount(blk[m], minlength=nblk).astype(float)
            s = np.bincount(blk[m], weights=diff[m], minlength=nblk)
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (C @ s) / (C @ cnt)
            lo, hi = ci(b)
            rows.append(dict(scale=scale, thin_p=p, stratum_at_p1=lab, n_bins=int(m.sum()),
                             median_umi_p1=float(base.umi[m].median()),
                             median_umi_thinned=float(t.umi[m].median()),
                             d_jsd_p1=float(d1[m].mean()), d_jsd_thinned=float(dp[m].mean()),
                             thinned_minus_p1=float(diff[m].mean()), ci_lo=lo, ci_hi=hi,
                             boot_p=float(2 * min((b >= 0).mean(), (b <= 0).mean()))))
    return rows


def main():
    summ, trend, lam = [], [], []
    for f in sorted((R / "s1").glob("s1_*_summary.csv")) + sorted((R / "c2").glob("c2_*_summary.csv")):
        summ.append(pd.read_csv(f))
        trend.append(pd.read_csv(str(f).replace("_summary.csv", "_trend.csv")))
        fit = json.loads(Path(str(f).replace("_summary.csv", "_fit.json")).read_text())
        if "auto" in fit:
            lam.append(dict(dataset="S1 MERFISH gut", arm=f"{fit['scale']} bins, p={fit['p']:g}",
                            lambda_auto=fit["auto"]["lambda_used"], median_umi=fit["median_umi"],
                            n_bins=fit["n_bins"], iters_auto=fit["auto"]["n_iter"],
                            iters_l0=fit["l0"]["n_iter"],
                            converged=fit["auto"]["converged"] and fit["l0"]["converged"]))
        else:
            lam.append(dict(dataset="C2 Xenium CRC", arm=f"{fit['res']} um",
                            lambda_auto=fit["lambda_auto"], median_umi=fit["median_umi"],
                            n_bins=fit["n_bins"], iters_auto=fit["meta"]["auto"].get("n_iterations"),
                            iters_l0=fit["meta"]["l0"].get("n_iterations"),
                            converged=fit["meta"]["auto"].get("converged") and fit["meta"]["l0"].get("converged")))
    summ.append(pd.read_csv(R / "gold" / "gold_summary.csv"))
    trend.append(pd.read_csv(R / "gold" / "gold_trend.csv"))
    fov = pd.read_csv(R / "gold" / "gold_per_fov.csv")
    for c, g in fov.groupby("coords"):
        lam.append(dict(dataset="Spotless gold", arm=f"{c} coords", lambda_auto=g.lambda_auto.median(),
                        lambda_min=g.lambda_auto.min(), lambda_max=g.lambda_auto.max(),
                        median_umi=float(g.median_umi.median()), n_bins=int(g.n_spots.sum())))
    S = pd.concat(summ, ignore_index=True)
    lead = ["dataset", "arm", "stratum", "n_bins", "median_umi", "lambda_auto", "d_jsd", "d_jsd_lo",
            "d_jsd_hi", "frac_improved", "frac_worse", "wilcoxon_p", "d_pearson", "d_pearson_lo",
            "d_pearson_hi", "d_rmse", "d_rmse_lo", "d_rmse_hi", "d_rare_aupr", "d_rare_aupr_lo",
            "d_rare_aupr_hi"]
    S = S[[c for c in lead if c in S] + [c for c in S if c not in lead]]
    S.to_csv(R / "summary_by_stratum.csv", index=False)
    pd.concat(trend, ignore_index=True).to_csv(R / "trend_tests.csv", index=False)
    pd.DataFrame(lam).to_csv(R / "lambda_by_dataset.csv", index=False)
    thin = []
    for scale in (100000, 1000000):
        if (R / "s1" / f"s1_{scale}_p1_per_bin.csv.gz").exists():
            thin += thinning_paired(scale)
    pd.DataFrame(thin).to_csv(R / "thinning_paired.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[S.n_bins >= 200][["dataset", "arm", "stratum", "n_bins", "median_umi", "d_jsd", "d_jsd_lo",
                              "d_jsd_hi", "frac_improved", "wilcoxon_p", "d_pearson",
                              "d_rare_aupr"]].to_string())
    print(pd.concat(trend, ignore_index=True).to_string())
    print(pd.DataFrame(lam).to_string())
    print(pd.DataFrame(thin).to_string())


if __name__ == "__main__":
    main()
