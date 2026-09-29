"""C2 (Xenium CRC P1 pseudo-Visium HD, self-reference): lambda auto vs 0 per bin.

Reuses the final-package fits final_default_auto / final_default_l0 (random_state=42,
validation/rerun_final/benchmarks/c2/run_fd_final.py); no refit.
Usage: python c2_run.py --res 2 --out DIR
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ps_common import analyze

BINS = Path("/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/bins")
PREDS = Path("/scratch/user/cafferychen777/fd_final/results/c2/preds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot-ap", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(BINS / f"bins_{a.res}um.npz", allow_pickle=True)
    Y = sparse.csr_matrix((d["Y_data"], d["Y_indices"], d["Y_indptr"]), shape=tuple(d["Y_shape"]))
    umi = np.asarray(Y.sum(1)).ravel()
    G = d["gt_props"].astype(np.float64)
    xy = d["centers"].astype(np.float64)
    P, meta = {}, {}
    for arm, mode in [("auto", "final_default_auto"), ("l0", "final_default_l0")]:
        z = np.load(PREDS / f"{a.res}um" / f"FlashDeconv__{mode}__umiNA.npz", allow_pickle=True)
        assert z["covered"].all()
        P[arm] = z["props"]
        meta[arm] = json.loads(str(z["meta"]))
    lam = float(re.search(r"lambda_used=([0-9.eE+-]+)", meta["auto"]["notes"]).group(1))
    lam0 = float(re.search(r"lambda_used=([0-9.eE+-]+)", meta["l0"]["notes"]).group(1))
    assert lam0 == 0.0
    fit = dict(res=a.res, lambda_auto=lam, n_bins=int(len(umi)), median_umi=float(np.median(umi)),
               zero_umi_bins=int((umi == 0).sum()), meta=meta)
    (out / f"c2_{a.res}um_fit.json").write_text(json.dumps(fit, indent=1, default=str))
    rows, per_bin, trend = analyze(
        "C2 Xenium CRC", f"{a.res} um", P["auto"], P["l0"], G, umi, xy, section=None,
        presence_thr=0.01, rare_idx=None, seed=0, n_boot_ap=a.n_boot_ap, lambda_auto=lam,
        extra=dict(bin_um=a.res))
    per_bin["x"], per_bin["y"] = xy[:, 0], xy[:, 1]
    per_bin.to_csv(out / f"c2_{a.res}um_per_bin.csv.gz", float_format="%.6g")
    pd.DataFrame(rows).to_csv(out / f"c2_{a.res}um_summary.csv", index=False)
    pd.DataFrame([trend]).to_csv(out / f"c2_{a.res}um_trend.csv", index=False)
    print(pd.DataFrame(rows)[["stratum", "n_bins", "d_jsd", "d_jsd_lo", "d_jsd_hi",
                              "frac_improved", "d_pearson"]].to_string(), flush=True)
    print(trend, flush=True)


if __name__ == "__main__":
    main()
