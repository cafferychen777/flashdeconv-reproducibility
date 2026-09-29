"""S1 (MERFISH-derived mouse gut, 8 um bins, held-out-slice self-reference): lambda auto vs 0.

Usage: python s1_run.py --scale 100000 --p 1.0 --data DIR --out DIR
p < 1 applies additional binomial thinning to the benchmark counts (numpy seed 1).
Writes <out>/s1_<scale>_p<p>_{props.npz, per_bin.csv.gz, summary.csv, trend.csv, fit.json}.
"""
import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ps_common import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--data", default="/scratch/user/cafferychen777/FlashDeconv/data/s1_merfish")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot-ap", type=int, default=200)
    ap.add_argument("--bins-csv", default="/scratch/user/cafferychen777/fd_final/results/penalty_sparsity/s1_gt_bins.csv.gz",
                    help="gt_bins.parquet converted to CSV (the fd_final env has no parquet engine)")
    a = ap.parse_args()
    import flashdeconv
    from flashdeconv import FlashDeconv
    from flashdeconv.io import prepare_data

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"s1_{a.scale}_p{a.p:g}"
    st = ad.read_h5ad(data / "st" / f"st_{a.scale}.h5ad")
    ref = ad.read_h5ad(data / "selfref" / "ref.h5ad")
    X = sparse.csr_matrix(st.X).astype(np.int64)
    if a.p < 1:
        rng = np.random.default_rng(1)
        X.data = rng.binomial(X.data, a.p)
        X.eliminate_zeros()
    st.X = X.astype(np.float32)
    umi = np.asarray(X.sum(1)).ravel()
    print(tag, st.shape, "median UMI", np.median(umi), "zero bins", int((umi == 0).sum()), flush=True)

    Y, Xs, coords, types, genes = prepare_data(st, ref, cell_type_key="cell_type")
    props, fit = {}, {}
    for arm, lam in [("auto", "auto"), ("l0", 0.0)]:
        t0 = time.perf_counter()
        m = FlashDeconv(lambda_spatial=lam, random_state=0)
        props[arm] = np.asarray(m.fit_transform(Y, Xs, coords, cell_type_names=types), np.float32)
        fit[arm] = dict(lambda_used=float(m.lambda_used_), n_iter=m.info_.get("n_iterations"),
                        converged=bool(m.info_.get("converged")), seconds=time.perf_counter() - t0,
                        n_genes=len(m.gene_idx_))
        print(arm, fit[arm], flush=True)
    fit.update(version=flashdeconv.__version__, n_bins=int(st.n_obs), median_umi=float(np.median(umi)),
               zero_umi_bins=int((umi == 0).sum()), p=a.p, scale=a.scale)
    (out / f"{tag}_fit.json").write_text(json.dumps(fit, indent=1))
    np.savez_compressed(out / f"{tag}_props.npz", auto=props["auto"], l0=props["l0"],
                        types=np.array(types), obs=np.array(st.obs_names))

    all_types = [t for t in (data / "types.txt").read_text().splitlines() if t]
    assert list(types) == all_types or set(types) == set(all_types), (types, all_types)
    bins = pd.read_csv(a.bins_csv, index_col=0)
    pos = pd.Series(np.arange(len(bins)), index=bins.index).loc[st.obs_names].to_numpy()
    gt_all = np.load(data / "gt_cellfrac.npy", mmap_mode="r")
    col = [all_types.index(t) for t in types]
    G = np.asarray(gt_all[pos][:, col], np.float64)
    glob = np.asarray(gt_all[: min(1_000_000, len(bins))][:, col]).mean(0)
    rare_idx = np.flatnonzero(glob < 0.05)
    xy = bins[["x", "y"]].to_numpy()[pos]
    rows, per_bin, trend = analyze(
        "S1 MERFISH gut", f"p={a.p:g}", props["auto"], props["l0"], G, umi, xy,
        section=bins.slice.to_numpy()[pos], presence_thr=0.0, rare_idx=rare_idx, seed=0,
        n_boot_ap=a.n_boot_ap, lambda_auto=fit["auto"]["lambda_used"],
        extra=dict(scale=a.scale, thin_p=a.p))
    per_bin["x"], per_bin["y"] = xy[:, 0], xy[:, 1]
    per_bin["n_cells"] = bins.n_cells.to_numpy()[pos]
    per_bin.index = st.obs_names
    per_bin.to_csv(out / f"{tag}_per_bin.csv.gz", float_format="%.6g")
    pd.DataFrame(rows).to_csv(out / f"{tag}_summary.csv", index=False)
    pd.DataFrame([trend]).to_csv(out / f"{tag}_trend.csv", index=False)
    print(pd.DataFrame(rows)[["stratum", "n_bins", "d_jsd", "d_jsd_lo", "d_jsd_hi",
                              "frac_improved", "d_pearson"]].to_string(), flush=True)
    print(trend, flush=True)


if __name__ == "__main__":
    main()
