"""FINAL rerun of the CRC Visium HD application (Oliveira et al. 2025, P1/P2/P5, 8 um)
with the final FlashDeconv package defaults (max_iter=1000, tol=1e-4, early stop;
gene_weighting="expected"). Copy of validation/rerun_v020/crc/run_v020.py; changes:
max_iter 100 -> 1000, fit label FINAL, FINAL-vs-V020 comparison, fdfinal fit log.

Original v0.2.0 docstring:
Rerun the CRC Visium HD application with
FlashDeconv v0.2.0 defaults (gene_weighting="expected", deterministic).

One job = one patient. Manuscript settings otherwise (d=512, lambda=auto,
rho=0.01, n_hvg=2000, 50 markers/type, k=6, max_iter=100, tol=1e-4, log-CPM,
full QC-Keep Chromium Flex reference, Level2, gene-symbol alignment as in the
original run). Fits:

  ORIG  archived published proportions (obsm['flashdeconv'] of the h5ad)
  V020  FlashDeconv v0.2.0 package defaults

All manuscript CRC statistics are recomputed for both fits with the validated
functions of validation/crc_seed_stability (crc_common.py). Additionally:
  * runtime of fit_transform (the quantity reported in the manuscript)
  * RCTD class fractions (method-independent)
  * multi-resolution niche: counts aggregated to 16/32/64 um on the array grid,
    re-deconvolved with v0.2.0, Neutrophil-focal kNN enrichment per resolution
  * compact outputs: float16 proportions (.npz, scratch), hotspot bin indices,
    compact figure data (coords + selected columns).
"""
import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/scratch/user/cafferychen777/fd_final/code")
import fdfinal  # noqa: E402,F401  (fit log hook; must precede model construction)
import crc_common as cc  # noqa: E402
import flashdeconv  # noqa: E402
from flashdeconv import FlashDeconv  # noqa: E402
from flashdeconv.io import prepare_data  # noqa: E402

OUT_ROOT = Path("/scratch/user/cafferychen777/fd_final/crc/results")
PROP_DIR = Path("/scratch/user/cafferychen777/fd_final/crc/props")
V020_PROPS = Path("/scratch/user/cafferychen777/fd_v020_crc/props")
cc.FD_KW["max_iter"] = 1000  # final package default (was 100 in the manuscript / v0.2.0 rerun)
MULTIRES = [16, 32, 64]
FIG_COLS = ["Neutrophil", "Tumor III", "Macrophage", "CD8 T cell", "CAF"]


def aggregate_counts(Xraw, array_coords, spatial, target_um, original_um=8):
    """Sum 8 um bins onto the coarser square grid of the Visium HD array: k x k blocks of
    8 um bins by integer division of (array_col, array_row) with k = target_um / 8.
    For 16 um this reproduces Space Ranger's square_016um in-tissue bins exactly
    (checked by check_spaceranger16.py)."""
    k = int(round(target_um / original_um))
    assert k * original_um == target_um
    grid = np.asarray(array_coords).astype(np.int64) // k
    key = grid[:, 0] * (grid[:, 1].max() + 1) + grid[:, 1]
    uniq, rows = np.unique(key, return_inverse=True)
    n = len(uniq)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(n, len(rows)))
    Xa = (A @ Xraw).tocsr()
    cnt = np.asarray(A.sum(1)).ravel()
    coords = (A @ spatial) / cnt[:, None]
    return Xa, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--skip-multires", action="store_true")
    args = ap.parse_args()
    sid = args.sample
    out = OUT_ROOT / sid
    out.mkdir(parents=True, exist_ok=True)
    PROP_DIR.mkdir(parents=True, exist_ok=True)
    env = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "numba_threads": os.environ.get("NUMBA_NUM_THREADS"),
           "flashdeconv_version": flashdeconv.__version__, "flashdeconv_path": flashdeconv.__file__,
           "python": platform.python_version()}
    try:
        env["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                            if l.startswith("model name")][0]
    except Exception:
        pass
    cc.log(f"env: {env}")

    st = sc.read_h5ad(cc.ST_DIR / f"{sid}_deconv.h5ad")
    orig = st.obsm["flashdeconv"].copy()
    st.obs = st.obs[[]]
    st.obsm.pop("flashdeconv", None)
    st.uns.pop("flashdeconv_params", None)
    barcodes = np.asarray(st.obs_names).astype(str)
    coords = np.asarray(st.obsm["spatial"], dtype=np.float64)
    array_coords = np.asarray(st.obsm["array_coords"])
    Xraw = st.X.tocsr() if sparse.issparse(st.X) else sparse.csr_matrix(st.X)
    umi = np.asarray(Xraw.sum(1)).ravel()
    vn = np.asarray(st.var_names).astype(str)
    expr = cc.gene_columns(Xraw, vn, list(dict.fromkeys(cc.NEUT_MARKERS + cc.NEG_MARKERS + cc.LR_GENES)))
    lin_expr = cc.gene_columns(Xraw, vn, cc.IMMUNE_MARKERS + cc.STROMAL_MARKERS)

    ref = cc.load_reference()
    t0 = time.perf_counter()
    Y, Xsig, coords_p, ctn, genes = prepare_data(st, ref, cell_type_key="Level2")
    t_prep = time.perf_counter() - t0
    types = [str(t) for t in ctn]
    assert np.allclose(coords_p, coords)
    cc.log(f"{sid}: {Y.shape[0]:,} bins, {len(genes):,} common genes, {len(types)} types, prep {t_prep:.1f}s")

    # ---- final default fit (timed exactly like the manuscript: fit_transform only)
    m = FlashDeconv(verbose=False, **cc.FD_KW)
    assert m.gene_weighting == "expected"
    t0 = time.perf_counter()
    P = m.fit_transform(Y, Xsig, coords, cell_type_names=ctn)
    t_fit = time.perf_counter() - t0
    diag = [{"sample": sid, "fit": "FINAL", "resolution_um": 8, "n_bins": int(Y.shape[0]),
             "fit_seconds": t_fit, "prepare_data_seconds": t_prep,
             "bins_per_second": Y.shape[0] / t_fit, "n_common_genes": len(genes),
             "converged": bool(m.info_.get("converged", False)),
             "n_iterations": int(m.info_.get("n_iterations", -1)),
             "lambda_used": float(m.lambda_used_), "n_selected_genes": int(len(m.gene_idx_)),
             "gene_weighting": m.gene_weighting, **env}]
    cc.log(f"FINAL fit: {diag[-1]}")
    # determinism check: a second fit must be bit-identical
    t0 = time.perf_counter()
    P2 = FlashDeconv(verbose=False, random_state=123, **cc.FD_KW).fit_transform(Y, Xsig, coords, cell_type_names=ctn)
    diag[-1]["refit_seconds"] = time.perf_counter() - t0
    diag[-1]["refit_identical"] = bool(np.array_equal(P, P2))
    diag[-1]["refit_maxabsdiff"] = float(np.abs(P - P2).max())
    del P2
    cc.log(f"determinism: identical={diag[-1]['refit_identical']}")
    pd.DataFrame(diag).to_csv(out / "fit_diagnostics.csv", index=False)

    zv = np.load(V020_PROPS / f"{sid}_v020_8um.npz", allow_pickle=True)
    assert [str(t) for t in zv["types"]] == types and np.array_equal(zv["barcodes"].astype(str), barcodes)
    fits = {"ORIG": orig[types].to_numpy(np.float32), "FINAL": np.asarray(P, dtype=np.float32)}
    v020 = zv["P"].astype(np.float32)
    np.savez_compressed(PROP_DIR / f"{sid}_final_8um.npz", P=fits["FINAL"].astype(np.float16),
                        coords=coords.astype(np.float32), barcodes=barcodes, types=np.array(types))
    fc = [types.index(c) for c in FIG_COLS]
    np.savez_compressed(out / "figdata.npz", coords=coords.astype(np.float32),
                        cols=np.array(FIG_COLS), FINAL=fits["FINAL"][:, fc].astype(np.float16),
                        V020=v020[:, fc].astype(np.float16),
                        ORIG=fits["ORIG"][:, fc].astype(np.float16))

    # ---- shared geometry
    tree = cKDTree(coords)
    med = cc.median_nn(coords)
    eps = cc.DBSCAN_EPS_FACTOR * med
    signed = cc.boundary_distance(barcodes, coords, sid)
    rc = pd.read_csv(cc.AUX / "rctd" / f"DeconvolutionResults_{sid.replace('_CRC', '')}CRC.csv.gz",
                     usecols=["barcode", "DeconvolutionClass", "DeconvolutionLabel1", "DeconvolutionLabel2"],
                     dtype="string", keep_default_na=False, low_memory=False).set_index("barcode")
    rc = rc.reindex(barcodes)
    rctd = {"cls": rc["DeconvolutionClass"].fillna("missing").astype(str).to_numpy(),
            "l1": rc["DeconvolutionLabel1"].fillna("NA").astype(str).to_numpy(),
            "l2": rc["DeconvolutionLabel2"].fillna("NA").astype(str).to_numpy()}
    cls_frac = pd.Series(rctd["cls"]).value_counts(normalize=True).rename("fraction").reset_index()
    cls_frac.insert(0, "sample", sid)
    cls_frac.to_csv(out / "rctd_class_fractions.csv", index=False)
    neut_j = types.index("Neutrophil")

    tabs = {k: [] for k in ["type_means", "compare_types", "compare_summary", "knn_enrichment",
                            "aggregates", "markers", "lr", "rctd", "lineage_markers", "boundary"]}
    hot_idx = {}
    for name, Pf in fits.items():
        cc.log(f"analysing {name}")
        tag = {"sample": sid, "fit": name}
        tabs["type_means"] += [{**tag, "cell_type": ct, "mean_proportion": float(v)}
                               for ct, v in zip(types, Pf.mean(0, dtype=np.float64))]
        if name == "FINAL":
            for rn, Rf in [("ORIG", fits["ORIG"]), ("V020", v020)]:
                ct_df, summ = cc.compare(Pf, Rf, types, neut_j)
                tabs["compare_types"] += ct_df.assign(ref=rn, **tag).to_dict("records")
                tabs["compare_summary"].append({**tag, "ref": rn, **summ})
        rows = cc.knn_enrichment(Pf, types, tree, coords, types, thr=0.10)
        rows += cc.knn_enrichment(Pf, types, tree, coords, ["Neutrophil"], thr=0.02)
        tabs["knn_enrichment"] += [{**tag, **r} for r in rows]
        agg, n_clu = cc.aggregates(Pf, types, coords, tree, med, eps)
        if len(agg):
            tabs["aggregates"] += agg.assign(n_dbscan_clusters=n_clu, **tag).to_dict("records")
        hot = Pf[:, neut_j] >= cc.NEUT_THR
        hot_idx[name] = np.flatnonzero(hot).astype(np.int32)
        tabs["markers"] += [{**tag, **r} for r in cc.marker_fc(hot, umi, expr)]
        tabs["lr"] += [{**tag, **r} for r in cc.lr_enrichment(hot, coords, tree, expr)]
        tabs["rctd"].append({**tag, **cc.rctd_breakdown(hot, rctd)})
        tabs["lineage_markers"] += [{**tag, **r} for r in cc.lineage_marker_validation(Pf, types, rctd, lin_expr)]
        tabs["boundary"] += [{**tag, **r} for r in cc.boundary_bands(Pf, types, signed)]
    np.savez_compressed(out / "hotspot_bins.npz", barcodes_order="h5ad obs order",
                        **{f"{k}_idx": v for k, v in hot_idx.items()})
    for k, v in tabs.items():
        pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)

    # ---- multi-resolution niche (re-deconvolution of aggregated counts)
    if not args.skip_multires:
        mr_rows = []
        for res in [8] + MULTIRES:
            if res == 8:
                Pr, cr = fits["FINAL"], coords
            else:
                Xa, cr = aggregate_counts(Xraw, array_coords, coords, res)
                sta = sc.AnnData(X=Xa)
                sta.var_names = st.var_names.copy()
                sta.obs_names = [f"bin_{i}" for i in range(Xa.shape[0])]
                sta.obsm["spatial"] = cr
                Ya, Xs, cra, ctn_a, _ = prepare_data(sta, ref, cell_type_key="Level2")
                assert [str(t) for t in ctn_a] == types
                ma = FlashDeconv(verbose=False, **cc.FD_KW)
                t0 = time.perf_counter()
                Pr = ma.fit_transform(Ya, Xs, cra, cell_type_names=ctn_a).astype(np.float32)
                tf = time.perf_counter() - t0
                cr = np.asarray(cra, dtype=np.float64)
                diag.append({"sample": sid, "fit": "FINAL", "resolution_um": res, "n_bins": int(Ya.shape[0]),
                             "fit_seconds": tf, "converged": bool(ma.info_.get("converged", False)),
                             "n_iterations": int(ma.info_.get("n_iterations", -1)),
                             "lambda_used": float(ma.lambda_used_), **env})
                cc.log(f"multires {res}um: {diag[-1]}")
            tr = cKDTree(cr)
            rows = cc.knn_enrichment(Pr, types, tr, cr, ["Neutrophil"], thr=0.10)
            n_hot = int((Pr[:, neut_j] >= 0.10).sum())
            rec = {"sample": sid, "resolution_um": res, "n_bins": int(Pr.shape[0]), "n_hot": n_hot}
            for r in rows:
                rec[f"ratio_{r['neighbor_type']}"] = r["enrichment_ratio"]
                rec[f"log2_{r['neighbor_type']}"] = r["log2_enrichment"]
            mr_rows.append(rec)
            pd.DataFrame(mr_rows).to_csv(out / "multires_enrichment.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "fit_diagnostics.csv", index=False)
    with open(out / "env.json", "w") as fh:
        json.dump(env, fh, indent=1)
    cc.log("done")


if __name__ == "__main__":
    main()
