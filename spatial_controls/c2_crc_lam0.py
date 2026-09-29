"""Control 2a: CRC Visium HD (P1/P2/P5, 8 um) refitted with lambda_spatial=0 (no Laplacian
penalty); every other argument as in validation/rerun_final/crc/run_final.py (cc.FD_KW,
max_iter=1000). Recomputes the same per-patient tables with the same crc_common functions
(fit label LAM0). The default-lambda (FINAL) tables are those already in
results/rerun_final/crc/<sample>/; the FINAL proportions (float16 npz) are only used for
FINAL-vs-LAM0 agreement."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, "/scratch/user/cafferychen777/fd_final/crc/code")
import crc_common as cc  # noqa: E402
import flashdeconv  # noqa: E402
from flashdeconv import FlashDeconv  # noqa: E402
from flashdeconv.io import prepare_data  # noqa: E402

OUT_ROOT = Path("/scratch/user/cafferychen777/controls_editor/crc")
FINAL_PROPS = Path("/scratch/user/cafferychen777/fd_final/crc/props")
cc.FD_KW["max_iter"] = 1000
KW = dict(cc.FD_KW, lambda_spatial=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    sid = ap.parse_args().sample
    out = OUT_ROOT / sid
    out.mkdir(parents=True, exist_ok=True)
    st = sc.read_h5ad(cc.ST_DIR / f"{sid}_deconv.h5ad")
    st.obs = st.obs[[]]
    st.obsm.pop("flashdeconv", None)
    st.uns.pop("flashdeconv_params", None)
    barcodes = np.asarray(st.obs_names).astype(str)
    coords = np.asarray(st.obsm["spatial"], dtype=np.float64)
    Xraw = st.X.tocsr() if sparse.issparse(st.X) else sparse.csr_matrix(st.X)
    umi = np.asarray(Xraw.sum(1)).ravel()
    vn = np.asarray(st.var_names).astype(str)
    expr = cc.gene_columns(Xraw, vn, list(dict.fromkeys(cc.NEUT_MARKERS + cc.NEG_MARKERS + cc.LR_GENES)))
    lin_expr = cc.gene_columns(Xraw, vn, cc.IMMUNE_MARKERS + cc.STROMAL_MARKERS)
    ref = cc.load_reference()
    Y, Xsig, coords_p, ctn, genes = prepare_data(st, ref, cell_type_key="Level2")
    types = [str(t) for t in ctn]
    assert np.allclose(coords_p, coords)
    m = FlashDeconv(verbose=False, **KW)
    t0 = time.perf_counter()
    P = np.asarray(m.fit_transform(Y, Xsig, coords, cell_type_names=ctn), dtype=np.float32)
    diag = {"sample": sid, "fit": "LAM0", "n_bins": int(Y.shape[0]), "fit_seconds": time.perf_counter() - t0,
            "converged": bool(m.info_.get("converged", False)), "n_iterations": int(m.info_.get("n_iterations", -1)),
            "lambda_used": float(m.lambda_used_), "version": flashdeconv.__version__}
    cc.log(f"LAM0 fit: {diag}")
    pd.DataFrame([diag]).to_csv(out / "fit_diagnostics.csv", index=False)
    del Y

    zf = np.load(FINAL_PROPS / f"{sid}_final_8um.npz", allow_pickle=True)
    assert [str(t) for t in zf["types"]] == types and np.array_equal(zf["barcodes"].astype(str), barcodes)
    Pfin = zf["P"].astype(np.float32)

    tree = cKDTree(coords)
    med = cc.median_nn(coords)
    eps = cc.DBSCAN_EPS_FACTOR * med
    signed = cc.boundary_distance(barcodes, coords, sid)
    rc = pd.read_csv(cc.AUX / "rctd" / f"DeconvolutionResults_{sid.replace('_CRC', '')}CRC.csv.gz",
                     usecols=["barcode", "DeconvolutionClass", "DeconvolutionLabel1", "DeconvolutionLabel2"],
                     dtype="string", keep_default_na=False, low_memory=False).set_index("barcode").reindex(barcodes)
    rctd = {"cls": rc["DeconvolutionClass"].fillna("missing").astype(str).to_numpy(),
            "l1": rc["DeconvolutionLabel1"].fillna("NA").astype(str).to_numpy(),
            "l2": rc["DeconvolutionLabel2"].fillna("NA").astype(str).to_numpy()}
    neut_j = types.index("Neutrophil")
    tag = {"sample": sid, "fit": "LAM0"}
    tabs = {}
    tabs["type_means"] = [{**tag, "cell_type": ct, "mean_proportion": float(v)}
                          for ct, v in zip(types, P.mean(0, dtype=np.float64))]
    ct_df, summ = cc.compare(P, Pfin, types, neut_j)
    tabs["compare_types"] = ct_df.assign(ref="FINAL", **tag).to_dict("records")
    tabs["compare_summary"] = [{**tag, "ref": "FINAL", **summ}]
    rows = cc.knn_enrichment(P, types, tree, coords, types, thr=0.10)
    rows += cc.knn_enrichment(P, types, tree, coords, ["Neutrophil"], thr=0.02)
    tabs["knn_enrichment"] = [{**tag, **r} for r in rows]
    agg, n_clu = cc.aggregates(P, types, coords, tree, med, eps)
    tabs["aggregates"] = agg.assign(n_dbscan_clusters=n_clu, **tag).to_dict("records") if len(agg) else []
    hot = P[:, neut_j] >= cc.NEUT_THR
    tabs["markers"] = [{**tag, **r} for r in cc.marker_fc(hot, umi, expr)]
    tabs["lr"] = [{**tag, **r} for r in cc.lr_enrichment(hot, coords, tree, expr)]
    tabs["rctd"] = [{**tag, **cc.rctd_breakdown(hot, rctd)}]
    tabs["lineage_markers"] = [{**tag, **r} for r in cc.lineage_marker_validation(P, types, rctd, lin_expr)]
    tabs["boundary"] = [{**tag, **r} for r in cc.boundary_bands(P, types, signed)]
    # smoothing diagnostic: mean Moran-like kNN autocorrelation of each type (k=6), FINAL vs LAM0
    _, nn = tree.query(coords, k=7)
    ac = []
    for j, ct in enumerate(types):
        for lab, M in [("FINAL", Pfin), ("LAM0", P)]:
            x = M[:, j].astype(np.float64)
            z = x - x.mean()
            ac.append({"sample": sid, "fit": lab, "cell_type": ct,
                       "knn6_autocorr": float((z[:, None] * z[nn[:, 1:]]).mean() / max(z.var(), 1e-30)),
                       "frac_bins_gt0.10": float((x >= 0.10).mean())})
    tabs["autocorr"] = ac
    for k, v in tabs.items():
        pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
    np.savez_compressed(out / "hotspot_bins.npz", LAM0_idx=np.flatnonzero(hot).astype(np.int32))
    np.savez_compressed(out / f"{sid}_lam0_8um.npz", P=P.astype(np.float16),
                        barcodes=barcodes, types=np.array(types))
    cc.log("done")


if __name__ == "__main__":
    main()
