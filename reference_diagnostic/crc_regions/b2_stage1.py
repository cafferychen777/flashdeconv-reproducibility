"""B2 stage 1 (one patient): FlashDeconv fit with package defaults, reference-incompleteness
scores, flagged-bin regions, unexplained genes + HPA suggestions per region, null blocks,
label-permutation control, split-half deconvolution-free test. See PROTOCOL.md."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, ndimage
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from sklearn.cluster import DBSCAN

PROJ = Path("/scratch/user/cafferychen777/FlashDeconv")
B2 = Path("/scratch/user/cafferychen777/fd_b2")
ST_DIR = PROJ / "analysis/crc_cohort_results"
REF_H5 = PROJ / "data/visium_hd_crc_cohort/scRNA_ref/HumanColonCancer_Flex_filtered_feature_bc_matrix.h5"
REF_META = PROJ / "data/visium_hd_crc_cohort/metadata/SingleCell_MetaData.csv.gz"
PATHO = B2 / "input/spacehack_8um_P2.csv"
HPA = B2 / "input/hpa_single_cell_type_ncpm.csv.gz"

import flashdeconv  # noqa: E402
from flashdeconv import FlashDeconv  # noqa: E402
from flashdeconv.io import prepare_data  # noqa: E402
from flashdeconv.core.refcheck import (reference_fit_scores, suggest_missing_types,  # noqa: E402
                                       unexplained_genes)

MIN_REGION = 100
MAX_REGIONS = 40
N_NULL = 50
N_PERM_REGIONS = 10
TOPG = 20
RNG = np.random.default_rng(0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_reference():
    ref = sc.read_10x_h5(REF_H5)
    meta = pd.read_csv(REF_META, compression="gzip").set_index("Barcode")
    common = ref.obs_names.intersection(meta.index)
    ref = ref[common].copy()
    ref = ref[(meta.loc[ref.obs_names, "QCFilter"] == "Keep").to_numpy()].copy()
    ref.obs["Level2"] = meta.loc[ref.obs_names, "Level2"].values
    return ref


def edge_distance(array_coords):
    rc = np.asarray(array_coords, dtype=np.int64)
    rc = rc - rc.min(0) + 1
    grid = np.zeros(rc.max(0) + 2, dtype=bool)
    grid[rc[:, 0], rc[:, 1]] = True
    d = ndimage.distance_transform_cdt(grid, metric="chessboard")
    return d[rc[:, 0], rc[:, 1]]  # 1 = edge bin


def hpa_markers(A, genes, n=50):
    cp = A / np.maximum(A.sum(1, keepdims=True), 1e-300) * 1e4
    tot = cp.sum(0)
    K = cp.shape[0]
    out = []
    for k in range(K):
        other = (tot - cp[k]) / (K - 1)
        lfc = np.log2((cp[k] + 0.1) / (other + 0.1))
        out.append(set(np.asarray(genes)[np.argsort(-lfc, kind="stable")[:n]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="keep a spatial window of ~N bins (test)")
    ap.add_argument("--min-region", type=int, default=MIN_REGION, help="test only")
    a = ap.parse_args()
    min_region = a.min_region
    sid = a.sample
    out = B2 / ("smoke" if a.smoke else "results") / sid
    out.mkdir(parents=True, exist_ok=True)
    log(f"flashdeconv {flashdeconv.__version__} from {flashdeconv.__file__}")

    st = sc.read_h5ad(ST_DIR / f"{sid}_deconv.h5ad")
    st.obs = st.obs[[]]
    if a.smoke:
        c = np.asarray(st.obsm["spatial"])
        d = ((c - np.median(c, 0)) ** 2).sum(1)
        st = st[np.argsort(d)[:a.smoke]].copy()
    st.obsm.pop("flashdeconv", None)
    st.uns.pop("flashdeconv_params", None)
    coords = np.asarray(st.obsm["spatial"], dtype=np.float64)
    arr = np.asarray(st.obsm["array_coords"])
    barcodes = np.asarray(st.obs_names).astype(str)
    Xraw = st.X.tocsr() if sparse.issparse(st.X) else sparse.csr_matrix(st.X)
    umi_total = np.asarray(Xraw.sum(1)).ravel()
    edge = edge_distance(arr)
    ref = load_reference()
    Y, Xsig, crd, ctn, genes = prepare_data(st, ref, cell_type_key="Level2")
    genes = np.asarray(genes).astype(str)
    types = [str(t) for t in ctn]
    del ref
    log(f"{sid}: {Y.shape[0]:,} bins, {len(genes):,} genes, {len(types)} types")

    t0 = time.perf_counter()
    m = FlashDeconv()
    m.fit(Y, Xsig, crd, cell_type_names=np.asarray(types))
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = reference_fit_scores(m)
    t_diag = time.perf_counter() - t0
    flag = s["flag_pooled"]
    P = m.proportions_
    dom = np.asarray(types)[P.argmax(1)]
    log(f"fit {t_fit:.0f}s ({m.info_.get('n_iterations')} it), diag {t_diag:.0f}s; "
        f"flag {s['flag'].mean():.4f}, flag_pooled {flag.mean():.4f}")

    # --- pathologist labels (P2 only; see PROTOCOL)
    patho = np.full(len(barcodes), "", dtype=object)
    if sid == "P2_CRC":
        pa = pd.read_csv(PATHO, sep="\t", header=None, names=["b", "c"]).set_index("b").c
        patho = pa.reindex(barcodes).fillna("").to_numpy(object)
        log(f"patho matched {np.mean(patho != ''):.4f}")

    # --- flag rates by strata
    rows = [{"stratum": "all", "level": "all", "n": len(flag), "flag_pooled": flag.mean(),
             "flag_single": s["flag"].mean()}]
    dec = pd.qcut(umi_total, 10, labels=False, duplicates="drop")
    for d in np.unique(dec):
        r = dec == d
        rows.append({"stratum": "umi_decile", "level": int(d), "n": int(r.sum()),
                     "flag_pooled": flag[r].mean(), "flag_single": s["flag"][r].mean(),
                     "median_umi": float(np.median(umi_total[r]))})
    band = np.select([edge <= 2, edge <= 5, edge <= 10], ["0-2", "3-5", "6-10"], ">10")
    for b in ["0-2", "3-5", "6-10", ">10"]:
        r = band == b
        rows.append({"stratum": "edge_band", "level": b, "n": int(r.sum()),
                     "flag_pooled": flag[r].mean() if r.any() else np.nan,
                     "flag_single": s["flag"][r].mean() if r.any() else np.nan})
    for t in types:
        r = dom == t
        if r.sum() >= 100:
            rows.append({"stratum": "dominant_type", "level": t, "n": int(r.sum()),
                         "flag_pooled": flag[r].mean(), "flag_single": s["flag"][r].mean()})
    if sid == "P2_CRC":
        for c in sorted(set(patho) - {""}):
            r = patho == c
            rows.append({"stratum": "pathologist", "level": c, "n": int(r.sum()),
                         "flag_pooled": flag[r].mean(), "flag_single": s["flag"][r].mean()})
    pd.DataFrame(rows).to_csv(out / "flag_rates.csv", index=False)

    # --- HPA atlas
    hpa = pd.read_csv(HPA, index_col=0)
    A = hpa.to_numpy(np.float64)
    agenes = hpa.columns.to_numpy(str)
    anames = hpa.index.to_numpy(str)
    amarkers = dict(zip(anames, hpa_markers(A, agenes)))

    def characterise(mask, label, kind):
        g = unexplained_genes(m, mask, gene_names=genes)
        t = suggest_missing_types(g, A, agenes, anames)
        fin = np.isfinite(g["score"])
        top = list(g["gene"][fin][:TOPG])
        tt = str(t["type"][0])
        return g, t, {"set": label, "kind": kind, "n_bins": int(mask.sum()),
                      "top_gene_score": float(g["score"][0]),
                      "top10_mean_score": float(np.mean(g["score"][fin][:10])),
                      "n_genes_scored": int(fin.sum()),
                      "top_hpa": tt, "top_hpa_score": float(t["mean_score"][0]),
                      "hpa2": str(t["type"][1]), "hpa2_score": float(t["mean_score"][1]),
                      "hpa3": str(t["type"][2]), "hpa3_score": float(t["mean_score"][2]),
                      "n_top20_in_hpa_markers": len(set(top) & amarkers[tt]),
                      "top_genes": ";".join(top)}

    # --- regions
    med_nn = float(np.median(cKDTree(coords[:20000]).query(coords[:20000], k=2)[0][:, 1]))
    fidx = np.flatnonzero(flag)
    lab = DBSCAN(eps=1.5 * med_nn, min_samples=5).fit_predict(coords[fidx])
    region = np.full(len(flag), -1, dtype=np.int32)
    sizes = pd.Series(lab[lab >= 0]).value_counts()
    big = sizes[sizes >= min_region].index[:MAX_REGIONS]
    for new, old in enumerate(big):
        region[fidx[lab == old]] = new
    log(f"med_nn {med_nn:.2f}px; {len(sizes)} clusters, {int((sizes >= min_region).sum())} >= "
        f"{min_region} bins; analysing {len(big)}; flagged in regions "
        f"{(region >= 0).sum() / max(len(fidx), 1):.3f}")

    summ, gene_rows, type_rows = [], [], []
    sec_med_umi = float(np.median(umi_total))

    def store(g, t, info, extra):
        info.update(extra)
        summ.append(info)
        for i in range(200):
            if not np.isfinite(g["score"][i]):
                break
            gene_rows.append({"set": info["set"], "rank": i + 1, "gene": g["gene"][i],
                              "score": g["score"][i], "observed": g["observed"][i],
                              "expected": g["expected"][i]})
        for i in range(10):
            type_rows.append({"set": info["set"], "rank": i + 1, "type": t["type"][i],
                              "mean_score": t["mean_score"][i],
                              "n_markers_scored": t["n_markers_scored"][i]})

    if flag.sum() >= 20:
        g, t, info = characterise(flag, "all_flagged", "all")
        store(g, t, info, {"median_umi_ratio": float(np.median(umi_total[flag])) / sec_med_umi})

    # background pool for UMI matching: non-flagged, > 10 bins from any flagged bin
    far = np.ones(len(flag), dtype=bool)
    if flag.any():
        dfl, _ = cKDTree(coords[flag]).query(coords, k=1, distance_upper_bound=10.5 * med_nn)
        far = ~flag & ~np.isfinite(dfl)
    bg_idx = np.flatnonzero(far)
    dec_all = np.searchsorted(np.quantile(umi_total, np.linspace(0, 1, 11)[1:-1]), umi_total)

    def split_half(ridx):
        c = coords[ridx] - coords[ridx].mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        pc = c @ u
        A_, B_ = ridx[pc <= np.median(pc)], ridx[pc > np.median(pc)]
        mA = np.zeros(len(flag), bool)
        mA[A_] = True
        gA = unexplained_genes(m, mA, gene_names=genes)
        ok = np.isfinite(gA["score"]) & (gA["observed"] >= 20)
        prog = list(gA["gene"][ok][:10])
        return prog, B_

    gpos = {g_: i for i, g_ in enumerate(genes)}
    Ycsc = sparse.csc_matrix(Y)

    def program_test(prog, test_idx):
        cols = [gpos[g_] for g_ in prog]
        progc = np.asarray(Ycsc[:, cols].sum(1)).ravel()
        frac = progc / np.maximum(umi_total, 1)
        bgs = []
        for d in np.unique(dec_all[test_idx]):
            need = 5 * int((dec_all[test_idx] == d).sum())
            pool = bg_idx[dec_all[bg_idx] == d]
            if len(pool):
                bgs.append(RNG.choice(pool, size=need, replace=len(pool) < need))
        bg = np.concatenate(bgs) if bgs else np.array([], int)
        if len(bg) == 0:
            return {}
        fr, fb = frac[test_idx], frac[bg]
        p = mannwhitneyu(fr, fb, alternative="greater").pvalue
        return {"prog_mean_frac_test": float(fr.mean()), "prog_mean_frac_bg": float(fb.mean()),
                "prog_ratio_mean": float(fr.mean() / max(fb.mean(), 1e-12)),
                "prog_ratio_median": float(np.median(fr) / max(np.median(fb), 1e-12))
                if np.median(fb) > 0 else np.inf if np.median(fr) > 0 else np.nan,
                "prog_detect_test": float((progc[test_idx] > 0).mean()),
                "prog_detect_bg": float((progc[bg] > 0).mean()),
                "prog_mwu_p": float(p), "prog_n_test": int(len(test_idx)),
                "prog_umi_test_median": float(np.median(umi_total[test_idx])),
                "prog_umi_bg_median": float(np.median(umi_total[bg]))}

    for r in range(len(big)):
        ridx = np.flatnonzero(region == r)
        mask = region == r
        g, t, info = characterise(mask, f"region_{r}", "region")
        extra = {"centroid_x": float(coords[ridx, 0].mean()), "centroid_y": float(coords[ridx, 1].mean()),
                 "median_umi_ratio": float(np.median(umi_total[ridx])) / sec_med_umi,
                 "edge_frac_le2": float((edge[ridx] <= 2).mean()),
                 "median_score_pooled": float(np.median(s["score_pooled"][ridx])),
                 "dominant_fitted": pd.Series(dom[ridx]).value_counts().index[0],
                 "dominant_fitted_frac": float(pd.Series(dom[ridx]).value_counts(normalize=True).iloc[0])}
        if sid == "P2_CRC":
            vc = pd.Series(patho[ridx]).value_counts(normalize=True)
            extra.update({"patho_top": vc.index[0], "patho_top_frac": float(vc.iloc[0]),
                          "patho_comp": ";".join(f"{k}:{v:.2f}" for k, v in vc.items())})
        prog, B_ = split_half(ridx)
        extra["split_program"] = ";".join(prog)
        if len(prog) >= 3:
            extra.update(program_test(prog, B_))
        store(g, t, info, extra)
        log(f"region {r}: n={mask.sum()} {info['top_genes'][:120]} | {info['top_hpa']} "
            f"{info['top_hpa_score']:.2f} | ratio {extra.get('prog_ratio_mean', np.nan):.2f}")

    # --- null blocks in well-explained tissue
    ok_idx = np.flatnonzero(~flag)
    ok_tree = cKDTree(coords[ok_idx])
    seeds = np.flatnonzero(~flag & (s["score_pooled"] < 0))
    real_sizes = sizes[sizes >= min_region].to_numpy() if len(big) else np.array([200])
    null_rows = []
    for j in range(N_NULL):
        n = int(RNG.choice(real_sizes))
        seed = RNG.choice(seeds)
        _, nn = ok_tree.query(coords[seed], k=n)
        mask = np.zeros(len(flag), bool)
        mask[ok_idx[np.atleast_1d(nn)]] = True
        g, t, info = characterise(mask, f"null_{j}", "null")
        info["median_umi_ratio"] = float(np.median(umi_total[mask])) / sec_med_umi
        null_rows.append(info)
    # --- label permutation among flagged bins (spatial coherence removed)
    for r in range(min(N_PERM_REGIONS, len(big))):
        n = int((region == r).sum())
        mask = np.zeros(len(flag), bool)
        mask[RNG.choice(fidx, size=n, replace=False)] = True
        g, t, info = characterise(mask, f"perm_region_{r}", "perm")
        null_rows.append(info)
    pd.DataFrame(summ).to_csv(out / "regions.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out / "null_regions.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(out / "region_genes.csv.gz", index=False, float_format="%.4g")
    pd.DataFrame(type_rows).to_csv(out / "region_hpa_types.csv", index=False, float_format="%.4g")

    np.savez_compressed(
        out / "perbin.npz", barcode=barcodes, x=coords[:, 0].astype(np.float32),
        y=coords[:, 1].astype(np.float32), umi=umi_total.astype(np.int32),
        n_umi_sel=s["n_umi"].astype(np.float32), score=s["score"].astype(np.float16),
        score_pooled=s["score_pooled"].astype(np.float16), flag=s["flag"], flag_pooled=flag,
        region=region, edge=edge.astype(np.int16), patho=patho.astype(str),
        dominant=dom.astype(str), maxprop=P.max(1).astype(np.float16))
    np.savez_compressed(out / "props.npz", P=P.astype(np.float16), types=np.asarray(types))
    json.dump({"sample": sid, "n_bins": int(len(flag)), "n_genes": int(len(genes)),
               "fit_seconds": t_fit, "diag_seconds": t_diag,
               "n_iterations": int(m.info_.get("n_iterations", -1)),
               "converged": bool(m.info_.get("converged", False)),
               "lambda_used": float(m.lambda_used_), "median_nn_px": med_nn,
               "flag_pooled_frac": float(flag.mean()), "flag_single_frac": float(s["flag"].mean()),
               "n_clusters": int(len(sizes)), "n_regions_ge_min": int((sizes >= min_region).sum()),
               "version": flashdeconv.__version__}, open(out / "meta.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
