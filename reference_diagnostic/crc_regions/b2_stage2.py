"""B2 stage 2: deconvolution-free checks of the programs named by stage 1.

(1) Cross-patient raw-count scan. For each program (genes defined on half A of a P2 region in
    stage 1), compute the per-bin program fraction in every patient, smooth it over a
    7 x 7-bin (56 um) window, and count bins above the level of the source region itself
    (median smoothed fraction in the P2 region). Report the hotspot clusters and the diagnostic
    pooled score inside them.
(2) P1 Xenium serial section: per-cell counts of the program genes on the panel.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

PROJ = Path("/scratch/user/cafferychen777/FlashDeconv")
B2 = Path("/scratch/user/cafferychen777/fd_b2")
ST_DIR = PROJ / "analysis/crc_cohort_results"
XEN = PROJ / "analysis/xenium_validation_results/xenium_P1_CRC_annotated.h5ad"
OUT = B2 / "results/stage2"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]
HALF = 3  # 7 x 7 bins window


def log(m):
    print(m, flush=True)


regs = pd.read_csv(B2 / "results/P2_CRC/regions.csv", keep_default_na=False)
regs = regs[regs["set"].str.startswith("region_")]
programs = {r["set"]: r["split_program"].split(";") for _, r in regs.iterrows()}
# full-region top-10 as a second definition (used for Xenium, where more genes help)
programs_full = {r["set"]: r["top_genes"].split(";")[:10] for _, r in regs.iterrows()}
log(json.dumps(programs, indent=0))


def smooth_frac(arr, prog_c, umi):
    rc = np.asarray(arr, dtype=np.int64)
    rc = rc - rc.min(0)
    shape = rc.max(0) + 1
    P = np.zeros(shape)
    U = np.zeros(shape)
    np.add.at(P, (rc[:, 0], rc[:, 1]), prog_c)
    np.add.at(U, (rc[:, 0], rc[:, 1]), umi)
    from scipy.ndimage import uniform_filter
    w = 2 * HALF + 1
    Ps = uniform_filter(P, size=w, mode="constant")
    Us = uniform_filter(U, size=w, mode="constant")
    return (Ps / np.maximum(Us, 1e-9))[rc[:, 0], rc[:, 1]]


rows, bins_out = [], {}
cache = {}
for sid in SAMPLES:
    st = sc.read_h5ad(ST_DIR / f"{sid}_deconv.h5ad", backed=None)
    X = st.X.tocsc() if sparse.issparse(st.X) else sparse.csc_matrix(st.X)
    vn = np.asarray(st.var_names).astype(str)
    pos = {}
    for i, g in enumerate(vn):
        pos.setdefault(g, i)
    umi = np.asarray(X.sum(1)).ravel()
    arr = np.asarray(st.obsm["array_coords"])
    xy = np.asarray(st.obsm["spatial"], dtype=np.float64)
    pb = np.load(B2 / f"results/{sid}/perbin.npz", allow_pickle=True)
    assert np.array_equal(pb["barcode"], np.asarray(st.obs_names).astype(str))
    sp = pb["score_pooled"].astype(np.float64)
    sp_rank = pd.Series(sp).rank(pct=True).to_numpy()
    reg = pb["region"]
    for name, genes in programs.items():
        cols = [pos[g] for g in genes if g in pos]
        pc = np.asarray(X[:, cols].sum(1)).ravel()
        sf = smooth_frac(arr, pc, umi)
        if sid == "P2_CRC":
            cache[name] = float(np.median(sf[reg == int(name.split("_")[1])]))
        bins_out[(sid, name)] = sf
    del X, st
    cache[f"{sid}_meta"] = (umi, xy, sp, sp_rank, reg)
    log(f"{sid} loaded")

for sid in SAMPLES:
    umi, xy, sp, sp_rank, reg = cache[f"{sid}_meta"]
    med_nn = 65.3
    for name in programs:
        sf = bins_out[(sid, name)]
        thr = cache[name]
        hot = np.flatnonzero((sf >= thr) & (umi > 0))
        row = {"sample": sid, "program": name, "genes": ";".join(programs[name]),
               "source_level": thr, "section_median_sf": float(np.median(sf)),
               "p999_sf": float(np.quantile(sf, 0.999)), "max_sf": float(sf.max()),
               "n_hot_bins": int(len(hot))}
        if len(hot) >= 5:
            lab = DBSCAN(eps=1.5 * med_nn, min_samples=5).fit_predict(xy[hot])
            sizes = pd.Series(lab[lab >= 0]).value_counts()
            row["n_hot_clusters_ge50"] = int((sizes >= 50).sum())
            row["largest_hot_cluster"] = int(sizes.iloc[0]) if len(sizes) else 0
            big = hot[np.isin(lab, sizes[sizes >= 50].index)] if len(sizes) else hot[:0]
            if len(big):
                row["hot_ge50_bins"] = int(len(big))
                row["hot_median_score_pooled"] = float(np.median(sp[big]))
                row["hot_median_score_pooled_pct"] = float(np.median(sp_rank[big]))
                row["hot_frac_score_pooled_gt1645"] = float((sp[big] > 1.645).mean())
                row["hot_frac_in_stage1_region"] = float((reg[big] >= 0).mean())
        rows.append(row)
        log(row)
pd.DataFrame(rows).to_csv(OUT / "cross_patient_program_scan.csv", index=False)
np.savez_compressed(OUT / "program_smoothed.npz",
                    **{f"{s}__{n}": v.astype(np.float16) for (s, n), v in bins_out.items()})

# ---- Xenium P1
xa = sc.read_h5ad(XEN)
log(f"xenium {xa.shape}; obs {list(xa.obs.columns)}; obsm {list(xa.obsm.keys())}")
Xx = xa.layers["counts"] if "counts" in xa.layers else xa.X
Xx = sparse.csr_matrix(Xx)
if not np.allclose(Xx.data[:1000], np.round(Xx.data[:1000])):
    log("WARNING: Xenium matrix is not integer counts")
vn = np.asarray(xa.var_names).astype(str)
lab = xa.obs["Level2"].astype(str).to_numpy()
conf = xa.obs["annotation_confidence"].to_numpy(float) if "annotation_confidence" in xa.obs else None
xy = None
for k in ("spatial", "X_spatial"):
    if k in xa.obsm:
        xy = np.asarray(xa.obsm[k], dtype=float)
if xy is None and {"x_centroid", "y_centroid"} <= set(xa.obs.columns):
    xy = xa.obs[["x_centroid", "y_centroid"]].to_numpy(float)
xrows, xtype = [], []
rng = np.random.default_rng(0)
for name in programs:
    genes = [g for g in dict.fromkeys(programs[name] + programs_full[name]) if g in set(vn)]
    row = {"program": name, "panel_genes": ";".join(genes), "n_panel_genes": len(genes)}
    if len(genes) >= 2:
        idx = [int(np.flatnonzero(vn == g)[0]) for g in genes]
        c = np.asarray(Xx[:, idx].sum(1)).ravel()
        high = c >= 3
        row.update({"n_cells": int(len(c)), "n_high": int(high.sum()),
                    "frac_high": float(high.mean()), "mean_count": float(c.mean()),
                    "p99_count": float(np.quantile(c, 0.99))})
        for g, j in zip(genes, idx):
            row[f"tx_{g}"] = int(Xx[:, j].sum())
        if conf is not None and high.any():
            row["conf_high_median"] = float(np.median(conf[high]))
            row["conf_all_median"] = float(np.median(conf))
        if xy is not None and high.sum() >= 10:
            tree = cKDTree(xy)
            _, nn = tree.query(xy[high], k=11)
            obs = high[nn[:, 1:]].mean()
            row["high_neighbour_frac"] = float(obs)
            row["high_neighbour_frac_expected"] = float(high.mean())
            hi_idx = np.flatnonzero(high)
            labx = DBSCAN(eps=30.0, min_samples=5).fit_predict(xy[hi_idx])
            sz = pd.Series(labx[labx >= 0]).value_counts()
            row["n_clusters_ge20_cells_eps30um"] = int((sz >= 20).sum())
            row["largest_cluster_cells"] = int(sz.iloc[0]) if len(sz) else 0
        if high.any():
            vc = pd.Series(lab[high]).value_counts(normalize=True)
            for t, v in vc.head(8).items():
                xtype.append({"program": name, "label": t, "frac_of_high": float(v),
                              "frac_of_all": float((lab == t).mean())})
    xrows.append(row)
    log(row)
pd.DataFrame(xrows).to_csv(OUT / "xenium_P1_programs.csv", index=False)
pd.DataFrame(xtype).to_csv(OUT / "xenium_P1_program_labels.csv", index=False)
log("done")
