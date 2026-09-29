"""Shared CRC analysis functions (verbatim from validation/crc_seed_stability/run_patient.py,
which reproduces all manuscript CRC numbers from the archived proportions).
Only the ablation-variant import and main() were removed; marker_fc additionally
records mean UMI, and lineage_marker_validation records median / % nonzero
(needed by the figure scripts)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

PROJ = Path("/scratch/user/cafferychen777/FlashDeconv")
from flashdeconv import FlashDeconv  # noqa: E402
from flashdeconv.io import prepare_data  # noqa: E402

ST_DIR = PROJ / "analysis" / "crc_cohort_results"
REF_H5 = PROJ / "data/visium_hd_crc_cohort/scRNA_ref/HumanColonCancer_Flex_filtered_feature_bc_matrix.h5"
REF_META = PROJ / "data/visium_hd_crc_cohort/metadata/SingleCell_MetaData.csv.gz"
AUX = PROJ / "data" / "crc_seed_stability"
OUT_ROOT = PROJ / "results" / "crc_seed_stability"

SEEDS = [42, 1, 2, 3, 4]
FD_KW = dict(sketch_dim=512, lambda_spatial="auto", rho_sparsity=0.01, n_hvg=2000,
             n_markers_per_type=50, k_neighbors=6, max_iter=100, tol=1e-4,
             preprocess="log_cpm")

LINEAGE_MAP = {
    "Tumor I": "Tumor", "Tumor II": "Tumor", "Tumor III": "Tumor",
    "Tumor IV": "Tumor", "Tumor V": "Tumor",
    "CAF": "Stromal", "Myofibroblast": "Stromal", "Fibroblast": "Stromal",
    "Proliferating Fibroblast": "Stromal", "Vascular Fibroblast": "Stromal",
    "Smooth Muscle": "Stromal", "SM Stress Response": "Stromal",
    "vSM": "Stromal", "Pericytes": "Stromal",
    "Macrophage": "Immune", "Proliferating Macrophages": "Immune",
    "CD8 T cell": "Immune", "CD4 T cell": "Immune",
    "NK": "Immune", "Plasma": "Immune", "Mature B": "Immune",
    "Memory B": "Immune", "Neutrophil": "Immune", "Mast": "Immune",
    "pDC": "Immune", "mRegDC": "Immune", "cDC I": "Immune",
    "Proliferating Immune II": "Immune",
    "Enterocyte": "Epithelial", "Goblet": "Epithelial", "Tuft": "Epithelial",
    "Epithelial": "Epithelial", "Neuroendocrine": "Epithelial",
    "Endothelial": "Endothelial", "Lymphatic Endothelial": "Endothelial",
    "Enteric Glial": "Other", "Adipocyte": "Other", "Unknown III (SM)": "Other",
}

# neutrophil_microdomain_analysis.py groups and Config (copied verbatim)
INNATE_IMMUNE = ["Macrophage", "Proliferating Macrophages", "Mast", "Neutrophil", "NK"]
ADAPTIVE_IMMUNE = ["CD4 T cell", "CD8 T cell", "Mature B", "Memory B", "Plasma"]
DC_TYPES = ["mRegDC", "cDC I", "pDC"]
STROMAL = ["CAF", "Endothelial", "Vascular Fibroblast", "Fibroblast",
           "Lymphatic Endothelial", "Smooth Muscle", "Pericytes"]
EPITHELIAL = ["Enterocyte", "Goblet", "Epithelial"]
TUMOR = ["Tumor I", "Tumor II", "Tumor III", "Tumor IV", "Tumor V"]
NICHE_CELL_TYPES = INNATE_IMMUNE + ADAPTIVE_IMMUNE + DC_TYPES + STROMAL + EPITHELIAL + TUMOR
NEUT_THR = 0.10
DBSCAN_EPS_FACTOR = 8.0
DBSCAN_MIN_SAMPLES = 20
NEIGH_RADIUS_FACTOR = 15.0
MIN_NEIGHBOR_BINS = 50
UMI_HIGH = 200
TUMOR_NICHE_THRESHOLD = 0.15

NEUT_MARKERS = ["S100A8", "S100A9", "FCGR3B", "CSF3R", "CXCR1", "CXCR2"]
NEG_MARKERS = ["CD3D", "CD79A", "KRT20", "COL1A1"]
LR_GENES = ["LAMP3", "IDO1", "PECAM1", "CD163", "CCR7", "CD274", "S100A8", "S100A9"]
LR_RADIUS = 100  # same units as original script (full-res pixel coordinates)
IMMUNE_MARKERS = ["PTPRC", "CD3D", "CD68", "CD14", "MS4A1", "MZB1", "JCHAIN"]
STROMAL_MARKERS = ["COL1A1", "COL1A2", "FAP", "ACTA2"]
BANDS = [(-500, -200), (-200, -100), (-100, -50), (-50, 0),
         (0, 50), (50, 100), (100, 200), (200, 500)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- loading

def load_reference():
    ref = sc.read_10x_h5(REF_H5)  # gene symbols as var_names (original run)
    meta = pd.read_csv(REF_META, compression="gzip").set_index("Barcode")
    common = ref.obs_names.intersection(meta.index)
    ref = ref[common].copy()
    ref = ref[(meta.loc[ref.obs_names, "QCFilter"] == "Keep").to_numpy()].copy()
    ref.obs["Level2"] = meta.loc[ref.obs_names, "Level2"].values
    log(f"reference: {ref.n_obs:,} cells, {ref.obs['Level2'].nunique()} Level2 types")
    return ref


def gene_columns(X, var_names, genes):
    """Dense float columns for the requested genes (first occurrence of a symbol)."""
    pos = {}
    for i, g in enumerate(var_names):
        pos.setdefault(g, i)
    genes = [g for g in genes if g in pos]
    idx = [pos[g] for g in genes]
    sub = X[:, idx]
    sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    return {g: sub[:, j].astype(float) for j, g in enumerate(genes)}


# ---------------------------------------------------------------- metrics

def jsd_rows(P, Q, chunk=100_000):
    out = np.empty(P.shape[0])
    for s in range(0, P.shape[0], chunk):
        p = np.clip(P[s:s + chunk].astype(np.float64), 0, None)
        q = np.clip(Q[s:s + chunk].astype(np.float64), 0, None)
        p /= p.sum(1, keepdims=True) + 1e-300
        q /= q.sum(1, keepdims=True) + 1e-300
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kp = np.where(p > 0, p * np.log2(p / m), 0.0).sum(1)
            kq = np.where(q > 0, q * np.log2(q / m), 0.0).sum(1)
        out[s:s + chunk] = 0.5 * (kp + kq)
    return out


def compare(P, R, types, neut_j):
    rows = []
    for j, ct in enumerate(types):
        a, b = P[:, j].astype(np.float64), R[:, j].astype(np.float64)
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
        rows.append({"cell_type": ct, "pearson": r,
                     "mean_abs_diff": float(np.abs(a - b).mean())})
    j = jsd_rows(P, R)
    hp, hr = P[:, neut_j] >= NEUT_THR, R[:, neut_j] >= NEUT_THR
    inter, union = int((hp & hr).sum()), int((hp | hr).sum())
    summ = {"median_jsd": float(np.median(j)), "mean_jsd": float(j.mean()),
            "p90_jsd": float(np.quantile(j, 0.9)),
            "dominant_agreement": float((P.argmax(1) == R.argmax(1)).mean()),
            "hotspot_jaccard": inter / union if union else np.nan,
            "hotspot_n_fit": int(hp.sum()), "hotspot_n_ref": int(hr.sum()),
            "hotspot_intersection": inter}
    return pd.DataFrame(rows), summ


def median_nn(coords, sample_size=20_000, seed=0):
    n = coords.shape[0]
    sample = coords if n <= sample_size else coords[
        np.random.default_rng(seed).choice(n, size=sample_size, replace=False)]
    d, _ = cKDTree(sample).query(sample, k=2)
    return float(np.median(d[:, 1]))


def knn_enrichment(P, types, tree, coords, focal_types, thr=0.10, k=30, max_focal=10_000):
    """spatial_neighborhood_enrichment() of crc_cohort_atlas_analysis.py."""
    gm = P.mean(0)
    rows = []
    for ct in focal_types:
        j = types.index(ct)
        idx = np.where(P[:, j] > thr)[0]
        if len(idx) < 10:
            continue
        rng = np.random.default_rng(42)
        if len(idx) > max_focal:
            idx = rng.choice(idx, size=max_focal, replace=False)
        _, nn = tree.query(coords[idx], k=k + 1)
        avg = P[nn[:, 1:].ravel()].reshape(len(idx), k, -1).mean(1).mean(0)
        enr = (avg + 1e-6) / (gm + 1e-6)
        for jj, nt in enumerate(types):
            rows.append({"focal_type": ct, "neighbor_type": nt, "threshold": thr,
                         "focal_n_bins": len(idx), "enrichment_ratio": float(enr[jj]),
                         "log2_enrichment": float(np.log2(enr[jj]))})
    return rows


def aggregates(P, types, coords, tree, med, eps):
    """detect_aggregates() + characterize_niche() of neutrophil_microdomain_analysis.py."""
    df = pd.DataFrame(P, columns=types)
    hot_idx = np.flatnonzero(df["Neutrophil"].values >= NEUT_THR)
    if len(hot_idx) < DBSCAN_MIN_SAMPLES:
        return pd.DataFrame(), 0
    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords[hot_idx])
    avail = [ct for ct in NICHE_CELL_TYPES if ct in types]
    gmean = df[avail].mean()
    r_neigh = NEIGH_RADIUS_FACTOR * med
    rows = []
    clusters = [c for c in np.unique(labels) if c != -1]
    for c in clusters:
        member_idx = hot_idx[labels == c]
        centroid = coords[member_idx].mean(0)
        neigh = tree.query_ball_point(centroid, r=r_neigh)
        ms = set(member_idx.tolist())
        sur = [i for i in neigh if i not in ms]
        if len(sur) < MIN_NEIGHBOR_BINS:
            continue
        local = df.iloc[sur][avail].mean()
        enr = np.log2((local + 1e-12) / (gmean + 1e-12))
        row = {"cluster": int(c), "n_hotspots": len(member_idx), "n_neighbors": len(sur),
               "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
               "total_niche_tumor": float(local[[t for t in TUMOR if t in avail]].sum())}
        for ct in avail:
            row[f"niche_log2_{ct}"] = float(enr[ct])
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["location"] = np.where(out["total_niche_tumor"] > TUMOR_NICHE_THRESHOLD,
                                   "tumor-proximal", "stromal-resident")
    return out, len(clusters)


def marker_fc(hot, umi, expr):
    rows = []
    bg = ~hot
    hh, bh = hot & (umi >= UMI_HIGH), bg & (umi >= UMI_HIGH)
    for g in NEUT_MARKERS + NEG_MARKERS:
        if g not in expr:
            continue
        e = expr[g]
        f = lambda a, b: (e[a].mean() + 1e-6) / (e[b].mean() + 1e-6) if a.any() and b.any() else np.nan
        rows.append({"gene": g, "marker_class": "neutrophil" if g in NEUT_MARKERS else "negative",
                     "fold_change": f(hot, bg), "fold_change_high_umi": f(hh, bh),
                     "n_hot": int(hot.sum()), "n_hot_high_umi": int(hh.sum()),
                     "hotspot_mean_expr": float(e[hot].mean()), "background_mean_expr": float(e[bg].mean()),
                     "mean_umi_hotspot": float(umi[hot].mean()), "mean_umi_background": float(umi[bg].mean())})
    return rows


def lr_enrichment(hot, coords, tree, expr):
    """neighborhood_gene_enrichment() of cross_cohort_evidence/03_lr_analysis.py."""
    hot_idx = np.where(hot)[0]
    nb = tree.query_ball_point(coords[hot_idx], r=LR_RADIUS)
    neigh = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in nb]))
    neigh_only = np.setdiff1d(neigh, hot_idx)
    bg = np.setdiff1d(np.setdiff1d(np.arange(len(hot)), hot_idx), neigh_only)
    rows = []
    for g in LR_GENES:
        if g not in expr:
            continue
        e = expr[g]
        he, ne, be = e[hot_idx], e[neigh_only], e[bg]
        if be.mean() == 0 and ne.mean() == 0:
            continue
        p = 1.0
        if ne.sum() > 0 and be.sum() > 0:
            rng = np.random.default_rng(42)
            bsub = rng.choice(be, size=min(10000, len(bg)), replace=False)
            nsub = rng.choice(ne, size=min(10000, len(neigh_only)), replace=False) \
                if len(neigh_only) > 10000 else ne
            try:
                p = stats.mannwhitneyu(nsub, bsub, alternative="greater").pvalue
            except ValueError:
                p = 1.0
        rows.append({"gene": g, "fold_neighborhood_vs_bg": ne.mean() / be.mean() if be.mean() > 0 else np.nan,
                     "fold_hotspot_vs_bg": he.mean() / be.mean() if be.mean() > 0 else np.nan,
                     "p_neigh_vs_bg": float(p), "n_hot": len(hot_idx),
                     "n_neighborhood": len(neigh_only)})
    return rows


def rctd_breakdown(hot, rctd):
    cls, l1, l2 = rctd["cls"], rctd["l1"], rctd["l2"]
    sing = cls == "singlet"
    doub = np.isin(cls, ["doublet_certain", "doublet_uncertain"])
    rej = cls == "reject"
    na = (cls == "NA") | (cls == "")
    n = int(hot.sum())
    return {"n_hotspot": n, "singlet": int((hot & sing).sum()), "doublet": int((hot & doub).sum()),
            "reject": int((hot & rej).sum()), "NA": int((hot & na).sum()),
            "neut_singlet_label1": int((hot & sing & (l1 == "Neutrophil")).sum()),
            "neut_singlet_label2": int((hot & sing & (l2 == "Neutrophil")).sum()),
            "neut_doublet": int((hot & doub & ((l1 == "Neutrophil") | (l2 == "Neutrophil"))).sum())}


def lineage_marker_validation(P, types, rctd, lin_expr):
    """marker_gene_validation() per sample: per-gene mean expression by category."""
    fd_lin = pd.Series(np.array(types)[P.argmax(1)]).map(LINEAGE_MAP).fillna("Other").to_numpy()
    sing = rctd["cls"] == "singlet"
    # DeconvolutionLabel1 is RCTD's singlet (first-type) call; Label2 is the runner-up type.
    rl = pd.Series(rctd["l1"]).map(LINEAGE_MAP).fillna("Other").to_numpy()
    cats = {"Agreed_Immune": sing & (rl == "Immune") & (fd_lin == "Immune"),
            "Agreed_Stromal": sing & (rl == "Stromal") & (fd_lin == "Stromal"),
            "RCTD_Immune_FD_Stromal": sing & (rl == "Immune") & (fd_lin == "Stromal"),
            "RCTD_Stromal_FD_Immune": sing & (rl == "Stromal") & (fd_lin == "Immune")}
    rows = []
    for g, e in lin_expr.items():
        for c, m in cats.items():
            if m.any():
                rows.append({"gene": g, "category": c, "n_bins": int(m.sum()),
                             "mean_expression": float(e[m].mean()),
                             "median_expression": float(np.median(e[m])),
                             "pct_nonzero": float(100 * (e[m] > 0).mean()),
                             "marker_class": "Immune" if g in IMMUNE_MARKERS else "Stromal"})
    return rows


def boundary_distance(barcodes, coords, sample):
    per = pd.read_csv(AUX / "periphery" / f"{sample.replace('_', '')}_periphery.csv.gz").set_index("barcode")
    periphery = np.full(len(barcodes), "", dtype=object)
    ok = pd.Index(barcodes).isin(per.index)
    periphery[ok] = per.loc[barcodes[ok], "Periphery"].values
    ann = periphery != ""
    if (~ann).any():
        _, nn = cKDTree(coords[ann]).query(coords[~ann], k=1)
        periphery[~ann] = periphery[ann][nn]
    is_t = periphery == "Tumor"
    n = len(barcodes)
    sub = np.random.default_rng(42).choice(n, size=min(10000, n), replace=False)
    d, _ = cKDTree(coords).query(coords[sub], k=2)
    umpp = 8.0 / np.median(d[:, 1])
    t_idx = np.where(is_t)[0]
    d_nt, _ = cKDTree(coords[~is_t]).query(coords[t_idx], k=1)
    b_idx = t_idx[d_nt <= 20.0 / umpp]
    db, _ = cKDTree(coords[b_idx]).query(coords, k=1)
    db = db * umpp
    return np.where(is_t, -db, db)


def boundary_bands(P, types, signed):
    rows = []
    for lo, hi in BANDS:
        m = (signed >= lo) & (signed < hi)
        if m.sum() < 10:
            continue
        mp = P[m].mean(0)
        for j, ct in enumerate(types):
            rows.append({"distance_min_um": lo, "distance_max_um": hi, "cell_type": ct,
                         "mean_proportion": float(mp[j]), "n_bins": int(m.sum())})
    return rows


