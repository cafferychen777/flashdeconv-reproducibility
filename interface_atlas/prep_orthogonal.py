"""B1 pilot: build orthogonal (deconvolution-free) cell tables.

Xenium CRC P1/P2/P5: annotate cells by correlation to CRC Flex Level2 centroids (same method as
validation/rerun_v020/crc/xenium/xenium_crc_validation.py::load_and_annotate_xenium, applied to all three patients),
then map to lineages. SPATCH COAD/OV/HCC CODEX: published SPATCH cell annotation mapped to lineages.
Writes OUT/orth_<SECTION>.npz with um coordinates, lineage label per cell, epithelial-marker counts and total counts.
"""
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
from scipy import sparse

from lineages import MARKERS_EPI, MARKERS_EPI_CRC_XEN, MARKERS_EPI_GENERIC, LINEAGES, crc_lineage, codex_lineage, lung_lineage, spatch_fine_labels, spatch_lineage

S = Path("/scratch/user/cafferychen777")
OUT = S / "b1_pilot" / "results"
XEN = S / "FlashDeconv_data_archive/xenium_crc_cohort"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ref_centroids(genes):
    ref = ad.read_h5ad(S / "FlashDeconv/data/runtime_benchmark_c1/ref.h5ad")
    ref.var_names = ref.var["gene_symbols"].astype(str).values
    ref.var_names_make_unique()
    genes = [g for g in genes if g in set(ref.var_names)]
    ref = ref[:, genes].copy()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    types = sorted(ref.obs["cell_type"].astype(str).unique())
    lab = ref.obs["cell_type"].astype(str).to_numpy()
    C = np.vstack([np.asarray(ref.X[lab == t].mean(0)).ravel() for t in types]).astype(np.float64)
    return genes, types, C


def xenium(pid):
    d = XEN / f"{pid}_CRC_outs_min"
    x = sc.read_10x_h5(d / "cell_feature_matrix.h5")
    x.var_names_make_unique()
    cells = pq.read_table(d / "cells.parquet").to_pandas().set_index("cell_id")
    cells.index = cells.index.astype(str)
    common = x.obs_names.intersection(cells.index)
    x = x[common].copy()
    x = x[cells.loc[x.obs_names, "transcript_counts"].to_numpy() >= 10].copy()
    coords = cells.loc[x.obs_names, ["x_centroid", "y_centroid"]].to_numpy(np.float32)
    genes, types, C = ref_centroids(list(x.var_names))
    log(f"{pid}: {x.n_obs} cells, {len(genes)} overlapping genes")
    xs = x[:, genes].copy()
    sc.pp.normalize_total(xs, target_sum=1e4)
    sc.pp.log1p(xs)
    X = xs.X.toarray().astype(np.float64)
    Xn = (X - X.mean(1, keepdims=True)) / np.where(X.std(1, keepdims=True) == 0, 1, X.std(1, keepdims=True))
    Cn = (C - C.mean(1, keepdims=True)) / np.where(C.std(1, keepdims=True) == 0, 1, C.std(1, keepdims=True))
    R = Xn @ Cn.T / len(genes)
    best = R.argmax(1)
    conf = R[np.arange(len(best)), best]
    fine = np.array(types)[best]
    fine[conf < 0.15] = "Unassigned"
    lin = np.array([crc_lineage(f) if f != "Unassigned" else "Unassigned" for f in fine])
    vn = {g: i for i, g in enumerate(x.var_names)}
    epi = [g for g in MARKERS_EPI + MARKERS_EPI_CRC_XEN if g in vn]
    Xr = x.X.tocsc() if sparse.issparse(x.X) else sparse.csc_matrix(x.X)
    m = np.asarray(Xr[:, [vn[g] for g in epi]].sum(1)).ravel()
    t = np.asarray(Xr.sum(1)).ravel()
    log(f"{pid}: epithelial markers used {epi}; lineage counts " + str(pd.Series(lin).value_counts().to_dict()))
    # Diagnostic marker fold changes for annotation sanity
    for l, g in [("CD8 T", "CD8A"), ("CD4 T", "CD4"), ("B", "MS4A1"), ("Macrophage/Mono", "CD68"),
                 ("Endothelial", "PECAM1"), ("Fibroblast", "COL1A1"), ("Plasma", "JCHAIN"), ("Neutrophil", "FCGR3B"),
                 ("Mast", "CPA3"), ("Epithelial", "EPCAM")]:
        if g in vn:
            e = np.asarray(Xr[:, vn[g]].todense()).ravel()
            a = lin == l
            if a.any():
                log(f"   {l:16s} {g:7s} in={e[a].mean():.2f} out={e[~a].mean():.3f} FC={(e[a].mean()+.01)/(e[~a].mean()+.01):.1f}")
    np.savez_compressed(OUT / f"orth_CRC_{pid}.npz", coords_um=coords, lineage=lin, fine=fine, conf=conf.astype(np.float32),
                        epi_counts=m.astype(np.float32), total=t.astype(np.float32), modality="Xenium", epi_markers=np.array(epi))


def codex(ct):
    a = ad.read_h5ad(S / f"spatch/data/{ct}/proteome/adata_codex.h5ad")
    lin = codex_lineage(a.obs["annotation"].astype(str).to_numpy())
    epi = (lin == "Epithelial").astype(np.float32)
    log(f"{ct} CODEX: {a.n_obs} cells; " + str(pd.Series(a.obs['annotation']).value_counts().to_dict()))
    np.savez_compressed(OUT / f"orth_SPATCH_{ct}.npz", coords_um=np.asarray(a.obsm["spatial"], dtype=np.float32), lineage=lin,
                        fine=a.obs["annotation"].astype(str).to_numpy(), epi_counts=epi, total=np.ones_like(epi),
                        modality="CODEX", epi_markers=np.array(["PanCK-annotated Epithelial cell"]))


NEW = {"LUNG_X1": ("lung_postxenium/xenium_v1", "lung"), "LUNG_X5K": ("lung_postxenium/xenium_prime5k", "lung"),
       "OV10X": ("ovarian/xenium_prime5k", "ov"), "OV10X_v1": ("ovarian/xenium_v1", "ov")}


def centroids_generic(kind, genes):
    if kind == "lung":
        ref = ad.read_h5ad(S / "b1_pilot/data/lung_reference/lung_cancer_ffpe_flex_4pt_36k.h5ad")
        ref.var_names = ref.var["feature_name"].astype(str).values
        ref.var_names_make_unique()
        lab = ref.obs["Harmonised_Level4"].astype(str).to_numpy()
        lin_of = lung_lineage
    else:
        ref = ad.read_h5ad(S / "spatch/data/OV/adata.h5ad")
        ref.var_names_make_unique()
        lab = spatch_fine_labels(ref.obs)
        lin_of = spatch_lineage
    genes = [g for g in genes if g in set(ref.var_names)]
    ref = ref[:, genes].copy()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    types = sorted(set(lab))
    C = np.vstack([np.asarray(ref.X[lab == t].mean(0)).ravel() for t in types]).astype(np.float64)
    return genes, types, C, lin_of


def xenium_generic(sec):
    sub, kind = NEW[sec]
    d = S / "b1_pilot/data" / sub
    x = sc.read_10x_h5(d / "cell_feature_matrix.h5")
    x.var_names_make_unique()
    cells = pq.read_table(d / "cells.parquet").to_pandas().set_index("cell_id")
    cells.index = cells.index.astype(str)
    x = x[x.obs_names.intersection(cells.index)].copy()
    x = x[cells.loc[x.obs_names, "transcript_counts"].to_numpy() >= 10].copy()
    coords = cells.loc[x.obs_names, ["x_centroid", "y_centroid"]].to_numpy(np.float32)
    genes, types, C, lin_of = centroids_generic(kind, list(x.var_names))
    log(f"{sec}: {x.n_obs} cells, {len(genes)} overlapping genes, {len(types)} ref types")
    xs = x[:, genes].copy()
    sc.pp.normalize_total(xs, target_sum=1e4)
    sc.pp.log1p(xs)
    Cn = (C - C.mean(1, keepdims=True)) / np.where(C.std(1, keepdims=True) == 0, 1, C.std(1, keepdims=True))
    best = np.empty(xs.n_obs, int); conf = np.empty(xs.n_obs)
    Xc = xs.X.tocsr()
    for s0 in range(0, xs.n_obs, 20000):
        X = Xc[s0:s0 + 20000].toarray().astype(np.float64)
        Xn = (X - X.mean(1, keepdims=True)) / np.where(X.std(1, keepdims=True) == 0, 1, X.std(1, keepdims=True))
        R = Xn @ Cn.T / len(genes)
        best[s0:s0 + 20000] = R.argmax(1); conf[s0:s0 + 20000] = R.max(1)
    fine = np.array(types)[best]
    fine[conf < 0.15] = "Unassigned"
    lin = np.array([lin_of(f) if f != "Unassigned" else "Unassigned" for f in fine])
    vn = {g: i for i, g in enumerate(x.var_names)}
    epi = [g for g in MARKERS_EPI_GENERIC if g in vn]
    Xr = x.X.tocsc()
    m = np.asarray(Xr[:, [vn[g] for g in epi]].sum(1)).ravel()
    t = np.asarray(Xr.sum(1)).ravel()
    log(f"{sec}: epithelial markers {epi}; lineage counts " + str(pd.Series(lin).value_counts().to_dict()))
    for l, g in [("CD8 T", "CD8A"), ("CD4 T", "CD4"), ("Treg", "FOXP3"), ("B", "MS4A1"), ("Macrophage/Mono", "CD68"),
                 ("Endothelial", "PECAM1"), ("Fibroblast", "COL1A1"), ("Pericyte/SMC", "ACTA2"), ("Plasma", "JCHAIN"),
                 ("Neutrophil", "CSF3R"), ("Mast", "CPA3"), ("DC", "LAMP3"), ("NK", "KLRD1"), ("Epithelial", "EPCAM")]:
        if g in vn:
            e = np.asarray(Xr[:, vn[g]].todense()).ravel(); a = lin == l
            if a.any():
                log(f"   {l:16s} {g:7s} in={e[a].mean():.2f} out={e[~a].mean():.3f} FC={(e[a].mean()+.01)/(e[~a].mean()+.01):.1f}")
    np.savez_compressed(OUT / f"orth_{sec}.npz", coords_um=coords, lineage=lin, fine=fine, conf=conf.astype(np.float32),
                        epi_counts=m.astype(np.float32), total=t.astype(np.float32), modality="Xenium", epi_markers=np.array(epi))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        for p in ["P1", "P2", "P5"]:
            xenium(p)
        for c in ["COAD", "OV", "HCC"]:
            codex(c)
    for a in args:
        xenium_generic(a)
