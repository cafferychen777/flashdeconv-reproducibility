"""B1 pilot: run FlashDeconv (final package, default arguments) on one Visium HD 8 um section.

Usage: python run_fd.py SECTION   (CRC_P1, CRC_P2, CRC_P5, SPATCH_COAD, SPATCH_OV, SPATCH_HCC, LUNG_X1, LUNG_X5K, OV10X)
Writes OUT/fd_<SECTION>.npz with um coordinates, fine + lineage proportions (float16), marker raw counts, total counts
and timing.
"""
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import flashdeconv as fd
from lineages import MARKERS_EPI, MARKERS_DIAG, LINEAGES, crc_lineage, lung_lineage, spatch_fine_labels, spatch_lineage
import pyarrow.parquet as pq
import scanpy as sc

S = Path("/scratch/user/cafferychen777")
OUT = S / "b1_pilot" / "results"
OUT.mkdir(parents=True, exist_ok=True)
HD10X = {"LUNG_X1": "lung_postxenium/hd_post_xenium_v1_exp1", "LUNG_X5K": "lung_postxenium/hd_post_xenium_prime5k_exp2",
         "OV10X": "ovarian/hd_ff_min_depth"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_section(sec):
    if sec in HD10X:
        d = S / "b1_pilot/data" / HD10X[sec] / "binned_outputs/square_008um"
        st = sc.read_10x_h5(d / "filtered_feature_bc_matrix.h5")
        pos = pq.read_table(d / "spatial/tissue_positions.parquet").to_pandas().set_index("barcode")
        st.obsm["spatial"] = pos.loc[st.obs_names, ["array_col", "array_row"]].to_numpy(float) * 8.0
        st = ad.AnnData(X=st.X, obs=pd.DataFrame(index=st.obs_names), var=pd.DataFrame(index=st.var_names),
                        obsm={"spatial": st.obsm["spatial"]})
        if sec.startswith("LUNG"):
            ref = ad.read_h5ad(S / "b1_pilot/data/lung_reference/lung_cancer_ffpe_flex_4pt_36k.h5ad")
            ref.var_names = ref.var["feature_name"].astype(str).values
            ref.var_names_make_unique()
            ref.obs["fine"] = ref.obs["Harmonised_Level4"].astype(str)
            lin_of = lung_lineage
        else:
            ref = ad.read_h5ad(S / "spatch/data/OV/adata.h5ad")
            ref.obs["fine"] = spatch_fine_labels(ref.obs)
            lin_of = spatch_lineage
    elif sec.startswith("CRC_"):
        pid = sec.split("_")[1]
        st = ad.read_h5ad(S / f"FlashDeconv/analysis/crc_cohort_results/{pid}_CRC_deconv.h5ad")
        st = ad.AnnData(X=st.X, obs=pd.DataFrame(index=st.obs_names), var=pd.DataFrame(index=st.var_names),
                        obsm={"spatial": np.asarray(st.obsm["array_coords"], dtype=float) * 8.0})
        ref = ad.read_h5ad(S / "FlashDeconv/data/runtime_benchmark_c1/ref.h5ad")
        ref.var_names = ref.var["gene_symbols"].astype(str).values
        ref.var_names_make_unique()
        ref.obs["fine"] = ref.obs["cell_type"].astype(str)
        lin_of = crc_lineage
    else:
        ct = sec.split("_")[1]
        st = ad.read_h5ad(S / f"spatch/data/{ct}/transcriptome/adata.h5ad")
        st = ad.AnnData(X=st.X, obs=pd.DataFrame(index=st.obs_names), var=pd.DataFrame(index=st.var_names),
                        obsm={"spatial": np.asarray(st.obsm["spatial"], dtype=float)})
        ref = ad.read_h5ad(S / f"spatch/data/{ct}/adata.h5ad")
        ref.obs["fine"] = spatch_fine_labels(ref.obs)
        lin_of = spatch_lineage
    st.var_names_make_unique()
    if not sparse.issparse(st.X):
        st.X = sparse.csr_matrix(st.X)
    st.X = st.X.tocsr().astype(np.float32)
    if not sparse.issparse(ref.X):
        ref.X = sparse.csr_matrix(ref.X)
    ref.X = ref.X.astype(np.float32)
    return st, ref, lin_of


def main(sec):
    t0 = time.perf_counter()
    st, ref, lin_of = load_section(sec)
    t_load = time.perf_counter() - t0
    tot = np.asarray(st.X.sum(1)).ravel()
    keep = tot > 0
    st = st[keep].copy()
    log(f"{sec}: ST {st.shape}, ref {ref.shape}, {ref.obs['fine'].nunique()} fine types; load {t_load:.1f}s")
    log("ref counts per fine type: " + json.dumps(ref.obs["fine"].value_counts().to_dict()))
    t1 = time.perf_counter()
    fd.tl.deconvolve(st, ref, cell_type_key="fine")
    t_fd = time.perf_counter() - t1
    P = st.obsm["flashdeconv"]
    names = list(P.columns)
    P = P.to_numpy(dtype=np.float32)
    params = st.uns.get("flashdeconv_params", {})
    log(f"deconvolve {t_fd:.1f}s; params: " + str({k: v for k, v in params.items() if k != 'cell_type_names'}))
    lin = np.array([lin_of(n) for n in names])
    L = np.zeros((P.shape[0], len(LINEAGES)), dtype=np.float32)
    for j, l in enumerate(LINEAGES):
        m = lin == l
        if m.any():
            L[:, j] = P[:, m].sum(1)
    log("fine->lineage: " + json.dumps(dict(zip(names, lin.tolist()))))
    log("mean lineage fractions: " + json.dumps({l: round(float(v), 4) for l, v in zip(LINEAGES, L.mean(0))}))
    vn = {g: i for i, g in enumerate(st.var_names)}
    genes = list(dict.fromkeys(g for g in MARKERS_EPI + MARKERS_DIAG if g in vn))
    M = st.X[:, [vn[g] for g in genes]].toarray().astype(np.float32)
    np.savez_compressed(
        OUT / f"fd_{sec}.npz", coords_um=st.obsm["spatial"].astype(np.float32), fine=P.astype(np.float16),
        fine_names=np.array(names), lineage=L.astype(np.float16), lineage_names=np.array(LINEAGES),
        marker_counts=M, marker_names=np.array(genes), total_counts=np.asarray(st.X.sum(1)).ravel().astype(np.float32),
        t_deconvolve_s=t_fd, t_load_s=t_load, n_bins=st.n_obs, n_genes=st.n_vars,
        n_iterations=params.get("n_iterations", -1), converged=params.get("converged", False),
        lambda_spatial=params.get("lambda_spatial", np.nan), fd_version=fd.__version__)
    log(f"saved; total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1])
