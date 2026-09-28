"""C1 runtime benchmark: build shared inputs from real Visium HD CRC 8 um bins.

Inputs (arseven):
  - Visium HD 8 um count matrices retained in
    analysis/crc_cohort_results/{P1,P2,P5}_CRC_deconv.h5ad (raw integer counts in X,
    1,595,565 bins in total, 18,085 features).
  - Chromium Flex reference (10x HumanColonCancer_Flex_Multiplex) + SingleCell_MetaData.

Outputs (DATA_DIR):
  - ref.h5ad            : reference, QCFilter == Keep, Level2 types with > 25 cells,
                          at most 10,000 cells per type (spacexr's default n_max_cells),
                          common genes only, raw counts (CSR float32).
  - ref_meta.csv        : barcode, cell_type, patient (for R methods).
  - st_{scale}.h5ad     : nested random subsets (seed 0) of the pooled 8 um bins,
                          common genes only, raw counts (CSR float32), obsm['spatial']
                          in um with per-patient x-offsets so tissues never overlap.
  - prep_summary.json
"""
import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

PROJ = Path("/scratch/user/cafferychen777/FlashDeconv")
ST_DIR = PROJ / "analysis/crc_cohort_results"
REF_H5 = PROJ / "data/visium_hd_crc_cohort/scRNA_ref/HumanColonCancer_Flex_filtered_feature_bc_matrix.h5"
REF_META = PROJ / "data/visium_hd_crc_cohort/metadata/SingleCell_MetaData.csv.gz"
DATA_DIR = Path(os.environ.get("C1_DATA", PROJ / "data/runtime_benchmark_c1"))
SAMPLES = ["P1_CRC", "P2_CRC", "P5_CRC"]
SCALES = [10_000, 100_000, 300_000, 1_000_000]
SEED = 0
MAX_CELLS_PER_TYPE = 10_000
MIN_CELLS_PER_TYPE = 26  # 10x Deconvolution.R keeps Level2 clusters with > 25 cells
BIN_UM = 8.0


def read_decode(ds):
    arr = ds[()]
    if arr.dtype.kind in ("O", "S"):
        arr = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def read_spatial_sample(path):
    """Read raw counts, gene ids and array coords from a retained h5ad without obs bloat."""
    with h5py.File(path, "r") as f:
        shape = tuple(f["X"].attrs["shape"])
        X = sparse.csr_matrix(
            (f["X/data"][()].astype(np.float32), f["X/indices"][()], f["X/indptr"][()]),
            shape=shape,
        )
        barcodes = read_decode(f["obs/_index"])
        gene_ids = read_decode(f["var/gene_ids"])
        array_coords = f["obsm/array_coords"][()].astype(np.float64)
    return X, barcodes, gene_ids, array_coords


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "samples": {}, "scales": {}}

    # ---------------- reference ----------------
    ref = sc.read_10x_h5(REF_H5)
    ref.var["gene_symbols"] = ref.var_names.astype(str)
    ref.var_names = ref.var["gene_ids"].astype(str).to_numpy()
    meta = pd.read_csv(REF_META).set_index("Barcode")
    common_cells = ref.obs_names.intersection(meta.index)
    ref = ref[common_cells].copy()
    keep = (meta.loc[ref.obs_names, "QCFilter"] == "Keep").to_numpy()
    ref = ref[keep].copy()
    ref.obs["cell_type"] = meta.loc[ref.obs_names, "Level2"].astype(str).to_numpy()
    ref.obs["patient"] = meta.loc[ref.obs_names, "Patient"].astype(str).to_numpy()
    counts = ref.obs["cell_type"].value_counts()
    keep_types = counts.index[counts >= MIN_CELLS_PER_TYPE]
    ref = ref[ref.obs["cell_type"].isin(keep_types)].copy()
    summary["ref_cells_qc_keep"] = int(ref.n_obs)

    rng = np.random.default_rng(SEED)
    sel = []
    for ct, idx in ref.obs.groupby("cell_type", observed=True).indices.items():
        if len(idx) > MAX_CELLS_PER_TYPE:
            idx = rng.choice(idx, MAX_CELLS_PER_TYPE, replace=False)
        sel.append(np.sort(idx))
    ref = ref[np.sort(np.concatenate(sel))].copy()

    # ---------------- spatial ----------------
    mats, coords_list, names, sample_lab = [], [], [], []
    gene_ids0 = None
    x_offset = 0.0
    for s in SAMPLES:
        X, bc, gids, arr = read_spatial_sample(ST_DIR / f"{s}_deconv.h5ad")
        if gene_ids0 is None:
            gene_ids0 = gids
        elif not np.array_equal(gene_ids0, gids):
            raise ValueError(f"Gene order differs in {s}")
        # array_coords are (array_col, array_row) in 8 um steps -> um; offset x by patient
        xy = arr * BIN_UM
        xy[:, 0] += x_offset - xy[:, 0].min()
        x_offset = xy[:, 0].max() + 5000.0
        mats.append(X)
        coords_list.append(xy)
        names.append(np.char.add(f"{s}:", bc.astype(str)))
        sample_lab.append(np.repeat(s, X.shape[0]))
        nnz = X.getnnz(axis=1)
        umi = np.asarray(X.sum(axis=1)).ravel()
        summary["samples"][s] = {
            "n_bins": int(X.shape[0]),
            "median_umi": float(np.median(umi)),
            "median_genes": float(np.median(nnz)),
        }

    # common genes (Ensembl), reference order
    st_pos = {g: i for i, g in enumerate(gene_ids0)}
    common = [g for g in ref.var_names if g in st_pos]
    ref = ref[:, common].copy()
    ref.X = sparse.csr_matrix(ref.X, dtype=np.float32)
    st_cols = np.array([st_pos[g] for g in common])
    sym = ref.var["gene_symbols"].to_numpy()

    Y = sparse.vstack(mats, format="csr")[:, st_cols].tocsr()
    Y.sort_indices()
    del mats
    coords = np.vstack(coords_list)
    obs_names = np.concatenate(names)
    samples = np.concatenate(sample_lab)
    n_total = Y.shape[0]
    summary["n_bins_total"] = int(n_total)
    summary["n_genes_common"] = len(common)

    ref.obs[["cell_type", "patient"]].rename_axis("barcode").to_csv(DATA_DIR / "ref_meta.csv")
    ref.obs = ref.obs[["cell_type", "patient"]]
    ref.var = pd.DataFrame({"gene_symbols": sym}, index=pd.Index(common))
    ref.write_h5ad(DATA_DIR / "ref.h5ad")
    summary["ref_cells"] = int(ref.n_obs)
    summary["ref_types"] = int(ref.obs["cell_type"].nunique())
    summary["ref_cells_per_type"] = ref.obs["cell_type"].value_counts().to_dict()

    perm = np.random.default_rng(SEED).permutation(n_total)
    for n in SCALES:
        idx = np.sort(perm[: min(n, n_total)])
        a = ad.AnnData(
            X=Y[idx],
            obs=pd.DataFrame({"sample": samples[idx]}, index=pd.Index(obs_names[idx])),
            var=pd.DataFrame({"gene_symbols": sym}, index=pd.Index(common)),
        )
        a.obsm["spatial"] = coords[idx]
        out = DATA_DIR / f"st_{n}.h5ad"
        a.write_h5ad(out)
        summary["scales"][str(n)] = {
            "n_bins": int(a.n_obs),
            "nnz": int(a.X.nnz),
            "per_sample": pd.Series(samples[idx]).value_counts().to_dict(),
            "file_gb": round(out.stat().st_size / 1e9, 3),
        }
        print(f"wrote {out} ({a.n_obs} bins)", flush=True)

    summary["prep_seconds"] = round(time.time() - t0, 1)
    with open(DATA_DIR / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
