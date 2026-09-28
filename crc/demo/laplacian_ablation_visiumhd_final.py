"""
Rerun of the Supp Note 3 Visium HD CRC Laplacian ablation (8 um only) with
FlashDeconv v0.2.0.

Original: validation/ablation_laplacian_visiumhd.py (arseven jobs 1442159/1442162).
Same data (10x Visium HD Human Colon Cancer, square_008um, in_tissue bins),
same reference (GSE132465, Cell_type, 6 major types), same FlashDeconv settings
and the same Moran's I definition (kNN k=6, row-standardized lag). The only
change is the package version (v0.2.0 default gene_weighting="expected").
tissue_positions.parquet was converted to csv.gz because the v0.2.0 env has no
pyarrow; content is unchanged.
"""

import sys as _sys  # final rerun: fit-logging hook (validation/rerun_final/fdfinal.py)
_sys.path.insert(0, "/scratch/user/cafferychen777/fd_final/code")
import fdfinal  # noqa: E402,F401
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import entropy
from pathlib import Path

import flashdeconv
from flashdeconv.core.deconv import FlashDeconv
from flashdeconv.io.loader import load_spatial_data, load_reference, align_genes

DATA_DIR = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])
REF_PATH = DATA_DIR / "GSE132465_CRC_reference.h5ad"


def load_visium_hd_crc_8um():
    adata = sc.read_10x_h5(str(DATA_DIR / "filtered_feature_bc_matrix.h5"))
    adata.var_names_make_unique()
    positions = pd.read_csv(DATA_DIR / "tissue_positions_008um.csv.gz").set_index("barcode")
    common_barcodes = adata.obs_names.intersection(positions.index)
    adata = adata[common_barcodes].copy()
    in_tissue_mask = positions.loc[adata.obs_names, "in_tissue"] == 1
    adata = adata[in_tissue_mask.values].copy()
    adata.obsm["spatial"] = positions.loc[
        adata.obs_names, ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].values
    print(f"  Loaded {adata.n_obs:,} spots, {adata.n_vars:,} genes")
    return adata


def compute_morans_i(values, coords, k=6):
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]
    z = values - values.mean()
    var = np.var(values)
    if var < 1e-15:
        return np.nan
    lag = np.mean(z[indices], axis=1)
    return np.mean(z * lag) / var


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("flashdeconv", flashdeconv.__version__, flashdeconv.__file__)

    adata_ref = sc.read_h5ad(str(REF_PATH))
    X_ref, cell_type_names, ref_genes = load_reference(adata_ref, cell_type_key="Cell_type")
    del adata_ref

    adata_st = load_visium_hd_crc_8um()
    Y_st, coords_st, genes_st = load_spatial_data(adata_st)
    del adata_st
    Y_al, X_al, common_genes = align_genes(Y_st, X_ref, genes_st, ref_genes)
    print(f"  Common genes: {len(common_genes)}; Y {Y_al.shape}, X {X_al.shape}")

    rows = []
    for condition, lambda_val in [("no_spatial", 0.0), ("auto", "auto")]:
        print(f"\n--- {condition} (lambda_spatial={lambda_val}) ---")
        t0 = time.time()
        model = FlashDeconv(
            sketch_dim=512,
            lambda_spatial=lambda_val,
            n_hvg=2000,
            n_markers_per_type=50,
            k_neighbors=6,
            # final rerun: max_iter/tol at package defaults (1000, 1e-4); original max_iter=200
            verbose=True,
            random_state=42,
        )
        props = model.fit_transform(Y_al, X_al, coords_st, cell_type_names=cell_type_names)
        elapsed = time.time() - t0
        lambda_used = model.lambda_used_
        print(f"  lambda_used={lambda_used:.4f}, time={elapsed:.1f}s")

        props_df = pd.DataFrame(props, columns=cell_type_names)
        purity = (props_df.max(axis=1) > 0.8).mean()
        arr = np.clip(props_df.values, 1e-10, 1.0)
        mix_entropy = entropy(arr.T, base=2).mean() / np.log2(props_df.shape[1])
        for ct in cell_type_names:
            mi = compute_morans_i(props_df[ct].values, coords_st, k=6)
            print(f"    Moran's I ({ct}): {mi:.5f}")
            rows.append(dict(bin_size="8um", condition=condition, lambda_used=lambda_used,
                             cell_type=ct, mean_proportion=props_df[ct].mean(),
                             morans_i=mi, signal_purity=purity, mixing_entropy=mix_entropy,
                             time_s=elapsed, n_spots=len(props_df)))
        np.savez_compressed(
            str(OUTPUT_DIR / f"props_8um_{condition}_final.npz"),
            proportions=props.astype(np.float32), cell_types=cell_type_names, coords=coords_st,
            lambda_used=lambda_used,
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "laplacian_ablation_visiumhd_8um_final.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
