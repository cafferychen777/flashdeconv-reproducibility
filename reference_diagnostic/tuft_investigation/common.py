"""Shared loaders for the Tuft-stem niche investigation."""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path("/Users/apple/Research/FlashDeconv")
DATA = ROOT / "validation/visium_hd_data/Visium_HD_Mouse_Small_Intestine_binned_outputs"
HERE = ROOT / "validation/tuft_investigation"
RES = HERE / "results"
RES.mkdir(exist_ok=True)

B, S = "brush cell", "epithelial fate stem cell"
CT = ["brush cell", "enterocyte", "enterocyte progenitor", "enteroendocrine cell",
      "epithelial fate stem cell", "goblet cell", "immature enterocyte",
      "immature goblet cell", "paneth cell", "transit amplifying cell"]


def load_raw(tag):
    """Load official 10x bins (tag '008um' or '016um') in h5 order, in-tissue only."""
    d = DATA / f"square_{tag}"
    ad = sc.read_10x_h5(d / "filtered_feature_bc_matrix.h5")
    ad.var_names_make_unique()
    pos = pd.read_parquet(d / "spatial" / "tissue_positions.parquet").set_index("barcode")
    common = ad.obs_names.intersection(pos.index)
    ad = ad[common].copy()
    ad = ad[(pos.loc[ad.obs_names, "in_tissue"] == 1).to_numpy()].copy()
    p = pos.loc[ad.obs_names]
    ad.obsm["spatial"] = p[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)
    ad.obs["array_row"] = p["array_row"].to_numpy()
    ad.obs["array_col"] = p["array_col"].to_numpy()
    ad.X = ad.X.tocsc()
    ad.obs["umi"] = np.asarray(ad.X.sum(1)).ravel()
    return ad


def gene_counts(ad, genes):
    out = {}
    for g in genes:
        if g in ad.var_names:
            j = ad.var_names.get_loc(g)
            out[g] = np.asarray(ad.X[:, j].todense()).ravel()
    return pd.DataFrame(out, index=ad.obs_names)


def load_original_props(size):
    """Original (Jun 2026, random_state=42) proportions from resolution_2um_analysis.

    Rows are in the same order as load_raw('0{size}um'); we attach barcodes and check coords.
    """
    f = HERE / f"original/resolution_2um_analysis/proportions_{size}um.csv.gz"
    return pd.read_csv(f)


def load_seed_props(seed, size):
    f = HERE / f"rerun_props/correct_seed{seed}/proportions_{size}um.csv.gz"
    return pd.read_csv(f).set_index("bin_id")
