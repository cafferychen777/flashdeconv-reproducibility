"""
Xenium-Based Orthogonal Validation of FlashDeconv CRC Deconvolution.

This script validates FlashDeconv's Visium HD deconvolution against an
independent Xenium spatial transcriptomics dataset (P1 CRC serial section).

Validation components:
1. Annotate Xenium cells via correlation with scRNA-seq reference
2. Virtual binning: compare FlashDeconv predictions against Xenium ground truth
3. Global proportion comparison (Xenium vs FlashDeconv vs RCTD)
4. Pathologist annotation concordance
5. Three-way method comparison

Data: Oliveira et al., Nature Genetics 2025
  - Xenium: 422 genes x 307K cells (P1 CRC)
  - scRNA-seq ref: 260K cells, 38 Level2 types (Chromium Flex)
  - Visium HD: 507K bins @8um (P1 CRC)
  - RCTD: Published singlet deconvolution results
"""

import os
import sys
import warnings
from pathlib import Path

# Force unbuffered stdout for progress visibility
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
try:
    import pyarrow.parquet as pq  # only needed for re-annotation (not rerun)
except ImportError:
    pq = None
from scipy import sparse
from scipy.stats import pearsonr, ttest_ind
from scipy.spatial.distance import cosine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import flashdeconv as fd

# v0.2.0 rerun: plotting helpers are not needed (figures are not regenerated)
try:
    from analysis.crc_style import (
        setup_nature_rcparams,
        CRC_LINEAGE_COLORS,
        panel_label,
        add_scalebar,
    )
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(os.environ.get("FLASHDECONV_PROJECT_ROOT", "/Users/apple/Research/FlashDeconv"))
COHORT_DIR = BASE / "data" / "visium_hd_crc_cohort"
XENIUM_DIR = COHORT_DIR / "xenium_data"
# v0.2.0 rerun: write outputs to a separate directory (never overwrite archived results)
OUTPUT_DIR = Path(os.environ["RERUN_OUT_DIR"])
FIGURE_DIR = OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Annotated Xenium cells are reused from the archived analysis
XENIUM_ANNOT = BASE / "analysis" / "xenium_validation_results" / "xenium_P1_CRC_annotated.h5ad"

# Data paths
XENIUM_H5 = XENIUM_DIR / "cell_feature_matrix.h5"
XENIUM_CELLS = XENIUM_DIR / "cells.parquet"
REF_H5 = COHORT_DIR / "scRNA_ref" / "HumanColonCancer_Flex_filtered_feature_bc_matrix.h5"
REF_META = COHORT_DIR / "metadata" / "SingleCell_MetaData.csv.gz"
FD_H5AD = BASE / "analysis" / "crc_cohort_results" / "P1_CRC_deconv.h5ad"
RCTD_CSV = COHORT_DIR / "metadata" / "DeconvolutionResults_P1CRC.csv.gz"
PATHO_CSV = COHORT_DIR / "pathologist_annotations" / "8um_squares_annotation.csv"

# Lineage mapping: 38 Level2 subtypes -> 6 lineages
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
    "Enteric Glial": "Other", "Adipocyte": "Other",
    "Unknown III (SM)": "Other",
}

# Canonical marker genes for annotation quality check
MARKER_GENES = {
    "Tumor": ["EPCAM", "KRT20", "MUC2"],
    "Macrophage": ["CD68", "CD14", "CSF1R"],
    "CD8 T cell": ["CD3D", "CD8A", "CD8B"],
    "CD4 T cell": ["CD3D", "CD4"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN"],
    "Endothelial": ["PECAM1", "VWF"],
    "Plasma": ["MZB1", "JCHAIN"],
}

BIN_SIZES_UM = [8, 16, 32, 64, 128]


# ── Step 1: Load and Annotate Xenium Cells ─────────────────────────────────

def load_and_annotate_xenium(
    xenium_h5: Path = XENIUM_H5,
    cells_parquet: Path = XENIUM_CELLS,
    ref_h5: Path = REF_H5,
    ref_meta: Path = REF_META,
    output_dir: Path = OUTPUT_DIR,
) -> sc.AnnData:
    """
    Load Xenium data, annotate cells via correlation with scRNA-seq reference.

    Returns annotated AnnData with obs["Level2"] and obsm["spatial"].
    """
    print("\n" + "=" * 70)
    print("Step 1: Load and Annotate Xenium Cells")
    print("=" * 70)

    # --- Load Xenium ---
    print("\nLoading Xenium cell_feature_matrix.h5...")
    adata_xen = sc.read_10x_h5(xenium_h5)
    adata_xen.var_names_make_unique()
    print(f"  Xenium: {adata_xen.n_obs:,} cells, {adata_xen.n_vars} genes")

    # Load spatial coordinates
    cells_df = pq.read_table(cells_parquet).to_pandas()
    cells_df = cells_df.set_index("cell_id")
    common_cells = adata_xen.obs_names.intersection(cells_df.index)
    adata_xen = adata_xen[common_cells].copy()

    adata_xen.obsm["spatial"] = cells_df.loc[
        adata_xen.obs_names, ["x_centroid", "y_centroid"]
    ].values.astype(np.float32)
    adata_xen.obs["transcript_counts"] = cells_df.loc[
        adata_xen.obs_names, "transcript_counts"
    ].values
    adata_xen.obs["cell_area"] = cells_df.loc[
        adata_xen.obs_names, "cell_area"
    ].values

    # Filter low-quality cells (<10 transcripts)
    min_counts = 10
    keep = adata_xen.obs["transcript_counts"] >= min_counts
    n_removed = (~keep).sum()
    adata_xen = adata_xen[keep].copy()
    print(f"  Filtered {n_removed:,} cells with <{min_counts} transcripts")
    print(f"  Remaining: {adata_xen.n_obs:,} cells")

    # Store raw counts
    adata_xen.layers["counts"] = adata_xen.X.copy()

    # --- Load scRNA-seq reference ---
    print("\nLoading scRNA-seq reference...")
    adata_ref = sc.read_10x_h5(ref_h5)
    adata_ref.var_names_make_unique()
    meta = pd.read_csv(ref_meta, compression="gzip").set_index("Barcode")
    common_bc = adata_ref.obs_names.intersection(meta.index)
    adata_ref = adata_ref[common_bc].copy()
    keep_ref = meta.loc[adata_ref.obs_names, "QCFilter"] == "Keep"
    adata_ref = adata_ref[keep_ref].copy()
    adata_ref.obs["Level2"] = meta.loc[adata_ref.obs_names, "Level2"].values
    print(f"  Reference: {adata_ref.n_obs:,} cells, {adata_ref.obs['Level2'].nunique()} types")

    # --- Find overlapping genes ---
    overlap_genes = sorted(set(adata_xen.var_names) & set(adata_ref.var_names))
    n_overlap = len(overlap_genes)
    print(f"\n  Overlapping genes: {n_overlap}")

    adata_xen_sub = adata_xen[:, overlap_genes].copy()
    adata_ref_sub = adata_ref[:, overlap_genes].copy()

    # --- Normalize both ---
    sc.pp.normalize_total(adata_xen_sub, target_sum=1e4)
    sc.pp.log1p(adata_xen_sub)

    sc.pp.normalize_total(adata_ref_sub, target_sum=1e4)
    sc.pp.log1p(adata_ref_sub)

    # --- Build reference centroids ---
    print("\nBuilding reference centroids (mean expression per Level2 type)...")
    cell_types = sorted(adata_ref_sub.obs["Level2"].unique())
    n_types = len(cell_types)
    centroids = np.zeros((n_types, n_overlap), dtype=np.float32)

    for i, ct in enumerate(cell_types):
        mask = adata_ref_sub.obs["Level2"] == ct
        X_ct = adata_ref_sub.X[mask.values]
        if sparse.issparse(X_ct):
            X_ct = X_ct.toarray()
        centroids[i] = X_ct.mean(axis=0)

    print(f"  Centroid matrix: ({n_types}, {n_overlap})")

    # --- Correlate each Xenium cell against centroids ---
    print("\nAnnotating cells via correlation...")
    X_xen = adata_xen_sub.X
    if sparse.issparse(X_xen):
        X_xen = X_xen.toarray()
    X_xen = X_xen.astype(np.float64)

    # Standardize rows (zero mean, unit std)
    X_mean = X_xen.mean(axis=1, keepdims=True)
    X_std = X_xen.std(axis=1, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_norm = (X_xen - X_mean) / X_std

    C_mean = centroids.mean(axis=1, keepdims=True)
    C_std = centroids.std(axis=1, keepdims=True)
    C_std[C_std == 0] = 1.0
    C_norm = (centroids - C_mean) / C_std

    # Correlation: X_norm @ C_norm.T / n_genes
    corr_matrix = X_norm @ C_norm.T / n_overlap  # (n_cells, n_types)

    # Assign cell types
    best_idx = np.argmax(corr_matrix, axis=1)
    best_corr = corr_matrix[np.arange(len(best_idx)), best_idx]

    labels = np.array([cell_types[i] for i in best_idx])
    # Mark low-confidence as Unassigned
    labels[best_corr < 0.15] = "Unassigned"

    # Store in original (full-gene) AnnData
    adata_xen.obs["Level2"] = labels
    adata_xen.obs["annotation_confidence"] = best_corr
    adata_xen.obs["lineage"] = adata_xen.obs["Level2"].map(LINEAGE_MAP).fillna("Other")

    # --- Report annotation quality ---
    n_assigned = (labels != "Unassigned").sum()
    pct_assigned = n_assigned / len(labels) * 100
    print(f"\n  Assigned: {n_assigned:,} / {len(labels):,} ({pct_assigned:.1f}%)")
    print(f"  Mean confidence (assigned): {best_corr[labels != 'Unassigned'].mean():.3f}")

    print("\n  Per-type annotation summary:")
    print(f"  {'Type':<35s} {'Count':>8s} {'Mean r':>8s}")
    print("  " + "-" * 55)
    for ct in cell_types:
        ct_mask = labels == ct
        n_ct = ct_mask.sum()
        if n_ct > 0:
            mean_r = best_corr[ct_mask].mean()
            flag = " *" if mean_r < 0.3 else ""
            print(f"  {ct:<35s} {n_ct:>8,} {mean_r:>8.3f}{flag}")
    n_unassigned = (labels == "Unassigned").sum()
    print(f"  {'Unassigned':<35s} {n_unassigned:>8,}")

    # --- Marker gene validation ---
    print("\n  Marker gene expression check:")
    var_set = set(adata_xen.var_names)
    X_raw = adata_xen.layers["counts"]
    for ct, markers in MARKER_GENES.items():
        available = [g for g in markers if g in var_set]
        if not available or ct not in cell_types:
            continue
        ct_mask = adata_xen.obs["Level2"] == ct
        other_mask = (adata_xen.obs["Level2"] != ct) & (adata_xen.obs["Level2"] != "Unassigned")
        if ct_mask.sum() == 0 or other_mask.sum() == 0:
            continue
        for gene in available[:2]:
            g_idx = list(adata_xen.var_names).index(gene)
            if sparse.issparse(X_raw):
                expr_ct = np.asarray(X_raw[ct_mask.values, g_idx].todense()).flatten()
                expr_other = np.asarray(X_raw[other_mask.values, g_idx].todense()).flatten()
            else:
                expr_ct = X_raw[ct_mask.values, g_idx].flatten()
                expr_other = X_raw[other_mask.values, g_idx].flatten()
            fc = (expr_ct.mean() + 0.01) / (expr_other.mean() + 0.01)
            print(f"    {ct}: {gene} FC={fc:.1f}x "
                  f"(in-type={expr_ct.mean():.2f}, other={expr_other.mean():.2f})")

    # Save
    out_path = output_dir / "xenium_P1_CRC_annotated.h5ad"
    adata_xen.write(out_path)
    print(f"\n  Saved: {out_path}")

    return adata_xen


# ── Step 2: Virtual Binning Validation ─────────────────────────────────────

def virtual_binning_validation(
    adata_xen: sc.AnnData,
    ref_h5: Path = REF_H5,
    ref_meta: Path = REF_META,
    bin_sizes: list = None,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Create virtual bins at multiple resolutions, compute ground truth from
    Xenium cell assignments, run FlashDeconv, and compare.
    """
    if bin_sizes is None:
        bin_sizes = BIN_SIZES_UM

    print("\n" + "=" * 70)
    print("Step 2: Virtual Binning Validation")
    print("=" * 70)

    # Load reference
    print("\nLoading scRNA-seq reference for deconvolution...")
    adata_ref = sc.read_10x_h5(ref_h5)
    adata_ref.var_names_make_unique()
    meta = pd.read_csv(ref_meta, compression="gzip").set_index("Barcode")
    common_bc = adata_ref.obs_names.intersection(meta.index)
    adata_ref = adata_ref[common_bc].copy()
    keep_ref = meta.loc[adata_ref.obs_names, "QCFilter"] == "Keep"
    adata_ref = adata_ref[keep_ref].copy()
    adata_ref.obs["Level2"] = meta.loc[adata_ref.obs_names, "Level2"].values

    # Xenium spatial extent
    coords = adata_xen.obsm["spatial"]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    cell_types = sorted(adata_xen.obs["Level2"].unique())
    cell_types = [ct for ct in cell_types if ct != "Unassigned"]
    ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    n_types = len(cell_types)

    # Get raw counts for expression aggregation
    X_raw = adata_xen.layers["counts"] if "counts" in adata_xen.layers else adata_xen.X

    all_records = []

    for bin_size in bin_sizes:
        print(f"\n{'='*50}")
        print(f"  Bin size: {bin_size} um")
        print(f"{'='*50}")

        # --- Create grid and assign cells ---
        x_edges = np.arange(x_min, x_max + bin_size, bin_size)
        y_edges = np.arange(y_min, y_max + bin_size, bin_size)
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1

        x_idx = np.clip(np.digitize(coords[:, 0], x_edges) - 1, 0, n_x - 1)
        y_idx = np.clip(np.digitize(coords[:, 1], y_edges) - 1, 0, n_y - 1)
        bin_ids = x_idx * n_y + y_idx

        # --- Ground truth proportions ---
        cell_labels = adata_xen.obs["Level2"].values
        unique_bins = np.unique(bin_ids)
        n_bins = len(unique_bins)
        bin_remap = {old: new for new, old in enumerate(unique_bins)}
        remapped_bins = np.array([bin_remap[b] for b in bin_ids])

        gt_counts = np.zeros((n_bins, n_types), dtype=np.float32)
        for cell_i in range(len(cell_labels)):
            ct = cell_labels[cell_i]
            if ct in ct_to_idx:
                gt_counts[remapped_bins[cell_i], ct_to_idx[ct]] += 1

        cells_per_bin = gt_counts.sum(axis=1)
        gt_props = gt_counts / np.maximum(cells_per_bin[:, None], 1)

        # --- Aggregate expression ---
        print(f"  Aggregating expression into {n_bins:,} non-empty bins...")
        # Build sparse aggregation matrix: (n_bins, n_cells)
        agg_rows = remapped_bins
        agg_cols = np.arange(len(remapped_bins))
        agg_mat = sparse.csr_matrix(
            (np.ones(len(agg_cols)), (agg_rows, agg_cols)),
            shape=(n_bins, adata_xen.n_obs),
        )
        X_binned = agg_mat @ (X_raw if sparse.issparse(X_raw) else sparse.csr_matrix(X_raw))

        # Bin centers
        bin_centers = np.zeros((n_bins, 2), dtype=np.float32)
        for cell_i in range(len(remapped_bins)):
            bid = remapped_bins[cell_i]
            bin_centers[bid] += coords[cell_i]
        bin_center_counts = np.bincount(remapped_bins, minlength=n_bins).astype(np.float32)
        bin_center_counts[bin_center_counts == 0] = 1
        bin_centers /= bin_center_counts[:, None]

        # Build AnnData for deconvolution
        adata_binned = sc.AnnData(X=X_binned)
        adata_binned.var_names = adata_xen.var_names.copy()
        adata_binned.obs_names = [f"bin_{i}" for i in range(n_bins)]
        adata_binned.obsm["spatial"] = bin_centers

        # --- Run FlashDeconv ---
        print(f"  Running FlashDeconv on {n_bins:,} bins...")
        n_overlap = len(set(adata_binned.var_names) & set(adata_ref.var_names))
        fd.tl.deconvolve(
            adata_binned,
            adata_ref,
            cell_type_key="Level2",
            sketch_dim=256,
            n_hvg=min(400, n_overlap),
            n_markers_per_type=30,
            preprocess="log_cpm",
            random_state=42,
            key_added="flashdeconv",
        )

        pred_props = adata_binned.obsm["flashdeconv"]

        # Align columns: ensure same cell type order
        pred_cols = list(pred_props.columns)
        shared_types = [ct for ct in cell_types if ct in pred_cols]
        pred_aligned = pred_props[shared_types].values
        gt_aligned = gt_props[:, [ct_to_idx[ct] for ct in shared_types]]

        # --- Compute metrics ---
        # Global Pearson r (flattened)
        p_flat = pred_aligned.flatten()
        g_flat = gt_aligned.flatten()
        mask_nz = (p_flat > 0) | (g_flat > 0)
        global_r = pearsonr(p_flat[mask_nz], g_flat[mask_nz])[0] if mask_nz.sum() > 10 else np.nan

        # Per-cell-type Pearson r
        per_type_r = {}
        for j, ct in enumerate(shared_types):
            p_col = pred_aligned[:, j]
            g_col = gt_aligned[:, j]
            if p_col.std() > 0 and g_col.std() > 0:
                per_type_r[ct] = pearsonr(p_col, g_col)[0]

        # Per-lineage Pearson r
        lineages = sorted(set(LINEAGE_MAP.values()) - {"Other"})
        per_lineage_r = {}
        for lin in lineages:
            lin_types = [ct for ct in shared_types if LINEAGE_MAP.get(ct) == lin]
            if not lin_types:
                continue
            lin_pred = pred_aligned[:, [shared_types.index(ct) for ct in lin_types]].sum(axis=1)
            lin_gt = gt_aligned[:, [shared_types.index(ct) for ct in lin_types]].sum(axis=1)
            if lin_pred.std() > 0 and lin_gt.std() > 0:
                per_lineage_r[lin] = pearsonr(lin_pred, lin_gt)[0]

        # RMSE per cell type
        per_type_rmse = {}
        for j, ct in enumerate(shared_types):
            per_type_rmse[ct] = np.sqrt(np.mean((pred_aligned[:, j] - gt_aligned[:, j]) ** 2))

        # Per-bin cosine similarity
        cos_sims = []
        for i in range(n_bins):
            p_vec = pred_aligned[i]
            g_vec = gt_aligned[i]
            if np.linalg.norm(p_vec) > 0 and np.linalg.norm(g_vec) > 0:
                cos_sims.append(1 - cosine(p_vec, g_vec))
        median_cosine = np.median(cos_sims) if cos_sims else np.nan

        # Stratify by cells per bin
        for label, lo, hi in [("1_cell", 0.5, 1.5), ("2-5_cells", 1.5, 5.5), (">5_cells", 5.5, 1e6)]:
            mask_strat = (cells_per_bin >= lo) & (cells_per_bin < hi)
            n_strat = mask_strat.sum()
            if n_strat < 20:
                continue
            p_s = pred_aligned[mask_strat].flatten()
            g_s = gt_aligned[mask_strat].flatten()
            nz_s = (p_s > 0) | (g_s > 0)
            r_strat = pearsonr(p_s[nz_s], g_s[nz_s])[0] if nz_s.sum() > 10 else np.nan
            all_records.append({
                "bin_size_um": bin_size,
                "metric": f"pearson_r_{label}",
                "value": r_strat,
                "n_bins": int(n_strat),
            })

        # Store main metrics
        all_records.append({"bin_size_um": bin_size, "metric": "global_pearson_r", "value": global_r, "n_bins": n_bins})
        all_records.append({"bin_size_um": bin_size, "metric": "median_cosine_sim", "value": median_cosine, "n_bins": n_bins})
        all_records.append({"bin_size_um": bin_size, "metric": "mean_per_type_r", "value": np.mean(list(per_type_r.values())), "n_bins": n_bins})
        all_records.append({"bin_size_um": bin_size, "metric": "mean_per_lineage_r", "value": np.mean(list(per_lineage_r.values())), "n_bins": n_bins})
        all_records.append({"bin_size_um": bin_size, "metric": "median_cells_per_bin", "value": float(np.median(cells_per_bin)), "n_bins": n_bins})

        for ct, r_val in per_type_r.items():
            all_records.append({"bin_size_um": bin_size, "metric": f"per_type_r__{ct}", "value": r_val, "n_bins": n_bins})
        for ct, rmse_val in per_type_rmse.items():
            all_records.append({"bin_size_um": bin_size, "metric": f"rmse__{ct}", "value": rmse_val, "n_bins": n_bins})
        for lin, r_val in per_lineage_r.items():
            all_records.append({"bin_size_um": bin_size, "metric": f"per_lineage_r__{lin}", "value": r_val, "n_bins": n_bins})

        print(f"\n  Results at {bin_size} um:")
        print(f"    Global Pearson r:      {global_r:.4f}")
        print(f"    Mean per-type r:       {np.mean(list(per_type_r.values())):.4f}")
        print(f"    Mean per-lineage r:    {np.mean(list(per_lineage_r.values())):.4f}")
        print(f"    Median cosine sim:     {median_cosine:.4f}")
        print(f"    Median cells/bin:      {np.median(cells_per_bin):.1f}")

    metrics_df = pd.DataFrame(all_records)
    out_path = output_dir / "virtual_binning_metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return metrics_df


# ── Step 3: Global Proportion Comparison ───────────────────────────────────

def global_proportion_comparison(
    adata_xen: sc.AnnData,
    fd_h5ad: Path = FD_H5AD,
    rctd_csv: Path = RCTD_CSV,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Compare global cell type proportions: Xenium vs FlashDeconv vs RCTD.
    """
    print("\n" + "=" * 70)
    print("Step 3: Global Proportion Comparison")
    print("=" * 70)

    cell_types = sorted(set(LINEAGE_MAP.keys()))

    # --- Xenium proportions ---
    xen_counts = adata_xen.obs["Level2"].value_counts()
    xen_total = xen_counts.drop("Unassigned", errors="ignore").sum()
    xen_props = {}
    for ct in cell_types:
        xen_props[ct] = xen_counts.get(ct, 0) / xen_total

    # --- FlashDeconv proportions ---
    print("\nLoading FlashDeconv results...")
    adata_fd = sc.read_h5ad(fd_h5ad)
    fd_mean = adata_fd.obsm["flashdeconv"].mean(axis=0)
    fd_props = {ct: fd_mean.get(ct, 0.0) for ct in cell_types}

    # --- RCTD proportions ---
    print("Loading RCTD results...")
    rctd = pd.read_csv(rctd_csv, compression="gzip", low_memory=False)
    rctd_singlet = rctd[rctd["DeconvolutionClass"] == "singlet"]
    rctd_counts = rctd_singlet["DeconvolutionLabel2"].value_counts()
    rctd_total = rctd_counts.sum()
    rctd_props = {}
    for ct in cell_types:
        rctd_props[ct] = rctd_counts.get(ct, 0) / rctd_total

    # Build comparison table
    records = []
    for ct in cell_types:
        lin = LINEAGE_MAP.get(ct, "Other")
        records.append({
            "cell_type": ct,
            "lineage": lin,
            "xenium_prop": xen_props.get(ct, 0),
            "flashdeconv_prop": fd_props.get(ct, 0),
            "rctd_prop": rctd_props.get(ct, 0),
        })

    df = pd.DataFrame(records)

    # Add lineage-level proportions
    lin_records = []
    for lin in sorted(set(LINEAGE_MAP.values())):
        lin_mask = df["lineage"] == lin
        lin_records.append({
            "cell_type": f"[LINEAGE] {lin}",
            "lineage": lin,
            "xenium_prop": df.loc[lin_mask, "xenium_prop"].sum(),
            "flashdeconv_prop": df.loc[lin_mask, "flashdeconv_prop"].sum(),
            "rctd_prop": df.loc[lin_mask, "rctd_prop"].sum(),
        })
    df = pd.concat([df, pd.DataFrame(lin_records)], ignore_index=True)

    # Pairwise correlations at cell-type level
    ct_df = df[~df["cell_type"].str.startswith("[LINEAGE]")]
    xen_vec = ct_df["xenium_prop"].values
    fd_vec = ct_df["flashdeconv_prop"].values
    rctd_vec = ct_df["rctd_prop"].values

    r_fd_xen = pearsonr(fd_vec, xen_vec)[0] if len(xen_vec) > 2 else np.nan
    r_rctd_xen = pearsonr(rctd_vec, xen_vec)[0] if len(xen_vec) > 2 else np.nan
    r_fd_rctd = pearsonr(fd_vec, rctd_vec)[0] if len(xen_vec) > 2 else np.nan

    print(f"\n  Cell-type level correlations (n={len(ct_df)} types):")
    print(f"    FlashDeconv vs Xenium:  r = {r_fd_xen:.4f}")
    print(f"    RCTD vs Xenium:         r = {r_rctd_xen:.4f}")
    print(f"    FlashDeconv vs RCTD:    r = {r_fd_rctd:.4f}")

    # Lineage-level correlations
    lin_df = df[df["cell_type"].str.startswith("[LINEAGE]")]
    xen_lin = lin_df["xenium_prop"].values
    fd_lin = lin_df["flashdeconv_prop"].values
    rctd_lin = lin_df["rctd_prop"].values

    r_fd_xen_lin = pearsonr(fd_lin, xen_lin)[0] if len(xen_lin) > 2 else np.nan
    r_rctd_xen_lin = pearsonr(rctd_lin, xen_lin)[0] if len(xen_lin) > 2 else np.nan

    print(f"\n  Lineage level correlations (n={len(lin_df)} lineages):")
    print(f"    FlashDeconv vs Xenium:  r = {r_fd_xen_lin:.4f}")
    print(f"    RCTD vs Xenium:         r = {r_rctd_xen_lin:.4f}")

    out_path = output_dir / "global_proportion_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return df


# ── Step 4: Pathologist Annotation Concordance ─────────────────────────────

def pathologist_concordance(
    fd_h5ad: Path = FD_H5AD,
    patho_csv: Path = PATHO_CSV,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Compare FlashDeconv lineage proportions with pathologist tissue annotations.
    """
    print("\n" + "=" * 70)
    print("Step 4: Pathologist Annotation Concordance")
    print("=" * 70)

    # Load FlashDeconv results
    print("\nLoading FlashDeconv results...")
    adata_fd = sc.read_h5ad(fd_h5ad)
    fd_props = adata_fd.obsm["flashdeconv"]

    # Load pathologist annotations (tab-separated, no header)
    print("Loading pathologist annotations...")
    patho = pd.read_csv(patho_csv, sep="\t", header=None, names=["barcode", "category"])
    patho = patho.set_index("barcode")
    print(f"  Loaded {len(patho):,} annotations")
    print(f"  Categories: {patho['category'].value_counts().to_dict()}")

    # Match barcodes
    common = fd_props.index.intersection(patho.index)
    print(f"  Matched barcodes: {len(common):,}")

    fd_matched = fd_props.loc[common]
    patho_matched = patho.loc[common, "category"]

    # Compute lineage proportions per bin
    lineages = sorted(set(LINEAGE_MAP.values()) - {"Other"})
    lineage_props = pd.DataFrame(index=common, columns=lineages, dtype=float)
    for lin in lineages:
        lin_types = [ct for ct in fd_matched.columns if LINEAGE_MAP.get(ct) == lin]
        if lin_types:
            lineage_props[lin] = fd_matched[lin_types].sum(axis=1).values

    # Global mean for statistical testing
    global_means = lineage_props.mean(axis=0)

    # Build concordance matrix
    categories = sorted(patho_matched.unique())
    categories = [c for c in categories if c != "Outside"]

    records = []
    for cat in categories:
        cat_mask = patho_matched == cat
        n_bins = cat_mask.sum()
        for lin in lineages:
            vals = lineage_props.loc[cat_mask.values, lin].values
            mean_prop = vals.mean()
            std_prop = vals.std()

            # One-sided t-test vs global mean
            t_stat, p_val = ttest_ind(
                vals,
                lineage_props[lin].values,
                equal_var=False,
                alternative="greater",
            )

            records.append({
                "category": cat,
                "lineage": lin,
                "mean_proportion": mean_prop,
                "std_proportion": std_prop,
                "global_mean": global_means[lin],
                "enrichment": mean_prop / (global_means[lin] + 1e-8),
                "t_statistic": t_stat,
                "p_value": p_val,
                "n_bins": int(n_bins),
            })

    conc_df = pd.DataFrame(records)

    # Print concordance matrix
    print("\n  Concordance matrix (mean lineage proportion per category):")
    pivot = conc_df.pivot(index="category", columns="lineage", values="mean_proportion")
    print(pivot.round(3).to_string())

    print("\n  Expected dominant lineages:")
    expected = {
        "Neoplasm": "Tumor",
        "Connective Tissue": "Stromal",
        "Non-neoplastic Epithelium": "Epithelial",
        "Smooth Muscle": "Stromal",
        "Vessel": "Endothelial",
        "Veins": "Endothelial",
    }
    for cat, expected_lin in expected.items():
        if cat in pivot.index:
            dominant = pivot.loc[cat].idxmax()
            match = "PASS" if dominant == expected_lin else "FAIL"
            print(f"    {cat}: dominant={dominant} (expected={expected_lin}) [{match}]")

    out_path = output_dir / "pathologist_concordance.csv"
    conc_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return conc_df


# ── Step 5: Three-way Comparison ───────────────────────────────────────────

def three_way_comparison(
    global_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Compare FlashDeconv vs RCTD accuracy using Xenium as ground truth.
    """
    print("\n" + "=" * 70)
    print("Step 5: Three-Way Comparison (FlashDeconv vs RCTD vs Xenium)")
    print("=" * 70)

    records = []

    # Cell-type level
    ct_df = global_df[~global_df["cell_type"].str.startswith("[LINEAGE]")].copy()
    for _, row in ct_df.iterrows():
        ct = row["cell_type"]
        xen = row["xenium_prop"]
        fd_val = row["flashdeconv_prop"]
        rctd_val = row["rctd_prop"]

        fd_err = abs(fd_val - xen)
        rctd_err = abs(rctd_val - xen)
        winner = "FlashDeconv" if fd_err < rctd_err else ("RCTD" if rctd_err < fd_err else "Tie")

        records.append({
            "level": "cell_type",
            "name": ct,
            "lineage": LINEAGE_MAP.get(ct, "Other"),
            "xenium_prop": xen,
            "flashdeconv_prop": fd_val,
            "rctd_prop": rctd_val,
            "fd_abs_error": fd_err,
            "rctd_abs_error": rctd_err,
            "winner": winner,
        })

    # Lineage level
    lin_df = global_df[global_df["cell_type"].str.startswith("[LINEAGE]")].copy()
    for _, row in lin_df.iterrows():
        lin = row["lineage"]
        xen = row["xenium_prop"]
        fd_val = row["flashdeconv_prop"]
        rctd_val = row["rctd_prop"]

        fd_err = abs(fd_val - xen)
        rctd_err = abs(rctd_val - xen)
        winner = "FlashDeconv" if fd_err < rctd_err else ("RCTD" if rctd_err < fd_err else "Tie")

        records.append({
            "level": "lineage",
            "name": lin,
            "lineage": lin,
            "xenium_prop": xen,
            "flashdeconv_prop": fd_val,
            "rctd_prop": rctd_val,
            "fd_abs_error": fd_err,
            "rctd_abs_error": rctd_err,
            "winner": winner,
        })

    comp_df = pd.DataFrame(records)

    # Summary
    ct_comp = comp_df[comp_df["level"] == "cell_type"]
    lin_comp = comp_df[comp_df["level"] == "lineage"]

    print("\n  Cell-type level (n=38):")
    ct_winners = ct_comp["winner"].value_counts()
    for w, n in ct_winners.items():
        print(f"    {w}: {n}")
    fd_mae = ct_comp["fd_abs_error"].mean()
    rctd_mae = ct_comp["rctd_abs_error"].mean()
    print(f"    FlashDeconv MAE: {fd_mae:.4f}")
    print(f"    RCTD MAE:        {rctd_mae:.4f}")

    print(f"\n  Lineage level (n={len(lin_comp)}):")
    lin_winners = lin_comp["winner"].value_counts()
    for w, n in lin_winners.items():
        print(f"    {w}: {n}")
    fd_mae_lin = lin_comp["fd_abs_error"].mean()
    rctd_mae_lin = lin_comp["rctd_abs_error"].mean()
    print(f"    FlashDeconv MAE: {fd_mae_lin:.4f}")
    print(f"    RCTD MAE:        {rctd_mae_lin:.4f}")

    out_path = output_dir / "three_way_comparison.csv"
    comp_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return comp_df


# ── Figures ────────────────────────────────────────────────────────────────

def plot_main_figure(
    adata_xen: sc.AnnData,
    binning_df: pd.DataFrame,
    concordance_df: pd.DataFrame,
    global_df: pd.DataFrame,
    figure_dir: Path = FIGURE_DIR,
):
    """
    Main validation figure: 2x3 panels, 7.2" x 5.0".
    """
    print("\n" + "=" * 70)
    print("Generating Main Figure")
    print("=" * 70)

    setup_nature_rcparams()

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    ((ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f)) = axes

    # ── Panel a: Xenium spatial map colored by lineage ──
    print("  Panel a: Xenium spatial map...")
    coords = adata_xen.obsm["spatial"]
    lineages_arr = adata_xen.obs["lineage"].values
    unique_lineages = sorted(set(lineages_arr) - {"Other", "Unassigned"})

    # Downsample to 50K points
    rng = np.random.default_rng(42)
    n_plot = min(50000, len(coords))
    idx = rng.choice(len(coords), n_plot, replace=False)

    for lin in unique_lineages:
        color = CRC_LINEAGE_COLORS.get(lin, "#999999")
        mask = lineages_arr[idx] == lin
        if mask.sum() > 0:
            ax_a.scatter(
                coords[idx[mask], 0], coords[idx[mask], 1],
                c=color, s=0.1, alpha=0.5, rasterized=True, label=lin,
            )
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("x (\u00b5m)")
    ax_a.set_ylabel("y (\u00b5m)")
    ax_a.legend(markerscale=10, loc="lower left", fontsize=4.5, handletextpad=0.3,
                frameon=False)
    add_scalebar(ax_a, length_um=1000, px_per_um=1.0)
    panel_label(ax_a, "a")

    # ── Panel b: Virtual binning scatter at 32um ──
    print("  Panel b: Virtual binning scatter at 32um...")
    # We'll re-run a quick 32um comparison for the scatter
    # For now, use the metrics to annotate
    r_32 = binning_df.loc[
        (binning_df["bin_size_um"] == 32) & (binning_df["metric"] == "global_pearson_r"),
        "value",
    ]
    r_32_val = r_32.values[0] if len(r_32) > 0 else np.nan

    # Placeholder scatter showing global r annotation
    # Use lineage-level per-type r values
    type_r_32 = binning_df[
        (binning_df["bin_size_um"] == 32) & (binning_df["metric"].str.startswith("per_type_r__"))
    ].copy()
    type_r_32["cell_type"] = type_r_32["metric"].str.replace("per_type_r__", "", regex=False)
    type_r_32["lineage"] = type_r_32["cell_type"].map(LINEAGE_MAP).fillna("Other")

    if len(type_r_32) > 0:
        for lin in unique_lineages:
            color = CRC_LINEAGE_COLORS.get(lin, "#999999")
            mask = type_r_32["lineage"] == lin
            sub = type_r_32[mask]
            if len(sub) > 0:
                # Jitter x for visibility
                x_vals = rng.normal(0, 0.02, size=len(sub))
                ax_b.scatter(x_vals, sub["value"].values, c=color, s=8, alpha=0.7, label=lin)

        ax_b.axhline(y=r_32_val, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_b.set_ylabel("Pearson r (predicted vs ground truth)")
        ax_b.set_xlim(-0.15, 0.15)
        ax_b.set_xticks([])
        ax_b.set_xlabel("Cell types (32 \u00b5m)")
        ax_b.legend(markerscale=1.5, loc="lower right", fontsize=4.5, handletextpad=0.3)
    panel_label(ax_b, "b")

    # ── Panel c: Multi-resolution accuracy ──
    print("  Panel c: Multi-resolution accuracy...")
    for metric_name, style, label in [
        ("global_pearson_r", "-", "Global"),
        ("mean_per_lineage_r", "--", "Lineage mean"),
        ("mean_per_type_r", ":", "Type mean"),
    ]:
        sub = binning_df[binning_df["metric"] == metric_name].sort_values("bin_size_um")
        if len(sub) > 0:
            ax_c.plot(sub["bin_size_um"], sub["value"], style, marker="o", markersize=3, label=label)

    ax_c.set_xscale("log")
    ax_c.set_xticks(BIN_SIZES_UM)
    ax_c.set_xticklabels([str(b) for b in BIN_SIZES_UM])
    ax_c.set_xlabel("Bin size (\u00b5m)")
    ax_c.set_ylabel("Pearson r")
    ax_c.set_ylim(0, 1)
    ax_c.legend(fontsize=4.5, handletextpad=0.3)
    panel_label(ax_c, "c")

    # ── Panel d: Pathologist concordance heatmap (enrichment) ──
    print("  Panel d: Pathologist concordance heatmap...")
    conc_df = concordance_df.copy()
    # Use log2 enrichment for better visualization of patterns
    conc_pivot = conc_df.pivot(index="category", columns="lineage", values="enrichment")
    cat_order = ["Neoplasm", "Connective Tissue", "Non-neoplastic Epithelium",
                 "Smooth Muscle", "Vessel", "Veins"]
    lin_order = ["Tumor", "Stromal", "Immune", "Epithelial", "Endothelial"]
    cat_order = [c for c in cat_order if c in conc_pivot.index]
    lin_order = [l for l in lin_order if l in conc_pivot.columns]
    conc_pivot = conc_pivot.loc[cat_order, lin_order]

    log2_enrich = np.log2(conc_pivot.values.clip(min=0.01))
    vmax = np.abs(log2_enrich).max()
    im = ax_d.imshow(log2_enrich, cmap="RdBu_r", aspect="auto",
                     vmin=-vmax, vmax=vmax)
    ax_d.set_xticks(range(len(lin_order)))
    ax_d.set_xticklabels(lin_order, rotation=45, ha="right", fontsize=5)
    ax_d.set_yticks(range(len(cat_order)))
    cat_labels_short = [c.replace("Non-neoplastic ", "Normal\n") for c in cat_order]
    ax_d.set_yticklabels(cat_labels_short, fontsize=5)
    for i in range(len(cat_order)):
        for j in range(len(lin_order)):
            val = conc_pivot.values[i, j]
            color = "white" if abs(log2_enrich[i, j]) > vmax * 0.6 else "black"
            ax_d.text(j, i, f"{val:.1f}x", ha="center", va="center", fontsize=4, color=color)
    plt.colorbar(im, ax=ax_d, shrink=0.7, label="log$_2$ enrichment")
    panel_label(ax_d, "d")

    # ── Panel e: Three-way bar chart (lineage level) ──
    print("  Panel e: Three-way bar chart...")
    lin_global = global_df[global_df["cell_type"].str.startswith("[LINEAGE]")].copy()
    lin_global["lin_name"] = lin_global["lineage"]
    lin_order_e = ["Tumor", "Stromal", "Immune", "Epithelial", "Endothelial"]
    lin_global = lin_global.set_index("lin_name").reindex(lin_order_e).reset_index()

    x = np.arange(len(lin_order_e))
    width = 0.25
    ax_e.bar(x - width, lin_global["xenium_prop"], width, label="Xenium", color="#56B4E9")
    ax_e.bar(x, lin_global["flashdeconv_prop"], width, label="FlashDeconv", color="#D55E00")
    ax_e.bar(x + width, lin_global["rctd_prop"], width, label="RCTD", color="#009E73")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(lin_order_e, rotation=45, ha="right", fontsize=5)
    ax_e.set_ylabel("Proportion")
    ax_e.legend(fontsize=4.5, handletextpad=0.3)
    panel_label(ax_e, "e")

    # ── Panel f: Per-cell-type Pearson r bar chart (32um) ──
    print("  Panel f: Per-type Pearson r bar chart...")
    if len(type_r_32) > 0:
        type_r_sorted = type_r_32.sort_values("value", ascending=True)
        colors = [CRC_LINEAGE_COLORS.get(lin, "#999999") for lin in type_r_sorted["lineage"]]
        ax_f.barh(range(len(type_r_sorted)), type_r_sorted["value"], color=colors, height=0.7)
        ax_f.set_yticks(range(len(type_r_sorted)))
        ax_f.set_yticklabels(type_r_sorted["cell_type"], fontsize=3.5)
        ax_f.set_xlabel("Pearson r (32 \u00b5m)")
        ax_f.set_xlim(0, 1)
    panel_label(ax_f, "f")

    plt.tight_layout()

    for ext in [".pdf", ".png"]:
        out_path = figure_dir / f"fig_xenium_validation{ext}"
        fig.savefig(out_path, dpi=300)
        print(f"  Saved: {out_path}")
    plt.close()


def plot_supplementary_annotation(
    adata_xen: sc.AnnData,
    figure_dir: Path = FIGURE_DIR,
):
    """Supplementary figure: annotation quality."""
    setup_nature_rcparams()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax_a, ax_b, ax_c = axes

    # Panel a: Confidence score distribution
    conf = adata_xen.obs["annotation_confidence"].values
    assigned = adata_xen.obs["Level2"] != "Unassigned"
    ax_a.hist(conf[assigned], bins=50, color="#0072B2", alpha=0.8, density=True)
    ax_a.axvline(x=0.15, color="red", linestyle="--", linewidth=0.5)
    ax_a.axvline(x=0.3, color="orange", linestyle="--", linewidth=0.5)
    ax_a.set_xlabel("Annotation confidence (Pearson r)")
    ax_a.set_ylabel("Density")
    panel_label(ax_a, "a")

    # Panel b: Cell type counts
    ct_counts = adata_xen.obs["Level2"].value_counts()
    ct_counts = ct_counts.drop("Unassigned", errors="ignore").sort_values()
    colors = [CRC_LINEAGE_COLORS.get(LINEAGE_MAP.get(ct, "Other"), "#999999") for ct in ct_counts.index]
    ax_b.barh(range(len(ct_counts)), ct_counts.values, color=colors, height=0.7)
    ax_b.set_yticks(range(len(ct_counts)))
    ax_b.set_yticklabels(ct_counts.index, fontsize=3.5)
    ax_b.set_xlabel("Number of cells")
    panel_label(ax_b, "b")

    # Panel c: Marker gene dotplot (simplified)
    # Show mean expression of key markers per assigned type
    var_set = set(adata_xen.var_names)
    X_raw = adata_xen.layers["counts"] if "counts" in adata_xen.layers else adata_xen.X
    plot_types = ["Tumor I", "Macrophage", "CD8 T cell", "CD4 T cell", "Fibroblast", "Endothelial", "Plasma"]
    plot_genes = []
    for ct in plot_types:
        for g in MARKER_GENES.get(ct, []):
            if g in var_set and g not in plot_genes:
                plot_genes.append(g)

    if plot_genes:
        dot_data = np.zeros((len(plot_types), len(plot_genes)))
        pct_data = np.zeros_like(dot_data)
        for i, ct in enumerate(plot_types):
            ct_mask = adata_xen.obs["Level2"] == ct
            if ct_mask.sum() == 0:
                continue
            for j, gene in enumerate(plot_genes):
                g_idx = list(adata_xen.var_names).index(gene)
                if sparse.issparse(X_raw):
                    expr = np.asarray(X_raw[ct_mask.values, g_idx].todense()).flatten()
                else:
                    expr = X_raw[ct_mask.values, g_idx].flatten()
                dot_data[i, j] = expr.mean()
                pct_data[i, j] = (expr > 0).mean() * 100

        # Normalize for color
        max_val = dot_data.max()
        if max_val > 0:
            dot_colors = dot_data / max_val
        else:
            dot_colors = dot_data

        for i in range(len(plot_types)):
            for j in range(len(plot_genes)):
                size = pct_data[i, j] / 100 * 80 + 2
                ax_c.scatter(j, i, s=size, c=plt.cm.Reds(dot_colors[i, j]),
                             edgecolors="black", linewidths=0.2)

        ax_c.set_xticks(range(len(plot_genes)))
        ax_c.set_xticklabels(plot_genes, rotation=90, fontsize=4.5)
        ax_c.set_yticks(range(len(plot_types)))
        ax_c.set_yticklabels(plot_types, fontsize=5)
        ax_c.set_xlim(-0.5, len(plot_genes) - 0.5)
        ax_c.set_ylim(-0.5, len(plot_types) - 0.5)
    panel_label(ax_c, "c")

    plt.tight_layout()

    for ext in [".pdf", ".png"]:
        out_path = figure_dir / f"sfig_annotation_quality{ext}"
        fig.savefig(out_path, dpi=300)
        print(f"  Saved: {out_path}")
    plt.close()


def plot_supplementary_multires(
    binning_df: pd.DataFrame,
    figure_dir: Path = FIGURE_DIR,
):
    """Supplementary figure: multi-resolution detail."""
    setup_nature_rcparams()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    ax_a, ax_b = axes

    # Panel a: Heatmap of per-type r x resolution
    type_metrics = binning_df[binning_df["metric"].str.startswith("per_type_r__")].copy()
    type_metrics["cell_type"] = type_metrics["metric"].str.replace("per_type_r__", "", regex=False)

    if len(type_metrics) > 0:
        pivot = type_metrics.pivot(index="cell_type", columns="bin_size_um", values="value")
        # Sort by mean r across resolutions
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        im = ax_a.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax_a.set_xticks(range(pivot.shape[1]))
        ax_a.set_xticklabels([f"{c}\u00b5m" for c in pivot.columns], fontsize=5)
        ax_a.set_yticks(range(pivot.shape[0]))
        ax_a.set_yticklabels(pivot.index, fontsize=3.5)
        ax_a.set_xlabel("Bin size")
        plt.colorbar(im, ax=ax_a, shrink=0.7, label="Pearson r")
    panel_label(ax_a, "d")

    # Panel e: Cells per bin distribution
    cpb_metrics = binning_df[binning_df["metric"] == "median_cells_per_bin"].sort_values("bin_size_um")
    if len(cpb_metrics) > 0:
        ax_b.bar(range(len(cpb_metrics)), cpb_metrics["value"], color="#0072B2")
        ax_b.set_xticks(range(len(cpb_metrics)))
        ax_b.set_xticklabels([f"{int(b)}\u00b5m" for b in cpb_metrics["bin_size_um"]])
        ax_b.set_xlabel("Bin size")
        ax_b.set_ylabel("Median cells per bin")
    panel_label(ax_b, "e")

    plt.tight_layout()

    for ext in [".pdf", ".png"]:
        out_path = figure_dir / f"sfig_multires_detail{ext}"
        fig.savefig(out_path, dpi=300)
        print(f"  Saved: {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Xenium-Based Orthogonal Validation of FlashDeconv CRC Deconvolution")
    print("=" * 70)

    # Step 1: Load and annotate Xenium cells
    cached_h5ad = OUTPUT_DIR / "xenium_P1_CRC_annotated.h5ad"
    if cached_h5ad.exists():
        print(f"\nLoading cached annotated Xenium from {cached_h5ad}...")
        adata_xen = sc.read_h5ad(cached_h5ad)
        print(f"  {adata_xen.n_obs:,} cells, {adata_xen.obs['Level2'].nunique()} types")
    else:
        adata_xen = load_and_annotate_xenium()

    # Ensure lineage column exists (may be missing from older cached files)
    if "lineage" not in adata_xen.obs.columns:
        adata_xen.obs["lineage"] = (
            adata_xen.obs["Level2"].map(LINEAGE_MAP).fillna("Other")
        )

    # Step 2: Virtual binning validation
    cached_binning = OUTPUT_DIR / "virtual_binning_metrics.csv"
    if cached_binning.exists():
        print(f"\nLoading cached virtual binning metrics from {cached_binning}...")
        binning_df = pd.read_csv(cached_binning)
    else:
        binning_df = virtual_binning_validation(adata_xen)

    # Step 3: Global proportion comparison
    cached_global = OUTPUT_DIR / "global_proportion_comparison.csv"
    if cached_global.exists():
        print(f"\nLoading cached global comparison from {cached_global}...")
        global_df = pd.read_csv(cached_global)
    else:
        global_df = global_proportion_comparison(adata_xen)

    # Step 4: Pathologist concordance
    cached_conc = OUTPUT_DIR / "pathologist_concordance.csv"
    if cached_conc.exists():
        print(f"\nLoading cached pathologist concordance from {cached_conc}...")
        concordance_df = pd.read_csv(cached_conc)
    else:
        concordance_df = pathologist_concordance()

    # Step 5: Three-way comparison
    cached_3way = OUTPUT_DIR / "three_way_comparison.csv"
    if cached_3way.exists():
        print(f"\nLoading cached three-way comparison from {cached_3way}...")
        three_way_df = pd.read_csv(cached_3way)
    else:
        three_way_df = three_way_comparison(global_df)

    # Figures
    print("\n" + "=" * 70)
    print("Generating Figures")
    print("=" * 70)

    plot_main_figure(adata_xen, binning_df, concordance_df, global_df)
    plot_supplementary_annotation(adata_xen)
    plot_supplementary_multires(binning_df)

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURE_DIR}")
    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB)")
    print("\nFigures:")
    for f in sorted(FIGURE_DIR.glob("*xenium*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
