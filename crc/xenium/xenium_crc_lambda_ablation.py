#!/usr/bin/env python
"""
CRC Xenium pseudo-bulk lambda ablation: lambda=auto vs lambda=0.

Addresses Reviewer 2's concern about Laplacian smoothing hurting rare cell types.
This script runs FlashDeconv on existing CRC Xenium pseudo-bulk bins (8um, 16um)
with lambda=0 (no spatial regularization) and compares against default auto-lambda.

Key question: Does removing spatial regularization improve rare cell type detection
in the multi-cell pseudo-bulk setting (where FlashDeconv is designed to operate)?

Dataset: Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE (388,175 cells, 480 genes)
Reference: Lee et al. (GSE132465) CRC scRNA-seq (63,689 cells, 36 subtypes)
"""

import sys as _sys  # final rerun: fit-logging hook (validation/rerun_final/fdfinal.py)
_sys.path.insert(0, "/scratch/user/cafferychen777/fd_final/code")
import fdfinal  # noqa: E402,F401
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve, auc

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/apple/Research/FlashDeconv")
import flashdeconv as fd

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("XENIUM_IO_DATA_DIR", "/Users/apple/Research/FlashDeconv/analysis/xenium_crc_io_data"))
REF_PATH = Path(os.environ.get("GSE132465_REF", "/Users/apple/Research/FlashDeconv/data/crc_reference/GSE132465_CRC_reference.h5ad"))
OUT_DIR = Path(os.environ["RERUN_OUT_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

H5_PATH = DATA_DIR / "Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_cell_feature_matrix.h5"
CELLS_PATH = DATA_DIR / "Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE_cells.csv.gz"

BIN_SIZES_UM = [8, 16]
LAMBDA_VALUES = [("auto", "auto"), ("no_spatial", 0.0)]
MIN_CORR_THRESHOLD = 0.1
PRESENCE_THRESHOLD = 0.01
RANDOM_SEED = 42


# ── Step 1: Load data ─────────────────────────────────────────────────────

def load_xenium():
    """Load Xenium expression and cell coordinates."""
    print("\n" + "=" * 70)
    print("Step 1: Loading Xenium data")
    print("=" * 70)

    cells = pd.read_csv(CELLS_PATH)
    coords = cells[["cell_id", "x_centroid", "y_centroid"]].copy()
    coords = coords.rename(columns={"x_centroid": "x", "y_centroid": "y"})

    adata = sc.read_10x_h5(str(H5_PATH))
    adata.var_names_make_unique()
    print(f"  Loaded {adata.shape[0]:,} cells x {adata.shape[1]} genes")
    assert adata.shape[0] == len(coords), "Cell count mismatch"

    adata.obs["x"] = coords["x"].values
    adata.obs["y"] = coords["y"].values
    adata.obs["cell_id"] = coords["cell_id"].values
    adata.obsm["spatial"] = coords[["x", "y"]].values.astype(np.float32)
    adata.layers["counts"] = adata.X.copy()

    return adata


def load_reference():
    """Load scRNA-seq CRC reference."""
    print("\nLoading scRNA-seq reference...")
    adata_ref = sc.read_h5ad(str(REF_PATH))
    print(f"  Reference: {adata_ref.n_obs:,} cells, {adata_ref.n_vars} genes")
    print(f"  Cell subtypes: {adata_ref.obs['Cell_subtype'].nunique()}")
    return adata_ref


# ── Step 2: Cell type annotation ──────────────────────────────────────────

def annotate_xenium_cells(adata_xen, adata_ref):
    """Annotate Xenium cells via Pearson correlation with scRNA-seq centroids."""
    print("\n" + "=" * 70)
    print("Step 2: Cell type annotation via correlation")
    print("=" * 70)

    overlap_genes = sorted(set(adata_xen.var_names) & set(adata_ref.var_names))
    n_overlap = len(overlap_genes)
    print(f"  Overlapping genes: {n_overlap}")

    xen_sub = adata_xen[:, overlap_genes].copy()
    ref_sub = adata_ref[:, overlap_genes].copy()

    sc.pp.normalize_total(xen_sub, target_sum=1e4)
    sc.pp.log1p(xen_sub)
    sc.pp.normalize_total(ref_sub, target_sum=1e4)
    sc.pp.log1p(ref_sub)

    cell_types = sorted(ref_sub.obs["Cell_subtype"].unique())
    n_types = len(cell_types)
    print(f"  Building centroids for {n_types} cell subtypes...")

    centroids = np.zeros((n_types, n_overlap), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        mask = ref_sub.obs["Cell_subtype"] == ct
        X_ct = ref_sub.X[mask.values]
        if sparse.issparse(X_ct):
            X_ct = X_ct.toarray()
        centroids[i] = X_ct.mean(axis=0)

    print("  Computing correlations...")
    X_xen = xen_sub.X
    if sparse.issparse(X_xen):
        X_xen = X_xen.toarray()
    X_xen = X_xen.astype(np.float64)

    X_mean = X_xen.mean(axis=1, keepdims=True)
    X_std = X_xen.std(axis=1, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_norm = (X_xen - X_mean) / X_std

    C_mean = centroids.mean(axis=1, keepdims=True)
    C_std = centroids.std(axis=1, keepdims=True)
    C_std[C_std == 0] = 1.0
    C_norm = (centroids - C_mean) / C_std

    corr_matrix = X_norm @ C_norm.T / n_overlap

    best_idx = np.argmax(corr_matrix, axis=1)
    best_corr = corr_matrix[np.arange(len(best_idx)), best_idx]

    labels = np.array([cell_types[i] for i in best_idx])
    labels[best_corr < MIN_CORR_THRESHOLD] = "Unassigned"

    adata_xen.obs["cell_type"] = labels
    adata_xen.obs["annotation_confidence"] = best_corr

    n_assigned = (labels != "Unassigned").sum()
    pct_assigned = n_assigned / len(labels) * 100
    print(f"\n  Assigned: {n_assigned:,} / {len(labels):,} ({pct_assigned:.1f}%)")

    return adata_xen, cell_types


# ── Step 3: Pseudo-bulk aggregation ───────────────────────────────────────

def create_pseudobulk_bins(adata_xen, cell_types, bin_size_um):
    """Create spatial bins and aggregate expression."""
    print(f"\n  Creating {bin_size_um}um bins...")

    coords = adata_xen.obsm["spatial"]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    x_edges = np.arange(x_min, x_max + bin_size_um, bin_size_um)
    y_edges = np.arange(y_min, y_max + bin_size_um, bin_size_um)
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1

    x_idx = np.clip(np.digitize(coords[:, 0], x_edges) - 1, 0, n_x - 1)
    y_idx = np.clip(np.digitize(coords[:, 1], y_edges) - 1, 0, n_y - 1)
    bin_ids = x_idx * n_y + y_idx

    cell_labels = adata_xen.obs["cell_type"].values
    active_types = [ct for ct in cell_types if ct != "Unassigned"]
    ct_to_idx = {ct: i for i, ct in enumerate(active_types)}
    n_types = len(active_types)

    unique_bins = np.unique(bin_ids)
    n_bins_total = len(unique_bins)
    bin_remap = {old: new for new, old in enumerate(unique_bins)}
    remapped_bins = np.array([bin_remap[b] for b in bin_ids])

    gt_counts = np.zeros((n_bins_total, n_types), dtype=np.float32)
    for cell_i in range(len(cell_labels)):
        ct = cell_labels[cell_i]
        if ct in ct_to_idx:
            gt_counts[remapped_bins[cell_i], ct_to_idx[ct]] += 1

    cells_per_bin = gt_counts.sum(axis=1)
    gt_props = gt_counts / np.maximum(cells_per_bin[:, None], 1)

    X_raw = adata_xen.layers["counts"] if "counts" in adata_xen.layers else adata_xen.X
    agg_rows = remapped_bins
    agg_cols = np.arange(len(remapped_bins))
    agg_mat = sparse.csr_matrix(
        (np.ones(len(agg_cols)), (agg_rows, agg_cols)),
        shape=(n_bins_total, adata_xen.n_obs),
    )
    X_binned = agg_mat @ (X_raw if sparse.issparse(X_raw) else sparse.csr_matrix(X_raw))

    bin_centers = np.zeros((n_bins_total, 2), dtype=np.float32)
    for cell_i in range(len(remapped_bins)):
        bid = remapped_bins[cell_i]
        bin_centers[bid] += coords[cell_i]
    bin_counts_arr = np.bincount(remapped_bins, minlength=n_bins_total).astype(np.float32)
    bin_counts_arr[bin_counts_arr == 0] = 1
    bin_centers /= bin_counts_arr[:, None]

    total_umi = np.array(X_binned.sum(axis=1)).flatten()
    valid_mask = total_umi > 0
    n_valid = valid_mask.sum()

    X_binned = X_binned[valid_mask]
    bin_centers = bin_centers[valid_mask]
    gt_props = gt_props[valid_mask]
    gt_counts_valid = gt_counts[valid_mask]
    cells_per_bin = cells_per_bin[valid_mask]

    adata_binned = sc.AnnData(X=X_binned)
    adata_binned.var_names = adata_xen.var_names.copy()
    adata_binned.obs_names = [f"bin_{i}" for i in range(n_valid)]
    adata_binned.obsm["spatial"] = bin_centers

    print(f"    Total bins: {n_valid:,}")
    print(f"    Median cells per bin: {np.median(cells_per_bin):.1f}")

    return adata_binned, gt_props, gt_counts_valid, cells_per_bin, active_types


# ── Step 4: Run FlashDeconv with specified lambda ────────────────────────

def run_flashdeconv(adata_binned, adata_ref, lambda_spatial, cell_type_key="Cell_subtype"):
    """Run FlashDeconv with a specific lambda_spatial setting."""
    print(f"\n  Running FlashDeconv (lambda_spatial={lambda_spatial}) on {adata_binned.n_obs:,} bins...")

    n_overlap = len(set(adata_binned.var_names) & set(adata_ref.var_names))
    print(f"    Gene overlap with reference: {n_overlap}")

    key_added = f"flashdeconv_lambda_{lambda_spatial}"

    fd.tl.deconvolve(
        adata_binned,
        adata_ref,
        cell_type_key=cell_type_key,
        sketch_dim=256,
        n_hvg=min(400, n_overlap),
        n_markers_per_type=30,
        lambda_spatial=lambda_spatial,
        preprocess="log_cpm",
        random_state=RANDOM_SEED,
        key_added=key_added,
    )

    return adata_binned, key_added


# ── Step 5: Compute metrics ──────────────────────────────────────────────

def compute_metrics(adata_binned, gt_props, cells_per_bin, active_types,
                    bin_size_um, condition, key_added):
    """Compute per-cell-type deconvolution accuracy metrics."""
    print(f"\n  Computing metrics for {bin_size_um}um, {condition}...")

    pred_props_df = adata_binned.obsm[key_added]
    pred_cols = list(pred_props_df.columns)

    shared_types = [ct for ct in active_types if ct in pred_cols]
    ct_to_idx = {ct: i for i, ct in enumerate(active_types)}

    pred_aligned = pred_props_df[shared_types].values
    gt_aligned = gt_props[:, [ct_to_idx[ct] for ct in shared_types]]

    n_bins = len(gt_aligned)
    per_type_rows = []

    for j, ct in enumerate(shared_types):
        p_col = pred_aligned[:, j]
        g_col = gt_aligned[:, j]
        gt_mean = g_col.mean()

        # Pearson r
        if p_col.std() > 0 and g_col.std() > 0:
            r_val, r_pval = pearsonr(p_col, g_col)
        else:
            r_val, r_pval = np.nan, np.nan

        # RMSE
        rmse_val = np.sqrt(np.mean((p_col - g_col) ** 2))

        # AUPRC (binary: present if proportion > threshold)
        true_bin = (g_col > PRESENCE_THRESHOLD).astype(int)
        if true_bin.sum() > 0 and true_bin.sum() < len(true_bin):
            prec_curve, rec_curve, _ = precision_recall_curve(true_bin, p_col)
            auprc = auc(rec_curve, prec_curve)
        else:
            auprc = np.nan

        # Abundance category
        if gt_mean < 0.05:
            category = "rare"
        elif gt_mean < 0.15:
            category = "moderate"
        else:
            category = "abundant"

        per_type_rows.append({
            "cell_type": ct,
            "bin_size_um": bin_size_um,
            "condition": condition,
            "pearson_r": r_val,
            "rmse": rmse_val,
            "auprc": auprc,
            "gt_mean_proportion": gt_mean,
            "pred_mean_proportion": p_col.mean(),
            "category": category,
        })

    # Overall Pearson r
    p_flat = pred_aligned.flatten()
    g_flat = gt_aligned.flatten()
    mask_nz = (p_flat > 0) | (g_flat > 0)
    if mask_nz.sum() > 0:
        overall_r, _ = pearsonr(p_flat[mask_nz], g_flat[mask_nz])
    else:
        overall_r = np.nan
    overall_rmse = np.sqrt(np.mean((p_flat - g_flat) ** 2))

    print(f"    Overall Pearson r: {overall_r:.4f}, RMSE: {overall_rmse:.4f}")

    return pd.DataFrame(per_type_rows), overall_r, overall_rmse


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CRC Xenium Lambda Ablation: auto vs lambda=0")
    print("=" * 70)

    # Load data
    adata_xen = load_xenium()
    adata_ref = load_reference()

    # Annotate cells
    adata_xen, cell_types = annotate_xenium_cells(adata_xen, adata_ref)

    # Run for each bin size and lambda setting
    all_per_type = []
    summary_rows = []

    for bin_size in BIN_SIZES_UM:
        print(f"\n{'=' * 70}")
        print(f"Processing {bin_size}um bins")
        print(f"{'=' * 70}")

        adata_binned, gt_props, gt_counts, cells_per_bin, active_types = \
            create_pseudobulk_bins(adata_xen, cell_types, bin_size)

        for condition, lambda_val in LAMBDA_VALUES:
            t0 = time.time()
            adata_binned, key_added = run_flashdeconv(
                adata_binned, adata_ref, lambda_spatial=lambda_val
            )
            elapsed = time.time() - t0

            per_type_df, overall_r, overall_rmse = compute_metrics(
                adata_binned, gt_props, cells_per_bin, active_types,
                bin_size, condition, key_added
            )
            all_per_type.append(per_type_df)

            summary_rows.append({
                "bin_size_um": bin_size,
                "condition": condition,
                "overall_pearson_r": overall_r,
                "overall_rmse": overall_rmse,
                "time_seconds": elapsed,
            })

            # Category-level summary
            for cat in ["rare", "moderate", "abundant"]:
                sub = per_type_df[per_type_df["category"] == cat]
                if len(sub) > 0:
                    print(f"    {cat:>10s} ({len(sub)} types): "
                          f"AUPRC={sub['auprc'].mean():.4f}  "
                          f"Pearson={sub['pearson_r'].mean():.4f}  "
                          f"RMSE={sub['rmse'].mean():.4f}")

    # ── Save results ──
    all_df = pd.concat(all_per_type, ignore_index=True)
    csv_path = OUT_DIR / "xenium_crc_lambda_ablation_per_celltype.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nPer-cell-type results saved to {csv_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "xenium_crc_lambda_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # ── Paired comparison ──
    print("\n" + "=" * 70)
    print("PAIRED COMPARISON: auto lambda vs no spatial (lambda=0)")
    print("=" * 70)

    for bin_size in BIN_SIZES_UM:
        print(f"\n  --- {bin_size}um bins ---")
        df_bin = all_df[all_df["bin_size_um"] == bin_size]

        df_auto = df_bin[df_bin["condition"] == "auto"].set_index("cell_type")
        df_none = df_bin[df_bin["condition"] == "no_spatial"].set_index("cell_type")

        common_types = df_auto.index.intersection(df_none.index)

        paired = pd.DataFrame({
            "category": df_auto.loc[common_types, "category"].values,
            "gt_mean": df_auto.loc[common_types, "gt_mean_proportion"].values,
            "auprc_auto": df_auto.loc[common_types, "auprc"].values,
            "auprc_none": df_none.loc[common_types, "auprc"].values,
            "pearson_auto": df_auto.loc[common_types, "pearson_r"].values,
            "pearson_none": df_none.loc[common_types, "pearson_r"].values,
            "rmse_auto": df_auto.loc[common_types, "rmse"].values,
            "rmse_none": df_none.loc[common_types, "rmse"].values,
        }, index=common_types)

        paired["delta_auprc"] = paired["auprc_auto"] - paired["auprc_none"]
        paired["delta_pearson"] = paired["pearson_auto"] - paired["pearson_none"]
        paired["delta_rmse"] = paired["rmse_auto"] - paired["rmse_none"]

        paired_path = OUT_DIR / f"xenium_crc_lambda_paired_{bin_size}um.csv"
        paired.to_csv(paired_path)

        print(f"\n  {'Category':>12s}  {'N':>3s}  {'ΔAUPRC':>8s}  {'ΔPearson':>9s}  {'ΔRMSE':>8s}")
        print("  " + "-" * 50)

        for cat in ["rare", "moderate", "abundant", "ALL"]:
            sub = paired if cat == "ALL" else paired[paired["category"] == cat]
            sub_valid = sub.dropna(subset=["delta_auprc"])
            if len(sub_valid) == 0:
                continue
            print(f"  {cat:>12s}  {len(sub_valid):>3d}  "
                  f"{sub_valid['delta_auprc'].mean():>+8.4f}  "
                  f"{sub_valid['delta_pearson'].mean():>+9.4f}  "
                  f"{sub_valid['delta_rmse'].mean():>+8.4f}")

        # Detailed rare types
        rare = paired[paired["category"] == "rare"].dropna(subset=["delta_auprc"])
        if len(rare) > 0:
            print(f"\n  Rare cell types ({bin_size}um):")
            print(f"  {'CellType':>25s} {'gt%':>6s} {'AUPRC_a':>8s} {'AUPRC_0':>8s} {'Δ':>7s} {'r_auto':>7s} {'r_0':>7s}")
            print("  " + "-" * 75)
            for ct, row in rare.sort_values("gt_mean").iterrows():
                print(f"  {ct:>25s} {row['gt_mean']*100:>5.2f}% "
                      f"{row['auprc_auto']:>8.3f} {row['auprc_none']:>8.3f} "
                      f"{row['delta_auprc']:>+7.3f} "
                      f"{row['pearson_auto']:>7.3f} {row['pearson_none']:>7.3f}")

    print(f"\n  Output directory: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"    {f.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
