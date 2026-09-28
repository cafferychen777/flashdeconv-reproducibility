"""
Xenium-Derived Pseudo-Visium HD Benchmark.

Generates ground-truth spatial transcriptomics benchmarks from Xenium
single-cell data by:
  1. Spatial binning at multiple resolutions (2, 4, 8, 16, 32 μm)
  2. Aggregating transcript counts per bin (with optional UMI downsampling)
  3. Computing ground-truth cell-type proportions from cell assignments
  4. Running FlashDeconv (auto λ, λ=0), marker scoring, and NNLS
  5. Reporting per-type and aggregate metrics at each resolution

Primary dataset: CRC P1 Xenium (Oliveira et al., Nature Genetics 2025)
  - 289,352 annotated cells, 422 genes, 39 Level2 types
  - scRNA-seq reference: 279,609 cells, 18,082 genes

This experiment directly addresses Reviewer 1's request for high-resolution
Visium HD simulation benchmarks using exact cell coordinates and transcripts.

Usage:
  python xenium_pseudo_visiumhd_benchmark.py [--bin-sizes 4 8 16 32]
                                              [--downsample 1.0]
                                              [--rctd-prep]
                                              [--output-dir DIR]
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr
from scipy.optimize import nnls
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_recall_curve, auc

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flashdeconv as fd
from flashdeconv.core.deconv import FlashDeconv

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(
    os.environ.get(
        "FLASHDECONV_PROJECT_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()
COHORT_DIR = BASE / "data" / "visium_hd_crc_cohort"
XENIUM_ANNOT = BASE / "analysis" / "xenium_validation_results" / "xenium_P1_CRC_annotated.h5ad"
REF_H5 = COHORT_DIR / "scRNA_ref" / "HumanColonCancer_Flex_filtered_feature_bc_matrix.h5"
REF_META = COHORT_DIR / "metadata" / "SingleCell_MetaData.csv.gz"

DEFAULT_OUTPUT = BASE / "validation" / "xenium_pseudo_visiumhd_results"
DEFAULT_BIN_SIZES = [4, 8, 16, 32]

# Real Visium HD median UMI per bin (from mouse intestine QC, qc_metrics_summary.csv)
VHD_TARGET_MEDIAN_UMI = {
    2: 17,
    4: 78,
    8: 325,
    16: 1314,
    32: 5256,   # extrapolated: 1314 × 4 (area scaling)
}

# Lineage mapping (consistent with xenium_crc_validation.py)
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


# ── Data Loading ─────────────────────────────────────────────────────────

def load_xenium(path: Path = XENIUM_ANNOT):
    """Load pre-annotated Xenium CRC P1 data."""
    print(f"Loading Xenium data from {path.name}...")
    adata = sc.read_h5ad(path)
    print(f"  {adata.n_obs:,} cells, {adata.n_vars} genes")
    print(f"  Cell types: {adata.obs['Level2'].nunique()}")
    return adata


def load_reference(h5_path: Path = REF_H5, meta_path: Path = REF_META):
    """Load scRNA-seq reference with QC filtering."""
    print("Loading scRNA-seq reference...")
    adata_ref = sc.read_10x_h5(h5_path)
    adata_ref.var_names_make_unique()
    meta = pd.read_csv(meta_path, compression="gzip").set_index("Barcode")
    common_bc = adata_ref.obs_names.intersection(meta.index)
    adata_ref = adata_ref[common_bc].copy()
    keep = meta.loc[adata_ref.obs_names, "QCFilter"] == "Keep"
    adata_ref = adata_ref[keep].copy()
    adata_ref.obs["Level2"] = meta.loc[adata_ref.obs_names, "Level2"].values
    print(f"  {adata_ref.n_obs:,} cells, {adata_ref.obs['Level2'].nunique()} types")
    return adata_ref


# ── Spatial Binning ──────────────────────────────────────────────────────

def create_bins(adata_xen, bin_size_um, downsample_rate=1.0,
                target_median_umi=None, rng=None):
    """
    Bin Xenium cells at a given spatial resolution.

    Returns:
        Y_binned: (n_bins, n_genes) aggregated expression matrix (dense)
        gt_props: (n_bins, n_types) ground-truth proportions
        gt_counts: (n_bins, n_types) ground-truth cell counts
        bin_centers: (n_bins, 2) spatial coordinates
        cell_types: list of cell type names
        stats: dict with summary statistics
    """
    coords = adata_xen.obsm["spatial"]
    cell_labels = adata_xen.obs["Level2"].values

    # Exclude Unassigned cells from ground truth but keep for expression
    cell_types = sorted([ct for ct in adata_xen.obs["Level2"].unique()
                         if ct != "Unassigned"])
    ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    n_types = len(cell_types)

    # Spatial grid
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_edges = np.arange(x_min, x_max + bin_size_um, bin_size_um)
    y_edges = np.arange(y_min, y_max + bin_size_um, bin_size_um)
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1

    # Assign cells to bins
    x_idx = np.clip(np.digitize(coords[:, 0], x_edges) - 1, 0, n_x - 1)
    y_idx = np.clip(np.digitize(coords[:, 1], y_edges) - 1, 0, n_y - 1)
    flat_bin_ids = x_idx * n_y + y_idx

    # Ground truth cell counts per bin
    gt_all = np.zeros((n_x * n_y, n_types), dtype=np.float32)
    for i in range(len(cell_labels)):
        ct = cell_labels[i]
        if ct in ct_to_idx:
            gt_all[flat_bin_ids[i], ct_to_idx[ct]] += 1

    # Filter: keep bins with >= 1 annotated cell
    cells_per_bin = gt_all.sum(axis=1)
    non_empty = cells_per_bin >= 1
    unique_bins = np.where(non_empty)[0]
    n_bins = len(unique_bins)
    bin_remap = {old: new for new, old in enumerate(unique_bins)}

    gt_counts = gt_all[unique_bins]
    gt_props = gt_counts / np.maximum(gt_counts.sum(axis=1, keepdims=True), 1)

    # Bin centers (mean of cell centroids, not grid center)
    remapped = np.array([bin_remap.get(b, -1) for b in flat_bin_ids])
    bin_centers = np.zeros((n_bins, 2), dtype=np.float32)
    bin_cell_counts = np.zeros(n_bins, dtype=np.float32)
    for i in range(len(remapped)):
        bid = remapped[i]
        if bid >= 0:
            bin_centers[bid] += coords[i]
            bin_cell_counts[bid] += 1
    bin_cell_counts[bin_cell_counts == 0] = 1
    bin_centers /= bin_cell_counts[:, None]

    # Aggregate expression via sparse aggregation matrix
    X_raw = adata_xen.layers["counts"] if "counts" in adata_xen.layers else adata_xen.X
    valid_mask = remapped >= 0
    rows = remapped[valid_mask]
    cols = np.arange(adata_xen.n_obs)[valid_mask]
    agg_mat = sparse.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), (rows, cols)),
        shape=(n_bins, adata_xen.n_obs),
    )
    Y_binned = agg_mat @ (X_raw if sparse.issparse(X_raw) else sparse.csr_matrix(X_raw))

    if sparse.issparse(Y_binned):
        Y_dense = Y_binned.toarray().astype(np.float32)
    else:
        Y_dense = np.array(Y_binned, dtype=np.float32)

    # UMI downsampling (simulate Visium HD capture efficiency)
    effective_rate = 1.0
    if target_median_umi is not None and rng is not None:
        # Adaptive: match Visium HD UMI density at this bin size
        current_median_umi = np.median(Y_dense.sum(axis=1))
        if current_median_umi > target_median_umi:
            effective_rate = target_median_umi / current_median_umi
            Y_dense = rng.binomial(Y_dense.astype(int), effective_rate).astype(np.float32)
            print(f"    Adaptive downsample: {current_median_umi:.0f} → "
                  f"{target_median_umi} UMI (rate={effective_rate:.4f})")
        else:
            print(f"    No downsampling needed ({current_median_umi:.0f} ≤ {target_median_umi})")
    elif downsample_rate < 1.0 and rng is not None:
        effective_rate = downsample_rate
        Y_dense = rng.binomial(Y_dense.astype(int), downsample_rate).astype(np.float32)
        print(f"    Fixed downsample: rate={downsample_rate:.4f}")

    avg_cells = gt_counts.sum(axis=1).mean()
    avg_umi = Y_dense.sum(axis=1).mean()
    multi_cell_pct = (gt_counts.sum(axis=1) > 1).mean() * 100

    stats = {
        "bin_size_um": bin_size_um,
        "n_bins": n_bins,
        "avg_cells_per_bin": avg_cells,
        "avg_umi_per_bin": avg_umi,
        "multi_cell_pct": multi_cell_pct,
        "median_cells_per_bin": np.median(gt_counts.sum(axis=1)),
        "downsample_rate": effective_rate,
    }

    print(f"  {bin_size_um}μm: {n_bins:,} bins, "
          f"avg {avg_cells:.1f} cells/bin, "
          f"avg {avg_umi:.0f} UMI/bin, "
          f"{multi_cell_pct:.1f}% multi-cell")

    return Y_dense, gt_props, gt_counts, bin_centers, cell_types, stats


# ── Deconvolution Methods ────────────────────────────────────────────────

def run_flashdeconv_raw(Y, X_ref, coords, cell_types, lambda_spatial="auto",
                        rho_sparsity=0.01, sketch_dim=512,
                        preprocess="log_cpm", random_state=0):
    """Run FlashDeconv with pre-built signature matrix."""
    model = FlashDeconv(
        sketch_dim=min(sketch_dim, Y.shape[1]),
        lambda_spatial=lambda_spatial,
        rho_sparsity=rho_sparsity,
        preprocess=preprocess,
        n_hvg=min(2000, Y.shape[1]),
        n_markers_per_type=50,
        max_iter=500,
        tol=1e-6,
        verbose=False,
        random_state=random_state,
    )
    props = model.fit_transform(Y, X_ref, coords, cell_type_names=cell_types)
    return props


def run_flashdeconv_ensemble(
    Y,
    X_ref,
    coords,
    cell_types,
    lambda_spatial,
    seeds,
):
    """Average matched production-pipeline fits before evaluation."""
    prediction_sum = None
    runtimes = []
    for seed in seeds:
        started = time.time()
        prediction = run_flashdeconv_raw(
            Y,
            X_ref,
            coords,
            cell_types,
            lambda_spatial=lambda_spatial,
            rho_sparsity=0.01,
            random_state=seed,
        )
        runtimes.append(time.time() - started)
        if prediction_sum is None:
            prediction_sum = prediction.astype(np.float64, copy=True)
        else:
            prediction_sum += prediction
        print(
            f"      seed {seed}: {runtimes[-1]:.2f}s",
            flush=True,
        )
    return prediction_sum / len(seeds), float(np.mean(runtimes))


def run_nnls(Y, X_ref):
    """Run per-spot NNLS deconvolution (no spatial regularization)."""
    n_spots = Y.shape[0]
    n_types = X_ref.shape[0]
    props = np.zeros((n_spots, n_types), dtype=np.float64)

    for i in range(n_spots):
        y = Y[i]
        if y.sum() == 0:
            continue
        beta, _ = nnls(X_ref.T, y)
        s = beta.sum()
        if s > 0:
            props[i] = beta / s

    return props


def run_marker_scoring(Y, X_ref, cell_types, n_markers=50):
    """
    Marker gene scoring: for each cell type, identify top markers by
    fold change, then score each bin by normalized mean expression of
    those markers.

    This is the baseline approach used in many spatial analysis tools.
    """
    n_types, n_genes = X_ref.shape

    # Compute fold change for each gene per type vs. rest
    mean_all = X_ref.mean(axis=0) + 1e-6
    scores_all = np.zeros((Y.shape[0], n_types))

    for k in range(n_types):
        fc = (X_ref[k] + 1e-6) / mean_all
        # Top markers by fold change
        top_idx = np.argsort(fc)[-n_markers:]

        # Score: mean expression of marker genes in each bin (CPM-normalized)
        y_sum = Y.sum(axis=1, keepdims=True)
        y_sum[y_sum == 0] = 1
        y_cpm = Y / y_sum * 1e4
        scores_all[:, k] = y_cpm[:, top_idx].mean(axis=1)

    # Normalize to proportions
    row_sums = scores_all.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    props = scores_all / row_sums

    return props


def run_rctd_py(Y_overlap, adata_xen, bin_centers, cell_types,
                overlap_genes, mode="full", max_ref_cells=20000,
                batch_size=30000):
    """
    Run RCTD-py (Python re-implementation of spacexr RCTD).

    Uses Xenium single-cell data as reference; binned expression as spatial.
    GPU-accelerated via PyTorch when available.

    Processes bins in batches to avoid CUDA OOM on large datasets.
    """
    import anndata
    import gc
    from rctd import Reference, run_rctd

    # Reference: Xenium single cells, overlap genes, exclude Unassigned
    ref = adata_xen[:, overlap_genes].copy()
    ref.obs["cell_type"] = ref.obs["Level2"].astype(str)
    ref = ref[ref.obs["cell_type"] != "Unassigned"].copy()

    if ref.n_obs > max_ref_cells:
        rng_sub = np.random.default_rng(42)
        idx = rng_sub.choice(ref.n_obs, max_ref_cells, replace=False)
        ref = ref[idx].copy()
        print(f"    Reference subsampled to {max_ref_cells:,} cells")

    reference = Reference(ref, cell_type_col="cell_type")

    n_bins = Y_overlap.shape[0]
    n_types = len(cell_types)
    props = np.zeros((n_bins, n_types), dtype=np.float64)

    # Process in batches to fit in GPU memory
    n_batches = max(1, (n_bins + batch_size - 1) // batch_size)
    print(f"    Processing {n_bins:,} bins in {n_batches} batch(es) of ≤{batch_size:,}")

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_bins)
        batch_Y = Y_overlap[start:end]
        batch_coords = bin_centers[start:end]

        spatial = anndata.AnnData(
            X=sparse.csr_matrix(batch_Y.astype(np.float32)),
            var=pd.DataFrame(index=overlap_genes),
            obs=pd.DataFrame(index=[f"bin_{i}" for i in range(start, end)]),
        )
        spatial.obsm["spatial"] = batch_coords.astype(np.float64)

        try:
            result = run_rctd(spatial, reference, mode=mode)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    GPU OOM on batch {b+1}, falling back to CPU...")
                import torch
                torch.cuda.empty_cache()
                gc.collect()
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                result = run_rctd(spatial, reference, mode=mode)
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                raise

        # FullResult is a NamedTuple: weights, cell_type_names, converged, pixel_mask
        batch_weights = np.array(result.weights, dtype=np.float64)
        rctd_ct_names = result.cell_type_names

        rctd_ct_to_j = {}
        for j, ct in enumerate(cell_types):
            if ct in rctd_ct_names:
                rctd_ct_to_j[rctd_ct_names.index(ct)] = j

        if result.pixel_mask is not None:
            kept_idx = np.where(result.pixel_mask)[0]
        else:
            kept_idx = np.arange(end - start)

        for rctd_k, our_j in rctd_ct_to_j.items():
            props[start + kept_idx, our_j] = batch_weights[:, rctd_k]

        # Free GPU memory between batches
        del result, spatial
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        if n_batches > 1:
            print(f"    Batch {b+1}/{n_batches} done ({end-start} bins)")

    # Normalize to proportions
    row_sums = props.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    props = props / row_sums
    return props


# ── Reference Matrix Builder ─────────────────────────────────────────────

def build_reference_from_xenium(adata_xen, cell_types):
    """
    Build (n_types x n_genes) signature matrix directly from Xenium data.

    Uses raw counts from annotated Xenium cells to compute mean expression
    per cell type. All Xenium genes are used (no overlap issue).
    """
    X_raw = adata_xen.layers["counts"] if "counts" in adata_xen.layers else adata_xen.X
    labels = adata_xen.obs["Level2"].values
    n_genes = adata_xen.n_vars
    n_types = len(cell_types)
    X_sig = np.zeros((n_types, n_genes), dtype=np.float64)

    for k, ct in enumerate(cell_types):
        mask = labels == ct
        n_ct = mask.sum()
        if n_ct == 0:
            print(f"  Warning: {ct} has 0 cells, using global mean")
            sub = X_raw
        else:
            sub = X_raw[mask]
        if sparse.issparse(sub):
            sub = sub.toarray()
        X_sig[k] = sub.mean(axis=0)

    gene_idx = list(range(n_genes))
    gene_names = list(adata_xen.var_names)
    print(f"  Xenium self-reference: {n_types} types, {n_genes} genes (all Xenium panel)")
    return X_sig, gene_idx, gene_names


def build_reference_from_scrnaseq(adata_ref, adata_xen, cell_types):
    """
    Build (n_types x n_overlap_genes) signature matrix from scRNA-seq reference.

    Genes are aligned to Xenium panel order.
    """
    xen_genes = list(adata_xen.var_names)
    ref_genes = list(adata_ref.var_names)

    overlap = sorted(set(xen_genes) & set(ref_genes))
    print(f"  Overlapping genes: {len(overlap)} / {len(xen_genes)} Xenium genes")

    ref_gene_idx = {g: i for i, g in enumerate(ref_genes)}
    xen_gene_idx = {g: i for i, g in enumerate(xen_genes)}

    ref_types = adata_ref.obs["Level2"].values
    X_ref_raw = adata_ref.X

    n_types = len(cell_types)
    n_overlap = len(overlap)
    X_sig = np.zeros((n_types, n_overlap), dtype=np.float64)

    overlap_ref_indices = [ref_gene_idx[g] for g in overlap]

    for k, ct in enumerate(cell_types):
        mask = ref_types == ct
        n_ct = mask.sum()
        if n_ct == 0:
            print(f"  Warning: {ct} not in reference, using global mean")
            sub = X_ref_raw[:, overlap_ref_indices]
        else:
            sub = X_ref_raw[mask][:, overlap_ref_indices]
        if sparse.issparse(sub):
            sub = sub.toarray()
        X_sig[k] = sub.mean(axis=0)

    overlap_xen_indices = [xen_gene_idx[g] for g in overlap]

    return X_sig, overlap_xen_indices, overlap


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(pred, true, cell_types):
    """Compute comprehensive deconvolution metrics."""
    records = []
    n_types = pred.shape[1]

    # Per-cell-type Pearson r
    per_type_r = {}
    per_type_rmse = {}
    per_type_auprc = {}
    for j, ct in enumerate(cell_types):
        p_col = pred[:, j]
        g_col = true[:, j]
        if p_col.std() > 0 and g_col.std() > 0:
            per_type_r[ct] = pearsonr(p_col, g_col)[0]
        else:
            per_type_r[ct] = np.nan

        per_type_rmse[ct] = np.sqrt(np.mean((p_col - g_col) ** 2))

        # AUPRC: binary detection (present > 1% vs absent)
        true_binary = (g_col > 0.01).astype(int)
        if true_binary.sum() > 0 and true_binary.sum() < len(true_binary):
            prec, rec, _ = precision_recall_curve(true_binary, p_col)
            per_type_auprc[ct] = auc(rec, prec)
        else:
            per_type_auprc[ct] = np.nan

    # Aggregate metrics (flattened)
    p_flat = pred.flatten()
    g_flat = true.flatten()
    nz_mask = (p_flat > 0) | (g_flat > 0)

    global_r = pearsonr(p_flat[nz_mask], g_flat[nz_mask])[0] if nz_mask.sum() > 10 else np.nan
    global_rmse = np.sqrt(np.mean((p_flat - g_flat) ** 2))

    # JSD
    jsd_vals = []
    for i in range(pred.shape[0]):
        p = pred[i] + 1e-10
        q = true[i] + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        jsd_vals.append(jensenshannon(p, q) ** 2)
    mean_jsd = np.nanmean(jsd_vals)

    # Global AUPRC
    true_binary_all = (g_flat > 0.01).astype(int)
    if true_binary_all.sum() > 0 and true_binary_all.sum() < len(true_binary_all):
        prec_all, rec_all, _ = precision_recall_curve(true_binary_all, p_flat)
        global_auprc = auc(rec_all, prec_all)
    else:
        global_auprc = np.nan

    # Per-lineage Pearson r
    lineages = sorted(set(LINEAGE_MAP.values()) - {"Other"})
    per_lineage_r = {}
    for lin in lineages:
        lin_types = [ct for ct in cell_types if LINEAGE_MAP.get(ct) == lin]
        if not lin_types:
            continue
        lin_idx = [cell_types.index(ct) for ct in lin_types]
        lin_pred = pred[:, lin_idx].sum(axis=1)
        lin_true = true[:, lin_idx].sum(axis=1)
        if lin_pred.std() > 0 and lin_true.std() > 0:
            per_lineage_r[lin] = pearsonr(lin_pred, lin_true)[0]

    return {
        "global_r": global_r,
        "global_rmse": global_rmse,
        "global_jsd": mean_jsd,
        "global_auprc": global_auprc,
        "mean_per_type_r": np.nanmean(list(per_type_r.values())),
        "mean_per_type_rmse": np.nanmean(list(per_type_rmse.values())),
        "mean_per_type_auprc": np.nanmean(list(per_type_auprc.values())),
        "mean_per_lineage_r": np.nanmean(list(per_lineage_r.values())),
        "per_type_r": per_type_r,
        "per_type_rmse": per_type_rmse,
        "per_type_auprc": per_type_auprc,
        "per_lineage_r": per_lineage_r,
    }


# ── RCTD Input Preparation ──────────────────────────────────────────────

def prepare_rctd_inputs(Y, gt_counts, bin_centers, cell_types, adata_ref,
                        overlap_genes, overlap_xen_indices,
                        bin_size_um, output_dir):
    """
    Prepare input files for RCTD (R package spacexr).

    Creates: counts.csv, coords.csv, ref_counts.csv, ref_celltypes.csv
    """
    rctd_dir = output_dir / f"rctd_input_{bin_size_um}um"
    rctd_dir.mkdir(parents=True, exist_ok=True)

    # Spatial counts (bins x genes)
    Y_overlap = Y[:, overlap_xen_indices]
    counts_df = pd.DataFrame(
        Y_overlap.astype(int),
        index=[f"bin_{i}" for i in range(Y_overlap.shape[0])],
        columns=overlap_genes,
    )
    counts_df.T.to_csv(rctd_dir / "counts.csv")

    # Coordinates
    coords_df = pd.DataFrame(
        bin_centers, index=counts_df.index, columns=["x", "y"]
    )
    coords_df.to_csv(rctd_dir / "coords.csv")

    # Reference counts (cells x genes) — subsample for memory
    ref_genes = list(adata_ref.var_names)
    ref_gene_idx = {g: i for i, g in enumerate(ref_genes)}
    ref_overlap_idx = [ref_gene_idx[g] for g in overlap_genes]

    X_ref = adata_ref.X[:, ref_overlap_idx]
    if sparse.issparse(X_ref):
        X_ref = X_ref.toarray()

    # Subsample reference to max 50K cells for RCTD tractability
    n_ref = X_ref.shape[0]
    max_ref = 50000
    if n_ref > max_ref:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_ref, max_ref, replace=False)
        X_ref_sub = X_ref[idx]
        ref_labels = adata_ref.obs["Level2"].values[idx]
        ref_barcodes = adata_ref.obs_names[idx]
    else:
        X_ref_sub = X_ref
        ref_labels = adata_ref.obs["Level2"].values
        ref_barcodes = adata_ref.obs_names

    ref_counts_df = pd.DataFrame(
        X_ref_sub.astype(int),
        index=ref_barcodes,
        columns=overlap_genes,
    )
    ref_counts_df.T.to_csv(rctd_dir / "ref_counts.csv")

    # Reference cell types
    ref_ct_df = pd.DataFrame(
        {"barcode": ref_barcodes, "cell_type": ref_labels}
    )
    ref_ct_df.to_csv(rctd_dir / "ref_celltypes.csv", index=False)

    print(f"  RCTD inputs saved to {rctd_dir}/")
    return rctd_dir


# ── Main Benchmark Pipeline ─────────────────────────────────────────────

def run_benchmark(bin_sizes=None, downsample_rate=1.0, downsample_vhd=False,
                  run_rctd=False, prepare_rctd=False,
                  output_dir=None, ref_mode="xenium", seeds=None):
    """Run the full pseudo-Visium HD benchmark.

    Parameters
    ----------
    ref_mode : {"xenium", "scrnaseq"}
        "xenium": build reference signatures from Xenium cell annotations
                  (self-contained, no external data needed)
        "scrnaseq": build from Chromium Flex scRNA-seq reference
                    (requires REF_H5 and REF_META files)
    """
    if bin_sizes is None:
        bin_sizes = DEFAULT_BIN_SIZES
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT
    if seeds is None:
        seeds = list(range(20))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Xenium-Derived Pseudo-Visium HD Benchmark")
    print("=" * 70)
    print(f"  Bin sizes: {bin_sizes} μm")
    print(f"  UMI downsample: {'VHD-adaptive' if downsample_vhd else downsample_rate}")
    print(f"  RCTD-py: {'enabled' if run_rctd else 'disabled'}")
    print(f"  Reference mode: {ref_mode}")
    print(f"  FlashDeconv seeds: {seeds}")
    print(f"  Output: {output_dir}")

    # Load data
    adata_xen = load_xenium()
    adata_ref = None
    if ref_mode == "scrnaseq":
        adata_ref = load_reference()

    # Exclude Unassigned for ground truth types
    cell_types = sorted([ct for ct in adata_xen.obs["Level2"].unique()
                         if ct != "Unassigned"])
    print(f"  Target cell types: {len(cell_types)}")

    # Build reference signature matrix
    print("\nBuilding reference signature matrix...")
    if ref_mode == "xenium":
        X_sig, overlap_xen_idx, overlap_genes = build_reference_from_xenium(
            adata_xen, cell_types
        )
    else:
        X_sig, overlap_xen_idx, overlap_genes = build_reference_from_scrnaseq(
            adata_ref, adata_xen, cell_types
        )
    print(f"  Signature matrix: ({X_sig.shape[0]} types, {X_sig.shape[1]} genes)")

    rng = np.random.default_rng(42)
    all_records = []
    per_type_records = []

    for bin_size in bin_sizes:
        print(f"\n{'=' * 60}")
        print(f"  Resolution: {bin_size} μm")
        print(f"{'=' * 60}")

        # Create bins (with optional VHD-adaptive downsampling)
        target_umi = VHD_TARGET_MEDIAN_UMI.get(bin_size) if downsample_vhd else None
        Y, gt_props, gt_counts, bin_centers, ct_list, stats = create_bins(
            adata_xen, bin_size, downsample_rate,
            target_median_umi=target_umi, rng=rng,
        )

        # Subset expression to overlapping genes
        Y_overlap = Y[:, overlap_xen_idx]

        # ── Method 1: FlashDeconv (auto λ) ──
        print(f"\n  Running FlashDeconv (λ=auto)...")
        try:
            fd_auto, t_fd_auto = run_flashdeconv_ensemble(
                Y_overlap, X_sig, bin_centers, cell_types,
                lambda_spatial="auto", seeds=seeds,
            )
            m = compute_metrics(fd_auto, gt_props, cell_types)
            print(f"    Pearson r={m['global_r']:.4f}, "
                  f"mean_type_r={m['mean_per_type_r']:.4f}, "
                  f"AUPRC={m['global_auprc']:.4f}, "
                  f"time={t_fd_auto:.1f}s")
            all_records.append({
                "method": "FlashDeconv_auto", "bin_size_um": bin_size,
                **stats, **{k: v for k, v in m.items()
                            if not isinstance(v, dict)},
                "time_s": t_fd_auto,
                "n_seeds": len(seeds),
            })
            for ct in cell_types:
                per_type_records.append({
                    "method": "FlashDeconv_auto", "bin_size_um": bin_size,
                    "cell_type": ct,
                    "lineage": LINEAGE_MAP.get(ct, "Other"),
                    "pearson_r": m["per_type_r"].get(ct, np.nan),
                    "rmse": m["per_type_rmse"].get(ct, np.nan),
                    "auprc": m["per_type_auprc"].get(ct, np.nan),
                })
        except Exception as e:
            print(f"    Error: {e}")
            t_fd_auto = np.nan

        # ── Method 2: FlashDeconv (λ=0) ──
        print(f"  Running FlashDeconv (λ=0)...")
        try:
            fd_nospatial, t_fd_nospatial = run_flashdeconv_ensemble(
                Y_overlap, X_sig, bin_centers, cell_types,
                lambda_spatial=0.0, seeds=seeds,
            )
            m = compute_metrics(fd_nospatial, gt_props, cell_types)
            print(f"    Pearson r={m['global_r']:.4f}, "
                  f"mean_type_r={m['mean_per_type_r']:.4f}, "
                  f"AUPRC={m['global_auprc']:.4f}, "
                  f"time={t_fd_nospatial:.1f}s")
            all_records.append({
                "method": "FlashDeconv_lambda0", "bin_size_um": bin_size,
                **stats, **{k: v for k, v in m.items()
                            if not isinstance(v, dict)},
                "time_s": t_fd_nospatial,
                "n_seeds": len(seeds),
            })
            for ct in cell_types:
                per_type_records.append({
                    "method": "FlashDeconv_lambda0", "bin_size_um": bin_size,
                    "cell_type": ct,
                    "lineage": LINEAGE_MAP.get(ct, "Other"),
                    "pearson_r": m["per_type_r"].get(ct, np.nan),
                    "rmse": m["per_type_rmse"].get(ct, np.nan),
                    "auprc": m["per_type_auprc"].get(ct, np.nan),
                })
        except Exception as e:
            print(f"    Error: {e}")

        # ── Method 3: NNLS ──
        print(f"  Running NNLS...")
        t0 = time.time()
        try:
            nnls_props = run_nnls(Y_overlap, X_sig)
            t_nnls = time.time() - t0
            m = compute_metrics(nnls_props, gt_props, cell_types)
            print(f"    Pearson r={m['global_r']:.4f}, "
                  f"mean_type_r={m['mean_per_type_r']:.4f}, "
                  f"AUPRC={m['global_auprc']:.4f}, "
                  f"time={t_nnls:.1f}s")
            all_records.append({
                "method": "NNLS", "bin_size_um": bin_size,
                **stats, **{k: v for k, v in m.items()
                            if not isinstance(v, dict)},
                "time_s": t_nnls,
            })
            for ct in cell_types:
                per_type_records.append({
                    "method": "NNLS", "bin_size_um": bin_size,
                    "cell_type": ct,
                    "lineage": LINEAGE_MAP.get(ct, "Other"),
                    "pearson_r": m["per_type_r"].get(ct, np.nan),
                    "rmse": m["per_type_rmse"].get(ct, np.nan),
                    "auprc": m["per_type_auprc"].get(ct, np.nan),
                })
        except Exception as e:
            print(f"    Error: {e}")

        # ── Method 4: Marker Scoring ──
        print(f"  Running Marker Scoring...")
        t0 = time.time()
        try:
            marker_props = run_marker_scoring(Y_overlap, X_sig, cell_types)
            t_marker = time.time() - t0
            m = compute_metrics(marker_props, gt_props, cell_types)
            print(f"    Pearson r={m['global_r']:.4f}, "
                  f"mean_type_r={m['mean_per_type_r']:.4f}, "
                  f"AUPRC={m['global_auprc']:.4f}, "
                  f"time={t_marker:.1f}s")
            all_records.append({
                "method": "MarkerScoring", "bin_size_um": bin_size,
                **stats, **{k: v for k, v in m.items()
                            if not isinstance(v, dict)},
                "time_s": t_marker,
            })
            for ct in cell_types:
                per_type_records.append({
                    "method": "MarkerScoring", "bin_size_um": bin_size,
                    "cell_type": ct,
                    "lineage": LINEAGE_MAP.get(ct, "Other"),
                    "pearson_r": m["per_type_r"].get(ct, np.nan),
                    "rmse": m["per_type_rmse"].get(ct, np.nan),
                    "auprc": m["per_type_auprc"].get(ct, np.nan),
                })
        except Exception as e:
            print(f"    Error: {e}")

        # ── Method 5: RCTD-py (optional, GPU-accelerated) ──
        if run_rctd and bin_size >= 4:
            print(f"  Running RCTD-py (mode=full)...")
            t0 = time.time()
            try:
                rctd_props = run_rctd_py(
                    Y_overlap, adata_xen, bin_centers, cell_types,
                    overlap_genes, mode="full",
                )
                t_rctd = time.time() - t0
                m = compute_metrics(rctd_props, gt_props, cell_types)
                print(f"    Pearson r={m['global_r']:.4f}, "
                      f"mean_type_r={m['mean_per_type_r']:.4f}, "
                      f"AUPRC={m['global_auprc']:.4f}, "
                      f"time={t_rctd:.1f}s")
                all_records.append({
                    "method": "RCTD", "bin_size_um": bin_size,
                    **stats, **{k: v for k, v in m.items()
                                if not isinstance(v, dict)},
                    "time_s": t_rctd,
                })
                for ct in cell_types:
                    per_type_records.append({
                        "method": "RCTD", "bin_size_um": bin_size,
                        "cell_type": ct,
                        "lineage": LINEAGE_MAP.get(ct, "Other"),
                        "pearson_r": m["per_type_r"].get(ct, np.nan),
                        "rmse": m["per_type_rmse"].get(ct, np.nan),
                        "auprc": m["per_type_auprc"].get(ct, np.nan),
                    })
            except Exception as e:
                print(f"    RCTD error: {e}")
                import traceback; traceback.print_exc()

        # ── Prepare RCTD CSV inputs (optional, requires scrnaseq ref) ──
        if prepare_rctd and bin_size >= 8 and adata_ref is not None:
            print(f"  Preparing RCTD input files...")
            prepare_rctd_inputs(
                Y, gt_counts, bin_centers, cell_types, adata_ref,
                overlap_genes, overlap_xen_idx, bin_size, output_dir,
            )

    # ── Save results ──
    summary_df = pd.DataFrame(all_records)
    summary_path = output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    pertype_df = pd.DataFrame(per_type_records)
    pertype_path = output_dir / "benchmark_per_type.csv"
    pertype_df.to_csv(pertype_path, index=False)
    print(f"Saved: {pertype_path}")

    # Print final summary
    print_summary(summary_df)

    return summary_df, pertype_df


def print_summary(df):
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25s} {'Bin(μm)':>8s} {'Pearson':>8s} "
          f"{'Type-r':>8s} {'AUPRC':>8s} {'JSD':>8s} {'Time(s)':>8s}")
    print("-" * 70)
    for _, row in df.sort_values(["bin_size_um", "method"]).iterrows():
        print(f"{row['method']:<25s} {row['bin_size_um']:>8.0f} "
              f"{row['global_r']:>8.4f} {row['mean_per_type_r']:>8.4f} "
              f"{row['global_auprc']:>8.4f} {row['global_jsd']:>8.4f} "
              f"{row['time_s']:>8.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Xenium pseudo-Visium HD benchmark")
    parser.add_argument("--bin-sizes", nargs="+", type=int,
                        default=DEFAULT_BIN_SIZES,
                        help="Bin sizes in μm (default: 4 8 16 32)")
    parser.add_argument("--downsample", type=float, default=1.0,
                        help="UMI downsample rate (default: 1.0 = no downsampling)")
    parser.add_argument("--downsample-vhd", action="store_true",
                        help="Adaptive downsample to match real Visium HD UMI density")
    parser.add_argument("--rctd", action="store_true",
                        help="Run RCTD-py deconvolution (requires rctd-py package)")
    parser.add_argument("--ref-mode", choices=["xenium", "scrnaseq"],
                        default="xenium",
                        help="Reference source: xenium (self-ref) or scrnaseq")
    parser.add_argument("--rctd-prep", action="store_true",
                        help="Prepare RCTD CSV input files")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory")
    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of matched FlashDeconv hash seeds")
    args = parser.parse_args()

    run_benchmark(
        bin_sizes=args.bin_sizes,
        downsample_rate=args.downsample,
        downsample_vhd=args.downsample_vhd,
        run_rctd=args.rctd,
        prepare_rctd=args.rctd_prep,
        output_dir=args.output_dir,
        ref_mode=args.ref_mode,
        seeds=list(range(args.seeds)),
    )


if __name__ == "__main__":
    main()
