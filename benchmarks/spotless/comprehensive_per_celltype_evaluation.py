"""
Comprehensive per-cell-type evaluation across all benchmarks.

Addresses Reviewer 1 (unified Pearson + AUPR + JSD) and Reviewer 2
(per-class AUPRC, precision, recall, F1) demands.

Evaluates FlashDeconv and key competing methods (RCTD, Cell2Location,
SpatialDWLS, Stereoscope) on:
  - Silver Standard: 6 tissues × 9 patterns (rep 1)
  - Liver: 4 Visium slides
  - SeqFISH+: 14 FOVs
  - STARMap: 1 dataset

Output:
  1. per_celltype_all_benchmarks.csv — per-cell-type metrics for all methods
  2. aggregate_all_benchmarks.csv — unified aggregate metrics (Pearson, AUPR, JSD, RMSE, precision, recall, F1)
  3. per_celltype_summary_by_abundance.csv — summary stratified by abundance category
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score
)
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/apple/Research/FlashDeconv")
from flashdeconv import FlashDeconv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/apple/Research/FlashDeconv/validation/benchmark_data/converted"
SPOTLESS_DIR = "/Users/apple/Research/FlashDeconv/validation/spotless/raw_results/deconv_proportions"
OUTPUT_DIR = "/Users/apple/Research/FlashDeconv/validation/results/comprehensive_evaluation"
PRESENCE_THRESHOLD = 0.01
RANDOM_STATE = 42

# Dataset mapping
TISSUE_NAMES = {1: "brain_cortex", 2: "cerebellum_cell", 3: "cerebellum_nucleus",
                4: "hippocampus", 5: "kidney", 6: "scc_p5"}

PATTERN_MAP = {
    1: "artificial_uniform_distinct",
    2: "artificial_diverse_distinct",
    3: "artificial_uniform_overlap",
    4: "artificial_diverse_overlap",
    5: "artificial_dominant_celltype_diverse",
    6: "artificial_partially_dominant_celltype_diverse",
    7: "artificial_dominant_rare_celltype_diverse",
    8: "artificial_regional_rare_celltype_diverse",
    9: "artificial_missing_celltypes_visium",
}

# Methods to evaluate from Spotless (focus on top performers)
SPOTLESS_METHODS = ["rctd", "cell2location", "spatialdwls", "stereoscope",
                    "music", "nnls", "destvi", "spotlight", "tangram",
                    "seurat", "dstg", "stride"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_silver_data(tissue_id, pattern_id):
    """Load silver standard count matrix and ground truth proportions."""
    prefix = f"silver_{tissue_id}_{pattern_id}"
    Y = scipy.io.mmread(os.path.join(DATA_DIR, f"{prefix}_counts.mtx")).toarray().T
    genes = np.loadtxt(os.path.join(DATA_DIR, f"{prefix}_genes.txt"), dtype=str)
    props = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}_proportions.csv"), index_col=0)
    return Y, genes, props


def load_reference_data(tissue_id):
    """Load scRNA-seq reference for a tissue."""
    tissue = TISSUE_NAMES[tissue_id]
    prefix = f"reference_{tissue_id}" if os.path.exists(
        os.path.join(DATA_DIR, f"reference_{tissue_id}_counts.mtx")
    ) else f"reference_{tissue}"
    ref_counts = scipy.io.mmread(os.path.join(DATA_DIR, f"{prefix}_counts.mtx")).toarray().T
    ref_celltypes = np.loadtxt(os.path.join(DATA_DIR, f"{prefix}_celltypes.txt"), dtype=str)
    ref_genes = np.loadtxt(os.path.join(DATA_DIR, f"{prefix}_genes.txt"), dtype=str)
    unique_types = np.unique(ref_celltypes)
    X = np.zeros((len(unique_types), ref_counts.shape[1]))
    for i, ct in enumerate(unique_types):
        X[i] = ref_counts[ref_celltypes == ct].mean(axis=0)
    return X, unique_types, ref_genes


def align_genes(Y, genes_Y, X, genes_X):
    """Align gene sets between spatial and reference data."""
    common = np.intersect1d(genes_Y, genes_X)
    idx_Y = np.array([np.where(genes_Y == g)[0][0] for g in common])
    idx_X = np.array([np.where(genes_X == g)[0][0] for g in common])
    return Y[:, idx_Y], X[:, idx_X], common


def load_spotless_prediction(tissue_name, pattern_name, method, rep=1):
    """Load a Spotless method prediction file."""
    dataset = f"{tissue_name}_{pattern_name}"
    pred_dir = os.path.join(SPOTLESS_DIR, dataset)
    pred_file = os.path.join(pred_dir, f"proportions_{method}_{dataset}_rep{rep}")

    if not os.path.exists(pred_file):
        return None, None

    # Read TSV: header = cell types, rows = spots
    with open(pred_file, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    data = []
    for line in lines[1:]:
        vals = [float(x) for x in line.strip().split('\t')]
        data.append(vals)

    pred = np.array(data)
    # Normalize cell type names (Spotless uses no dots: L23IT -> L2.3.IT)
    return pred, header


def align_celltype_names(pred_types, true_types):
    """Map between Spotless cell type names and our ground truth names.

    Spotless removes dots: L2.3.IT -> L23IT, L5.IT -> L5IT, etc.
    """
    # Build mapping from cleaned name to original
    true_clean = {}
    for t in true_types:
        cleaned = t.replace(".", "")
        true_clean[cleaned] = t

    mapping = {}
    for p in pred_types:
        cleaned = p.replace(".", "")
        if cleaned in true_clean:
            mapping[p] = true_clean[cleaned]
        elif p in true_types:
            mapping[p] = p

    return mapping


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_per_celltype_metrics(pred_props, true_props_df, cell_types):
    """Compute per-cell-type metrics.

    Returns list of dicts with: cell_type, pearson, auprc, precision, recall, f1,
    rmse, jsd_contribution, mean_abundance, category.
    """
    pred_df = pd.DataFrame(pred_props, columns=cell_types)
    common_types = sorted(set(pred_df.columns) & set(true_props_df.columns))

    rows = []
    for ct in common_types:
        pred = pred_df[ct].values.astype(float)
        true = true_props_df[ct].values.astype(float)
        mean_abundance = true.mean()

        # Pearson
        if true.std() > 0 and pred.std() > 0:
            r, _ = pearsonr(pred, true)
        else:
            r = np.nan

        # RMSE
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        # Binary classification: present (>threshold) vs absent
        true_bin = (true > PRESENCE_THRESHOLD).astype(int)
        pred_clipped = np.clip(pred, 0, 1)

        # AUPRC
        if true_bin.sum() > 0 and true_bin.sum() < len(true_bin):
            auprc = average_precision_score(true_bin, pred_clipped)
        else:
            auprc = np.nan

        # Precision, Recall, F1 at the threshold
        pred_bin = (pred_clipped > PRESENCE_THRESHOLD).astype(int)
        if true_bin.sum() > 0:
            prec = precision_score(true_bin, pred_bin, zero_division=0)
            rec = recall_score(true_bin, pred_bin, zero_division=0)
            f1 = f1_score(true_bin, pred_bin, zero_division=0)
        else:
            prec = np.nan
            rec = np.nan
            f1 = np.nan

        # Abundance category
        if mean_abundance < 0.05:
            category = "rare"
        elif mean_abundance < 0.15:
            category = "moderate"
        else:
            category = "abundant"

        rows.append({
            "cell_type": ct,
            "mean_abundance": mean_abundance,
            "category": category,
            "pearson": r,
            "auprc": auprc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "rmse": rmse,
        })

    return rows


def compute_aggregate_metrics(pred_props, true_props_df, cell_types):
    """Compute aggregate metrics matching Spotless format.

    Returns dict with: pearson, auprc, jsd, rmse, accuracy, balanced_accuracy,
    sensitivity, specificity, precision, f1.
    """
    pred_df = pd.DataFrame(pred_props, columns=cell_types)
    common_types = sorted(set(pred_df.columns) & set(true_props_df.columns))

    pred = pred_df[common_types].values.astype(float)
    true = true_props_df[common_types].values.astype(float)

    n_spots, n_types = pred.shape

    # Pearson (flattened)
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    if true_flat.std() > 0 and pred_flat.std() > 0:
        corr, _ = pearsonr(pred_flat, true_flat)
    else:
        corr = np.nan

    # RMSE
    rmse_per_spot = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    rmse = np.mean(rmse_per_spot)

    # JSD
    eps = 1e-10
    pred_norm = np.clip(pred, 0, None) + eps
    pred_norm = pred_norm / pred_norm.sum(axis=1, keepdims=True)
    true_norm = true + eps
    true_norm = true_norm / true_norm.sum(axis=1, keepdims=True)
    jsd_vals = [jensenshannon(pred_norm[i], true_norm[i]) ** 2 for i in range(n_spots)]
    jsd = np.mean(jsd_vals)

    # Per-cell-type JSD contribution
    m = 0.5 * (pred_norm + true_norm)
    def kl(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    jsd_contrib = {}
    for j, ct in enumerate(common_types):
        contrib = 0
        for i in range(n_spots):
            p_i, t_i, m_i = pred_norm[i, j], true_norm[i, j], m[i, j]
            if t_i > eps:
                contrib += 0.5 * t_i * np.log(t_i / m_i)
            if p_i > eps:
                contrib += 0.5 * p_i * np.log(p_i / m_i)
        jsd_contrib[ct] = contrib / n_spots

    # AUPR (micro-averaged)
    true_bin = (true > PRESENCE_THRESHOLD).astype(int).flatten()
    pred_clipped = np.clip(pred, 0, 1).flatten()
    if len(np.unique(true_bin)) >= 2:
        aupr = average_precision_score(true_bin, pred_clipped)
    else:
        aupr = np.nan

    # Binary classification metrics at threshold
    pred_bin = (np.clip(pred, 0, 1) > PRESENCE_THRESHOLD).astype(int).flatten()
    if len(np.unique(true_bin)) >= 2:
        prec = precision_score(true_bin, pred_bin, zero_division=0)
        rec = recall_score(true_bin, pred_bin, zero_division=0)
        f1 = f1_score(true_bin, pred_bin, zero_division=0)

        # Specificity
        tn = np.sum((true_bin == 0) & (pred_bin == 0))
        fp = np.sum((true_bin == 0) & (pred_bin == 1))
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        # Accuracy
        acc = np.mean(true_bin == pred_bin)

        # Balanced accuracy
        sens = rec
        bal_acc = 0.5 * (sens + spec)
    else:
        prec = rec = f1 = spec = acc = bal_acc = np.nan

    return {
        "corr": corr,
        "rmse": rmse,
        "jsd": jsd,
        "aupr": aupr,
        "precision": prec,
        "sensitivity": rec,
        "f1": f1,
        "specificity": spec,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "jsd_contributions": jsd_contrib,
    }


# ---------------------------------------------------------------------------
# Silver Standard Evaluation
# ---------------------------------------------------------------------------

def evaluate_silver_standard():
    """Run comprehensive evaluation on all Silver Standard datasets."""
    all_per_ct = []
    all_aggregate = []

    for tid in sorted(TISSUE_NAMES.keys()):
        tissue = TISSUE_NAMES[tid]
        print(f"\n{'='*60}")
        print(f"Tissue {tid}: {tissue}")
        print(f"{'='*60}")

        # Load reference
        try:
            X, cell_types, ref_genes = load_reference_data(tid)
        except Exception as e:
            print(f"  Error loading reference: {e}")
            continue

        for pid in sorted(PATTERN_MAP.keys()):
            pattern = PATTERN_MAP[pid]
            print(f"\n  Pattern {pid}: {pattern}")

            # Load spatial data
            try:
                Y, sp_genes, true_props = load_silver_data(tid, pid)
            except Exception as e:
                print(f"    Skipping: {e}")
                continue

            Y_a, X_a, common_genes = align_genes(Y, sp_genes, X, ref_genes)
            true_cols = [ct for ct in cell_types if ct in true_props.columns]
            true_sub = true_props[true_cols]
            n_spots = Y_a.shape[0]
            print(f"    {n_spots} spots, {len(common_genes)} genes, {len(true_cols)} cell types")

            # --- FlashDeconv ---
            coords = np.array(
                [[i % int(np.ceil(np.sqrt(n_spots))),
                  i // int(np.ceil(np.sqrt(n_spots)))]
                 for i in range(n_spots)], dtype=float
            )

            t0 = time.time()
            model = FlashDeconv(
                sketch_dim=min(512, Y_a.shape[1]),
                lambda_spatial="auto",
                preprocess="log_cpm",
                n_hvg=2000,
                rho_sparsity=0,
                max_iter=200,
                verbose=False,
                random_state=RANDOM_STATE,
            )
            fd_pred = model.fit_transform(Y_a, X_a, coords)
            fd_time = time.time() - t0

            # Per-cell-type metrics
            fd_ct = compute_per_celltype_metrics(fd_pred, true_sub, cell_types)
            for row in fd_ct:
                row.update({"method": "FlashDeconv", "tissue": tissue,
                           "tissue_id": tid, "pattern": pattern, "pattern_id": pid,
                           "benchmark": "silver_standard"})
            all_per_ct.extend(fd_ct)

            # Aggregate metrics
            fd_agg = compute_aggregate_metrics(fd_pred, true_sub, cell_types)
            fd_agg_row = {k: v for k, v in fd_agg.items() if k != "jsd_contributions"}
            fd_agg_row.update({"method": "FlashDeconv", "tissue": tissue,
                              "tissue_id": tid, "pattern": pattern, "pattern_id": pid,
                              "benchmark": "silver_standard", "time_s": fd_time})
            all_aggregate.append(fd_agg_row)

            print(f"    FlashDeconv: {fd_time:.2f}s, Pearson={fd_agg['corr']:.3f}, "
                  f"AUPR={fd_agg['aupr']:.3f}, JSD={fd_agg['jsd']:.4f}")

            # --- Spotless competing methods ---
            for method in SPOTLESS_METHODS:
                pred_data, pred_types = load_spotless_prediction(
                    tissue, pattern, method, rep=1
                )
                if pred_data is None:
                    continue

                # Align cell types
                type_map = align_celltype_names(pred_types, list(true_sub.columns))
                if not type_map:
                    continue

                # Reorder prediction columns to match ground truth
                matched_pred_types = []
                matched_true_types = []
                pred_cols = []
                for j, pt in enumerate(pred_types):
                    if pt in type_map:
                        matched_pred_types.append(pt)
                        matched_true_types.append(type_map[pt])
                        pred_cols.append(j)

                if not pred_cols:
                    continue

                pred_aligned = pred_data[:, pred_cols]
                # Clip negatives and normalize
                pred_aligned = np.clip(pred_aligned, 0, None)
                row_sums = pred_aligned.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                pred_aligned = pred_aligned / row_sums

                # Verify spot count
                if pred_aligned.shape[0] != len(true_sub):
                    continue

                true_aligned = true_sub[matched_true_types]

                # Per-cell-type metrics
                method_ct = compute_per_celltype_metrics(
                    pred_aligned, true_aligned, matched_true_types
                )
                for row in method_ct:
                    row.update({"method": method, "tissue": tissue,
                               "tissue_id": tid, "pattern": pattern, "pattern_id": pid,
                               "benchmark": "silver_standard"})
                all_per_ct.extend(method_ct)

                # Aggregate metrics
                method_agg = compute_aggregate_metrics(
                    pred_aligned, true_aligned, matched_true_types
                )
                method_agg_row = {k: v for k, v in method_agg.items()
                                  if k != "jsd_contributions"}
                method_agg_row.update({"method": method, "tissue": tissue,
                                      "tissue_id": tid, "pattern": pattern,
                                      "pattern_id": pid, "benchmark": "silver_standard"})
                all_aggregate.append(method_agg_row)

            # Save incrementally
            if len(all_per_ct) > 0:
                pd.DataFrame(all_per_ct).to_csv(
                    os.path.join(OUTPUT_DIR, "per_celltype_all_benchmarks.csv"),
                    index=False
                )
            if len(all_aggregate) > 0:
                pd.DataFrame(all_aggregate).to_csv(
                    os.path.join(OUTPUT_DIR, "aggregate_all_benchmarks.csv"),
                    index=False
                )

    return all_per_ct, all_aggregate


# ---------------------------------------------------------------------------
# Liver evaluation
# ---------------------------------------------------------------------------

def evaluate_liver():
    """Evaluate on liver Visium slides with unified metrics."""
    all_per_ct = []
    all_aggregate = []

    liver_samples = ["JB01", "JB02", "JB03", "JB04"]

    for sample in liver_samples:
        dataset_name = f"liver_mouseVisium_{sample}"
        pred_dir = os.path.join(SPOTLESS_DIR, dataset_name)

        if not os.path.exists(pred_dir):
            print(f"  Liver {sample}: prediction dir not found, skipping")
            continue

        # Load ground truth from any method prediction to get spot count
        # Then load our liver data
        print(f"\n  Liver {sample}:")

        # Check for converted liver data
        liver_prefix = os.path.join(DATA_DIR, f"liver_{sample}")
        counts_file = f"{liver_prefix}_counts.mtx"
        if not os.path.exists(counts_file):
            print(f"    No converted data found for {sample}")
            continue

        Y = scipy.io.mmread(counts_file).toarray().T
        genes = np.loadtxt(f"{liver_prefix}_genes.txt", dtype=str)

        # Load liver reference
        ref_prefix = os.path.join(DATA_DIR, "reference_liver")
        if not os.path.exists(f"{ref_prefix}_counts.mtx"):
            print("    No liver reference found")
            continue

        ref_counts = scipy.io.mmread(f"{ref_prefix}_counts.mtx").toarray().T
        ref_celltypes = np.loadtxt(f"{ref_prefix}_celltypes.txt", dtype=str)
        ref_genes = np.loadtxt(f"{ref_prefix}_genes.txt", dtype=str)
        unique_types = np.unique(ref_celltypes)
        X = np.zeros((len(unique_types), ref_counts.shape[1]))
        for i, ct in enumerate(unique_types):
            X[i] = ref_counts[ref_celltypes == ct].mean(axis=0)

        # Load ground truth proportions
        props_file = f"{liver_prefix}_proportions.csv"
        if not os.path.exists(props_file):
            print(f"    No proportions file for {sample}")
            continue

        true_props = pd.read_csv(props_file, index_col=0)

        # Align
        Y_a, X_a, common_genes = align_genes(Y, genes, X, ref_genes)
        true_cols = [ct for ct in unique_types if ct in true_props.columns]
        true_sub = true_props[true_cols]

        print(f"    {Y_a.shape[0]} spots, {len(common_genes)} genes, {len(true_cols)} types")

        # FlashDeconv
        coords = np.array(
            [[i % int(np.ceil(np.sqrt(Y_a.shape[0]))),
              i // int(np.ceil(np.sqrt(Y_a.shape[0])))]
             for i in range(Y_a.shape[0])], dtype=float
        )

        t0 = time.time()
        model = FlashDeconv(
            sketch_dim=min(512, Y_a.shape[1]),
            lambda_spatial="auto",
            preprocess="log_cpm",
            n_hvg=2000,
            rho_sparsity=0,
            max_iter=200,
            verbose=False,
            random_state=RANDOM_STATE,
        )
        fd_pred = model.fit_transform(Y_a, X_a, coords)
        fd_time = time.time() - t0

        fd_ct = compute_per_celltype_metrics(fd_pred, true_sub, unique_types)
        for row in fd_ct:
            row.update({"method": "FlashDeconv", "tissue": f"liver_{sample}",
                       "tissue_id": -1, "pattern": "real", "pattern_id": -1,
                       "benchmark": "liver"})
        all_per_ct.extend(fd_ct)

        fd_agg = compute_aggregate_metrics(fd_pred, true_sub, unique_types)
        fd_agg_row = {k: v for k, v in fd_agg.items() if k != "jsd_contributions"}
        fd_agg_row.update({"method": "FlashDeconv", "tissue": f"liver_{sample}",
                          "tissue_id": -1, "pattern": "real", "pattern_id": -1,
                          "benchmark": "liver", "time_s": fd_time})
        all_aggregate.append(fd_agg_row)

        print(f"    FlashDeconv: Pearson={fd_agg['corr']:.3f}, AUPR={fd_agg['aupr']:.3f}, "
              f"JSD={fd_agg['jsd']:.4f}")

        # Spotless competing methods for liver
        for method in SPOTLESS_METHODS:
            pred_data, pred_types = load_spotless_prediction(
                "liver_mouseVisium", sample, method, rep=1
            )
            if pred_data is None:
                # Try alternative naming
                pred_file = os.path.join(
                    pred_dir,
                    f"proportions_{method}_{dataset_name}"
                )
                if os.path.exists(pred_file):
                    with open(pred_file, 'r') as f:
                        lines = f.readlines()
                    pred_types = lines[0].strip().split('\t')
                    pred_data = np.array([[float(x) for x in l.strip().split('\t')]
                                         for l in lines[1:]])
                else:
                    continue

            type_map = align_celltype_names(pred_types, list(true_sub.columns))
            if not type_map:
                continue

            matched_true_types = []
            pred_cols = []
            for j, pt in enumerate(pred_types):
                if pt in type_map:
                    matched_true_types.append(type_map[pt])
                    pred_cols.append(j)

            if not pred_cols or pred_data.shape[0] != len(true_sub):
                continue

            pred_aligned = np.clip(pred_data[:, pred_cols], 0, None)
            rs = pred_aligned.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            pred_aligned = pred_aligned / rs
            true_aligned = true_sub[matched_true_types]

            method_ct = compute_per_celltype_metrics(pred_aligned, true_aligned, matched_true_types)
            for row in method_ct:
                row.update({"method": method, "tissue": f"liver_{sample}",
                           "tissue_id": -1, "pattern": "real", "pattern_id": -1,
                           "benchmark": "liver"})
            all_per_ct.extend(method_ct)

            method_agg = compute_aggregate_metrics(pred_aligned, true_aligned, matched_true_types)
            method_agg_row = {k: v for k, v in method_agg.items() if k != "jsd_contributions"}
            method_agg_row.update({"method": method, "tissue": f"liver_{sample}",
                                  "tissue_id": -1, "pattern": "real", "pattern_id": -1,
                                  "benchmark": "liver"})
            all_aggregate.append(method_agg_row)

    return all_per_ct, all_aggregate


# ---------------------------------------------------------------------------
# Gold Standard (SeqFISH+, STARMap) evaluation
# ---------------------------------------------------------------------------

def evaluate_gold_standard():
    """Evaluate on SeqFISH+ and STARMap datasets."""
    all_per_ct = []
    all_aggregate = []

    # SeqFISH+ (Eng2019) - cortex_svz and ob, fov 0-6
    for region in ["cortex_svz", "ob"]:
        for fov in range(7):
            dataset = f"Eng2019_{region}"
            fov_name = f"fov{fov}"

            # Check for converted data
            prefix = os.path.join(DATA_DIR, f"Eng2019_{region}_{fov_name}")
            props_file = f"{prefix}_proportions.csv"
            counts_file = f"{prefix}_counts.mtx"

            if not os.path.exists(props_file) or not os.path.exists(counts_file):
                continue

            Y = scipy.io.mmread(counts_file).toarray().T
            genes = np.loadtxt(f"{prefix}_genes.txt", dtype=str)
            true_props = pd.read_csv(props_file, index_col=0)

            # Load reference
            ref_prefix = os.path.join(DATA_DIR, f"reference_Eng2019_{region}")
            if not os.path.exists(f"{ref_prefix}_counts.mtx"):
                continue

            ref_counts = scipy.io.mmread(f"{ref_prefix}_counts.mtx").toarray().T
            ref_celltypes = np.loadtxt(f"{ref_prefix}_celltypes.txt", dtype=str)
            ref_genes = np.loadtxt(f"{ref_prefix}_genes.txt", dtype=str)
            unique_types = np.unique(ref_celltypes)
            X = np.zeros((len(unique_types), ref_counts.shape[1]))
            for i, ct in enumerate(unique_types):
                X[i] = ref_counts[ref_celltypes == ct].mean(axis=0)

            Y_a, X_a, common_genes = align_genes(Y, genes, X, ref_genes)
            true_cols = [ct for ct in unique_types if ct in true_props.columns]
            if not true_cols:
                continue
            true_sub = true_props[true_cols]

            if Y_a.shape[0] != len(true_sub):
                continue

            print(f"  {dataset} {fov_name}: {Y_a.shape[0]} spots, {len(common_genes)} genes")

            # FlashDeconv
            coords = np.array(
                [[i % int(np.ceil(np.sqrt(Y_a.shape[0]))),
                  i // int(np.ceil(np.sqrt(Y_a.shape[0])))]
                 for i in range(Y_a.shape[0])], dtype=float
            )

            model = FlashDeconv(
                sketch_dim=min(512, Y_a.shape[1]),
                lambda_spatial="auto",
                preprocess="log_cpm",
                n_hvg=min(2000, Y_a.shape[1]),
                rho_sparsity=0,
                max_iter=200,
                verbose=False,
                random_state=RANDOM_STATE,
            )
            fd_pred = model.fit_transform(Y_a, X_a, coords)

            fd_ct = compute_per_celltype_metrics(fd_pred, true_sub, unique_types)
            for row in fd_ct:
                row.update({"method": "FlashDeconv", "tissue": f"{region}_{fov_name}",
                           "tissue_id": -1, "pattern": "gold", "pattern_id": -1,
                           "benchmark": f"seqfish_{region}"})
            all_per_ct.extend(fd_ct)

            fd_agg = compute_aggregate_metrics(fd_pred, true_sub, unique_types)
            fd_agg_row = {k: v for k, v in fd_agg.items() if k != "jsd_contributions"}
            fd_agg_row.update({"method": "FlashDeconv", "tissue": f"{region}_{fov_name}",
                              "tissue_id": -1, "pattern": "gold", "pattern_id": -1,
                              "benchmark": f"seqfish_{region}"})
            all_aggregate.append(fd_agg_row)

            print(f"    FlashDeconv: Pearson={fd_agg['corr']:.3f}, AUPR={fd_agg['aupr']:.3f}, "
                  f"JSD={fd_agg['jsd']:.4f}")

            # Spotless methods
            for method in SPOTLESS_METHODS:
                pred_file = os.path.join(
                    SPOTLESS_DIR, dataset,
                    f"proportions_{method}_{dataset}_{fov_name}"
                )
                if not os.path.exists(pred_file):
                    continue

                with open(pred_file, 'r') as f:
                    lines = f.readlines()
                pred_types = lines[0].strip().split('\t')
                pred_data = np.array([[float(x) for x in l.strip().split('\t')]
                                      for l in lines[1:]])

                type_map = align_celltype_names(pred_types, list(true_sub.columns))
                if not type_map:
                    continue

                matched_true_types = []
                pred_cols = []
                for j, pt in enumerate(pred_types):
                    if pt in type_map:
                        matched_true_types.append(type_map[pt])
                        pred_cols.append(j)

                if not pred_cols or pred_data.shape[0] != len(true_sub):
                    continue

                pred_aligned = np.clip(pred_data[:, pred_cols], 0, None)
                rs = pred_aligned.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1
                pred_aligned = pred_aligned / rs
                true_aligned = true_sub[matched_true_types]

                method_ct = compute_per_celltype_metrics(
                    pred_aligned, true_aligned, matched_true_types
                )
                for row in method_ct:
                    row.update({"method": method, "tissue": f"{region}_{fov_name}",
                               "tissue_id": -1, "pattern": "gold", "pattern_id": -1,
                               "benchmark": f"seqfish_{region}"})
                all_per_ct.extend(method_ct)

                method_agg = compute_aggregate_metrics(
                    pred_aligned, true_aligned, matched_true_types
                )
                method_agg_row = {k: v for k, v in method_agg.items()
                                  if k != "jsd_contributions"}
                method_agg_row.update({"method": method, "tissue": f"{region}_{fov_name}",
                                      "tissue_id": -1, "pattern": "gold", "pattern_id": -1,
                                      "benchmark": f"seqfish_{region}"})
                all_aggregate.append(method_agg_row)

    # STARMap (Wang2018)
    prefix = os.path.join(DATA_DIR, "Wang2018_visp_rep0410")
    if os.path.exists(f"{prefix}_proportions.csv") and os.path.exists(f"{prefix}_counts.mtx"):
        Y = scipy.io.mmread(f"{prefix}_counts.mtx").toarray().T
        genes = np.loadtxt(f"{prefix}_genes.txt", dtype=str)
        true_props = pd.read_csv(f"{prefix}_proportions.csv", index_col=0)

        ref_prefix = os.path.join(DATA_DIR, "reference_Wang2018_visp")
        if os.path.exists(f"{ref_prefix}_counts.mtx"):
            ref_counts = scipy.io.mmread(f"{ref_prefix}_counts.mtx").toarray().T
            ref_celltypes = np.loadtxt(f"{ref_prefix}_celltypes.txt", dtype=str)
            ref_genes = np.loadtxt(f"{ref_prefix}_genes.txt", dtype=str)
            unique_types = np.unique(ref_celltypes)
            X = np.zeros((len(unique_types), ref_counts.shape[1]))
            for i, ct in enumerate(unique_types):
                X[i] = ref_counts[ref_celltypes == ct].mean(axis=0)

            Y_a, X_a, common_genes = align_genes(Y, genes, X, ref_genes)
            true_cols = [ct for ct in unique_types if ct in true_props.columns]
            true_sub = true_props[true_cols]

            print(f"\n  STARMap: {Y_a.shape[0]} spots, {len(common_genes)} genes")

            coords = np.array(
                [[i % int(np.ceil(np.sqrt(Y_a.shape[0]))),
                  i // int(np.ceil(np.sqrt(Y_a.shape[0])))]
                 for i in range(Y_a.shape[0])], dtype=float
            )

            model = FlashDeconv(
                sketch_dim=min(512, Y_a.shape[1]),
                lambda_spatial="auto",
                preprocess="log_cpm",
                n_hvg=min(2000, Y_a.shape[1]),
                rho_sparsity=0,
                max_iter=200,
                verbose=False,
                random_state=RANDOM_STATE,
            )
            fd_pred = model.fit_transform(Y_a, X_a, coords)

            fd_ct = compute_per_celltype_metrics(fd_pred, true_sub, unique_types)
            for row in fd_ct:
                row.update({"method": "FlashDeconv", "tissue": "starmap",
                           "tissue_id": -1, "pattern": "gold", "pattern_id": -1,
                           "benchmark": "starmap"})
            all_per_ct.extend(fd_ct)

            fd_agg = compute_aggregate_metrics(fd_pred, true_sub, unique_types)
            fd_agg_row = {k: v for k, v in fd_agg.items() if k != "jsd_contributions"}
            fd_agg_row.update({"method": "FlashDeconv", "tissue": "starmap",
                              "tissue_id": -1, "pattern": "gold", "pattern_id": -1,
                              "benchmark": "starmap"})
            all_aggregate.append(fd_agg_row)

            print(f"    FlashDeconv: Pearson={fd_agg['corr']:.3f}, AUPR={fd_agg['aupr']:.3f}, "
                  f"JSD={fd_agg['jsd']:.4f}")

            # Spotless methods for STARMap
            starmap_dir = os.path.join(SPOTLESS_DIR, "Wang2018_visp")
            if os.path.exists(starmap_dir):
                for method in SPOTLESS_METHODS:
                    pred_file = os.path.join(
                        starmap_dir,
                        f"proportions_{method}_Wang2018_visp_rep0410"
                    )
                    if not os.path.exists(pred_file):
                        continue

                    with open(pred_file, 'r') as f:
                        lines = f.readlines()
                    pred_types = lines[0].strip().split('\t')
                    pred_data = np.array([[float(x) for x in l.strip().split('\t')]
                                          for l in lines[1:]])

                    type_map = align_celltype_names(pred_types, list(true_sub.columns))
                    if not type_map:
                        continue

                    matched_true_types = []
                    pred_cols = []
                    for j, pt in enumerate(pred_types):
                        if pt in type_map:
                            matched_true_types.append(type_map[pt])
                            pred_cols.append(j)

                    if not pred_cols or pred_data.shape[0] != len(true_sub):
                        continue

                    pred_aligned = np.clip(pred_data[:, pred_cols], 0, None)
                    rs = pred_aligned.sum(axis=1, keepdims=True)
                    rs[rs == 0] = 1
                    pred_aligned = pred_aligned / rs
                    true_aligned = true_sub[matched_true_types]

                    method_ct = compute_per_celltype_metrics(
                        pred_aligned, true_aligned, matched_true_types
                    )
                    for row in method_ct:
                        row.update({"method": method, "tissue": "starmap",
                                   "tissue_id": -1, "pattern": "gold", "pattern_id": -1,
                                   "benchmark": "starmap"})
                    all_per_ct.extend(method_ct)

                    method_agg = compute_aggregate_metrics(
                        pred_aligned, true_aligned, matched_true_types
                    )
                    method_agg_row = {k: v for k, v in method_agg.items()
                                      if k != "jsd_contributions"}
                    method_agg_row.update({"method": method, "tissue": "starmap",
                                          "tissue_id": -1, "pattern": "gold",
                                          "pattern_id": -1, "benchmark": "starmap"})
                    all_aggregate.append(method_agg_row)

    return all_per_ct, all_aggregate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE PER-CELL-TYPE EVALUATION")
    print("=" * 70)

    # Silver Standard
    print("\n\n### SILVER STANDARD ###")
    ss_ct, ss_agg = evaluate_silver_standard()

    # Liver
    print("\n\n### LIVER ###")
    liv_ct, liv_agg = evaluate_liver()

    # Gold Standard
    print("\n\n### GOLD STANDARD (SeqFISH+ / STARMap) ###")
    gs_ct, gs_agg = evaluate_gold_standard()

    # Combine all results
    all_ct = ss_ct + liv_ct + gs_ct
    all_agg = ss_agg + liv_agg + gs_agg

    ct_df = pd.DataFrame(all_ct)
    agg_df = pd.DataFrame(all_agg)

    ct_df.to_csv(os.path.join(OUTPUT_DIR, "per_celltype_all_benchmarks.csv"), index=False)
    agg_df.to_csv(os.path.join(OUTPUT_DIR, "aggregate_all_benchmarks.csv"), index=False)

    # Summary by abundance category
    print("\n\n" + "=" * 70)
    print("SUMMARY: FlashDeconv vs Top Methods by Abundance Category")
    print("=" * 70)

    for benchmark in ct_df["benchmark"].unique():
        bench_df = ct_df[ct_df["benchmark"] == benchmark]
        print(f"\n  Benchmark: {benchmark}")
        for method in ["FlashDeconv", "rctd", "cell2location"]:
            sub = bench_df[bench_df["method"] == method]
            if len(sub) == 0:
                continue
            print(f"\n    {method}:")
            for cat in ["rare", "moderate", "abundant"]:
                cat_sub = sub[sub["category"] == cat]
                n = len(cat_sub)
                if n == 0:
                    continue
                print(f"      {cat:12s} (n={n:3d}): "
                      f"Pearson={cat_sub['pearson'].mean():.3f}  "
                      f"AUPRC={cat_sub['auprc'].mean():.3f}  "
                      f"Prec={cat_sub['precision'].mean():.3f}  "
                      f"Recall={cat_sub['recall'].mean():.3f}  "
                      f"F1={cat_sub['f1'].mean():.3f}")

    # Unified aggregate summary
    print("\n\n" + "=" * 70)
    print("UNIFIED AGGREGATE METRICS (Pearson + AUPR + JSD)")
    print("=" * 70)

    for benchmark in agg_df["benchmark"].unique():
        bench_df = agg_df[agg_df["benchmark"] == benchmark]
        print(f"\n  Benchmark: {benchmark}")
        summary = bench_df.groupby("method").agg({
            "corr": "mean", "aupr": "mean", "jsd": "mean", "rmse": "mean",
            "precision": "mean", "sensitivity": "mean", "f1": "mean"
        }).sort_values("corr", ascending=False)

        print(f"    {'Method':<18s} {'Pearson':>8s} {'AUPR':>8s} {'JSD':>8s} "
              f"{'RMSE':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
        print(f"    {'-'*82}")
        for method, row in summary.iterrows():
            print(f"    {method:<18s} {row['corr']:8.4f} {row['aupr']:8.4f} "
                  f"{row['jsd']:8.4f} {row['rmse']:8.4f} {row['precision']:8.4f} "
                  f"{row['sensitivity']:8.4f} {row['f1']:8.4f}")

    # Per-cell-type summary for FlashDeconv
    fd_ct = ct_df[ct_df["method"] == "FlashDeconv"]
    summary_by_cat = fd_ct.groupby("category").agg({
        "pearson": ["mean", "std", "count"],
        "auprc": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"],
        "rmse": ["mean", "std"],
    })
    summary_by_cat.to_csv(
        os.path.join(OUTPUT_DIR, "per_celltype_summary_by_abundance.csv")
    )

    print(f"\n\nResults saved to {OUTPUT_DIR}/")
    print(f"  - per_celltype_all_benchmarks.csv ({len(ct_df)} rows)")
    print(f"  - aggregate_all_benchmarks.csv ({len(agg_df)} rows)")
    print(f"  - per_celltype_summary_by_abundance.csv")


if __name__ == "__main__":
    main()
