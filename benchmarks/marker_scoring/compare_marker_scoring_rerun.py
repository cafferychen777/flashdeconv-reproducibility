"""
Compare FlashDeconv vs marker gene scoring on Spotless Silver Standards.

Addresses Reviewer 2's question about marker gene scoring (Pritykin et al., 2026)
being competitive with deconvolution. Tests in the correct setting: simulated
multi-cell Visium spots (not single-cell Xenium).

Two marker selection methods are compared as a sensitivity analysis:
  A) Max-gap: top-N genes by max-minus-second-max mean expression gap on the
     aggregated reference signature matrix (deterministic, O(K*G)).
  B) Wilcoxon: top-N upregulated genes by Wilcoxon rank-sum test (one-vs-rest)
     on individual single cells, following the standard Scanpy/Pritykin protocol.

Scoring protocol is identical for both: mean normalized expression of marker
genes minus mean of a size-matched control gene set (Scanpy sc.tl.score_genes).

Output: CSV with per-cell-type Pearson, AUPRC, RMSE for FlashDeconv vs marker
scoring (both selection methods), stratified by abundance category.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import pearsonr, ranksums
from sklearn.metrics import precision_recall_curve, auc
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/apple/Research/FlashDeconv")
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks")
import fdfinal  # noqa: F401  (fit log / FD_PROTOCOL)
import seedpatch  # noqa: F401  (FD_SEED)
from flashdeconv import FlashDeconv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/apple/Research/FlashDeconv/validation/benchmark_data/converted"
OUTPUT_DIR = os.environ["RERUN_OUT"]
TISSUE_IDS = [1, 2, 3, 4, 5, 6]
TISSUE_NAMES = {
    1: "brain_cortex", 2: "cerebellum_cell", 3: "cerebellum_nucleus",
    4: "hippocampus", 5: "kidney", 6: "scc_p5",
}
N_PATTERNS = {1: 11, 2: 9, 3: 9, 4: 9, 5: 9, 6: 9}
N_MARKER_GENES = 50
PRESENCE_THRESHOLD = 0.01
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_silver_data(tissue_id, sample_id):
    prefix = f"silver_{tissue_id}_{sample_id}"
    Y = scipy.io.mmread(os.path.join(DATA_DIR, f"{prefix}_counts.mtx")).toarray().T
    genes = np.loadtxt(os.path.join(DATA_DIR, f"{prefix}_genes.txt"), dtype=str)
    props = pd.read_csv(
        os.path.join(DATA_DIR, f"{prefix}_proportions.csv"), index_col=0
    )
    return Y, genes, props


def load_reference_data(tissue_id):
    prefix = f"reference_{tissue_id}"
    ref_counts = scipy.io.mmread(
        os.path.join(DATA_DIR, f"{prefix}_counts.mtx")
    ).toarray().T
    ref_celltypes = np.loadtxt(
        os.path.join(DATA_DIR, f"{prefix}_celltypes.txt"), dtype=str
    )
    ref_genes = np.loadtxt(os.path.join(DATA_DIR, f"{prefix}_genes.txt"), dtype=str)
    unique_types = np.unique(ref_celltypes)
    # Mean expression per cell type
    X = np.zeros((len(unique_types), ref_counts.shape[1]))
    for i, ct in enumerate(unique_types):
        X[i] = ref_counts[ref_celltypes == ct].mean(axis=0)
    return X, unique_types, ref_genes, ref_counts, ref_celltypes


def align_genes(Y, genes_Y, X, genes_X):
    common = np.intersect1d(genes_Y, genes_X)
    idx_Y = np.array([np.where(genes_Y == g)[0][0] for g in common])
    idx_X = np.array([np.where(genes_X == g)[0][0] for g in common])
    return Y[:, idx_Y], X[:, idx_X], common, idx_X


# ---------------------------------------------------------------------------
# Marker gene scoring
# ---------------------------------------------------------------------------

def select_marker_genes(X, cell_types, n_markers=50):
    """Select top marker genes per cell type using max-minus-second-max gap."""
    K, G = X.shape
    # Log-CPM normalize reference
    row_sums = X.sum(axis=1, keepdims=True) + 1e-10
    X_norm = np.log1p(1e4 * X / row_sums)

    markers = {}
    for k in range(K):
        gaps = np.zeros(G)
        for g in range(G):
            vals = X_norm[:, g].copy()
            this_val = vals[k]
            vals[k] = -np.inf
            second_max = vals.max()
            gaps[g] = this_val - second_max
        top_idx = np.argsort(gaps)[::-1][:n_markers]
        markers[cell_types[k]] = top_idx
    return markers


def select_marker_genes_wilcoxon(ref_counts, ref_celltypes, cell_types,
                                 n_markers=50):
    """Select top marker genes via Wilcoxon rank-sum test (one-vs-rest).

    For each cell type, perform a one-vs-rest Wilcoxon rank-sum test on
    log-CPM-normalized single-cell expression, then select the top N
    significantly upregulated genes. This matches the standard Scanpy
    rank_genes_groups / Pritykin et al. protocol.
    """
    # Log-CPM normalize individual cells
    row_sums = ref_counts.sum(axis=1, keepdims=True) + 1e-10
    ref_norm = np.log1p(1e4 * ref_counts / row_sums)

    G = ref_counts.shape[1]
    markers = {}

    for ct in cell_types:
        mask = ref_celltypes == ct
        in_group = ref_norm[mask]
        out_group = ref_norm[~mask]

        log_fc = in_group.mean(axis=0) - out_group.mean(axis=0)
        pvals = np.ones(G)

        for g in range(G):
            in_vals = in_group[:, g]
            out_vals = out_group[:, g]
            if in_vals.std() > 0 or out_vals.std() > 0:
                _, pvals[g] = ranksums(in_vals, out_vals)

        # Rank by significance among upregulated genes
        scores = np.where(log_fc > 0, -np.log10(pvals + 1e-300), -np.inf)
        top_idx = np.argsort(scores)[::-1][:n_markers]
        markers[ct] = top_idx

    return markers


def marker_gene_score(Y, markers, cell_types):
    """Score each spot for each cell type using mean expression of markers.

    Follows Scanpy sc.tl.score_genes logic: score = mean(marker_genes) - mean(control).
    Simplified version: just mean normalized expression of marker genes.
    """
    # Log-CPM normalize spatial data
    row_sums = Y.sum(axis=1, keepdims=True) + 1e-10
    Y_norm = np.log1p(1e4 * Y / row_sums)

    n_spots = Y.shape[0]
    K = len(cell_types)
    scores = np.zeros((n_spots, K))

    # Control genes: random set for background subtraction (Scanpy approach)
    rng = np.random.RandomState(RANDOM_STATE)
    all_marker_idx = set()
    for idx_set in markers.values():
        all_marker_idx.update(idx_set)
    non_marker_genes = np.array([g for g in range(Y.shape[1]) if g not in all_marker_idx])

    for k, ct in enumerate(cell_types):
        marker_idx = markers[ct]
        # Background: sample same number of non-marker genes
        n_ctrl = min(len(marker_idx), len(non_marker_genes))
        ctrl_idx = rng.choice(non_marker_genes, n_ctrl, replace=False)
        scores[:, k] = Y_norm[:, marker_idx].mean(axis=1) - Y_norm[:, ctrl_idx].mean(axis=1)

    # Clip negative scores to 0 and normalize to proportions (softmax-like)
    scores = np.maximum(scores, 0)
    row_sums = scores.sum(axis=1, keepdims=True) + 1e-10
    props = scores / row_sums
    return props, scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_per_celltype(pred_props, true_props_df, cell_types):
    pred_df = pd.DataFrame(pred_props, columns=cell_types)
    common_types = sorted(set(pred_df.columns) & set(true_props_df.columns))
    rows = []
    for ct in common_types:
        pred = pred_df[ct].values
        true = true_props_df[ct].values
        mean_abundance = true.mean()

        if true.std() > 0 and pred.std() > 0:
            r, _ = pearsonr(pred, true)
        else:
            r = np.nan

        rmse = np.sqrt(np.mean((pred - true) ** 2))

        true_bin = (true > PRESENCE_THRESHOLD).astype(int)
        if true_bin.sum() > 0 and true_bin.sum() < len(true_bin):
            prec_curve, rec_curve, _ = precision_recall_curve(true_bin, pred)
            auprc = auc(rec_curve, prec_curve)
        else:
            auprc = np.nan

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
            "rmse": rmse,
            "auprc": auprc,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for tid in TISSUE_IDS:
        tname = TISSUE_NAMES[tid]
        print(f"\n{'='*60}")
        print(f"Tissue {tid}: {tname}")
        print(f"{'='*60}")

        # Load reference
        X, cell_types, ref_genes, ref_counts, ref_celltypes = load_reference_data(tid)
        print(f"  Reference: {X.shape[0]} cell types, {X.shape[1]} genes")

        # Process first pattern only (consistent with ablation studies)
        sid = 1
        print(f"\n  Pattern {sid}:")
        Y, sp_genes, true_props = load_silver_data(tid, sid)
        Y_a, X_a, common_genes, idx_X = align_genes(Y, sp_genes, X, ref_genes)
        ref_counts_a = ref_counts[:, idx_X]  # align single-cell counts
        print(f"    {Y_a.shape[0]} spots, {Y_a.shape[1]} common genes, {len(cell_types)} types")

        # Align true proportions
        true_cols = [ct for ct in cell_types if ct in true_props.columns]
        true_sub = true_props[true_cols]

        # --- FlashDeconv ---
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
        print(f"    FlashDeconv: {fd_time:.2f}s")

        fd_results = evaluate_per_celltype(fd_pred, true_sub, cell_types)
        for row in fd_results:
            row["method"] = "FlashDeconv"
            row["tissue"] = tname
            row["tissue_id"] = tid
            row["time_s"] = fd_time

        # --- Marker gene scoring (max-gap selection) ---
        t0 = time.time()
        markers_gap = select_marker_genes(X_a, cell_types, n_markers=N_MARKER_GENES)
        mg_gap_props, mg_gap_scores = marker_gene_score(Y_a, markers_gap, cell_types)
        mg_gap_time = time.time() - t0
        print(f"    Marker scoring (max-gap): {mg_gap_time:.2f}s")

        mg_gap_results = evaluate_per_celltype(mg_gap_props, true_sub, cell_types)
        for row in mg_gap_results:
            row["method"] = "MarkerScoring_maxgap"
            row["tissue"] = tname
            row["tissue_id"] = tid
            row["time_s"] = mg_gap_time

        # --- Marker gene scoring (Wilcoxon rank-sum selection) ---
        t0 = time.time()
        markers_wilcox = select_marker_genes_wilcoxon(
            ref_counts_a, ref_celltypes, cell_types, n_markers=N_MARKER_GENES
        )
        mg_wlx_props, mg_wlx_scores = marker_gene_score(
            Y_a, markers_wilcox, cell_types
        )
        mg_wlx_time = time.time() - t0
        print(f"    Marker scoring (Wilcoxon): {mg_wlx_time:.2f}s")

        mg_wlx_results = evaluate_per_celltype(mg_wlx_props, true_sub, cell_types)
        for row in mg_wlx_results:
            row["method"] = "MarkerScoring_wilcoxon"
            row["tissue"] = tname
            row["tissue_id"] = tid
            row["time_s"] = mg_wlx_time

        all_results.extend(fd_results)
        all_results.extend(mg_gap_results)
        all_results.extend(mg_wlx_results)

        # Marker overlap between the two selection methods
        n_overlap = []
        for ct in cell_types:
            if ct in markers_gap and ct in markers_wilcox:
                overlap = len(set(markers_gap[ct]) & set(markers_wilcox[ct]))
                n_overlap.append(overlap)
        mean_overlap = np.mean(n_overlap) if n_overlap else 0
        print(f"    Marker overlap (max-gap vs Wilcoxon): "
              f"{mean_overlap:.1f}/{N_MARKER_GENES} genes per type")

        # Print comparison
        print(f"\n    {'Cell type':30s} {'Cat':8s} "
              f"{'FD r':>7s} {'Gap r':>7s} {'Wlx r':>7s} "
              f"{'FD AUC':>7s} {'Gap AUC':>7s} {'Wlx AUC':>7s}")
        print(f"    {'-'*90}")
        for fd_row, gap_row, wlx_row in zip(
            fd_results, mg_gap_results, mg_wlx_results
        ):
            ct = fd_row["cell_type"]
            cat = fd_row["category"]
            print(f"    {ct:30s} {cat:8s} "
                  f"{fd_row['pearson']:7.3f} {gap_row['pearson']:7.3f} "
                  f"{wlx_row['pearson']:7.3f} "
                  f"{fd_row['auprc']:7.3f} {gap_row['auprc']:7.3f} "
                  f"{wlx_row['auprc']:7.3f}")

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "marker_scoring_comparison.csv"), index=False)

    # Summary by category
    print("\n" + "=" * 60)
    print("SUMMARY BY ABUNDANCE CATEGORY")
    print("=" * 60)

    for method in ["FlashDeconv", "MarkerScoring_maxgap", "MarkerScoring_wilcoxon"]:
        sub = df[df["method"] == method]
        print(f"\n  {method}:")
        for cat in ["rare", "moderate", "abundant"]:
            cat_sub = sub[sub["category"] == cat]
            n = len(cat_sub)
            print(f"    {cat:12s} (n={n:2d}): Pearson={cat_sub['pearson'].mean():.3f}  "
                  f"AUPRC={cat_sub['auprc'].mean():.3f}  RMSE={cat_sub['rmse'].mean():.4f}")

    # Sensitivity analysis: compare the two marker selection methods
    print("\n" + "=" * 60)
    print("SENSITIVITY: MAX-GAP vs WILCOXON MARKER SELECTION")
    print("=" * 60)
    gap = df[df["method"] == "MarkerScoring_maxgap"]
    wlx = df[df["method"] == "MarkerScoring_wilcoxon"]
    for cat in ["rare", "moderate", "abundant", "all"]:
        if cat == "all":
            g_sub, w_sub = gap, wlx
        else:
            g_sub = gap[gap["category"] == cat]
            w_sub = wlx[wlx["category"] == cat]
        n = len(g_sub)
        print(f"  {cat:12s} (n={n:2d}): "
              f"Gap Pearson={g_sub['pearson'].mean():.3f} vs "
              f"Wlx Pearson={w_sub['pearson'].mean():.3f}  |  "
              f"Gap AUPRC={g_sub['auprc'].mean():.3f} vs "
              f"Wlx AUPRC={w_sub['auprc'].mean():.3f}")

    print(f"\n  Results saved to {OUTPUT_DIR}/marker_scoring_comparison.csv")


if __name__ == "__main__":
    main()
