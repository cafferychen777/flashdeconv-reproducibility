"""
Laplacian ablation experiment: lambda=0 (no spatial smoothing) vs default auto-tuned lambda.

Addresses Reviewer 2's concern:
  "At minimum, an ablation study removing the Laplacian term (lambda=0) should be included."

Key hypothesis being tested (R2):
  The Laplacian low-pass filter propagates non-zero rare-cell scores from true-positive
  spots to neighboring false-positive spots, increasing recall while reducing spatial
  specificity (precision).

Metrics computed per cell type:
  - AUPRC (main metric)
  - Pearson correlation
  - RMSE
  - Precision / Recall at binary threshold
  - Mean ground-truth abundance (for rare/moderate/abundant grouping)

Runs on all 6 Spotless silver standard tissues, sample 1.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import pearsonr
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
)
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/apple/Research/FlashDeconv")
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final")
sys.path.insert(0, "/Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks")
import fdfinal  # noqa: F401  (fit log / FD_PROTOCOL)
import seedpatch  # noqa: F401  (FD_SEED)
from flashdeconv.core.deconv import FlashDeconv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/Users/apple/Research/FlashDeconv/validation/benchmark_data/converted"
TISSUE_IDS = [1, 2, 3, 4, 5, 6]
SAMPLE_ID = 1
PRESENCE_THRESHOLD = 0.01  # cell type "present" if proportion > 1%
OUTPUT_DIR = os.environ["RERUN_OUT"]

# ---------------------------------------------------------------------------
# Data loading (same as exp_alpha_sensitivity.py)
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
    X = np.zeros((len(unique_types), ref_counts.shape[1]))
    for i, ct in enumerate(unique_types):
        X[i] = ref_counts[ref_celltypes == ct].mean(axis=0)
    return X, unique_types, ref_genes


def align_genes(Y, genes_Y, X, genes_X):
    common = np.intersect1d(genes_Y, genes_X)
    idx_Y = np.array([np.where(genes_Y == g)[0][0] for g in common])
    idx_X = np.array([np.where(genes_X == g)[0][0] for g in common])
    return Y[:, idx_Y], X[:, idx_X], common


def make_coords(n_spots):
    side = int(np.ceil(np.sqrt(n_spots)))
    return np.array([[i % side, i // side] for i in range(n_spots)], dtype=float)


# ---------------------------------------------------------------------------
# Per-cell-type evaluation
# ---------------------------------------------------------------------------

def evaluate_per_celltype(pred_props, true_props_df, cell_types):
    """Compute per-cell-type metrics.

    Returns a list of dicts, one per cell type.
    """
    pred_df = pd.DataFrame(pred_props, columns=cell_types)
    common_types = sorted(set(pred_df.columns) & set(true_props_df.columns))

    rows = []
    for ct in common_types:
        pred = pred_df[ct].values
        true = true_props_df[ct].values
        mean_abundance = true.mean()

        # Pearson correlation
        if true.std() > 0 and pred.std() > 0:
            r, _ = pearsonr(pred, true)
        else:
            r = np.nan

        # RMSE
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        # AUPRC (binary: present if proportion > threshold)
        true_bin = (true > PRESENCE_THRESHOLD).astype(int)
        if true_bin.sum() > 0 and true_bin.sum() < len(true_bin):
            prec_curve, rec_curve, _ = precision_recall_curve(true_bin, pred)
            auprc = auc(rec_curve, prec_curve)
        else:
            auprc = np.nan

        # Binary precision/recall at the same threshold on predictions
        pred_bin = (pred > PRESENCE_THRESHOLD).astype(int)
        if pred_bin.sum() > 0 and true_bin.sum() > 0:
            precision = precision_score(true_bin, pred_bin, zero_division=0)
            recall = recall_score(true_bin, pred_bin, zero_division=0)
        else:
            precision = np.nan
            recall = np.nan

        # Abundance category
        if mean_abundance < 0.05:
            category = "rare"
        elif mean_abundance < 0.15:
            category = "moderate"
        else:
            category = "abundant"

        rows.append(
            {
                "cell_type": ct,
                "mean_abundance": mean_abundance,
                "category": category,
                "pearson": r,
                "rmse": rmse,
                "auprc": auprc,
                "precision": precision,
                "recall": recall,
                "n_true_positive_spots": int(true_bin.sum()),
                "n_pred_positive_spots": int(pred_bin.sum()),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_ablation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("LAPLACIAN ABLATION: lambda=0 vs auto-tuned lambda")
    print("=" * 80)

    all_rows = []

    for tissue_id in TISSUE_IDS:
        print(f"\n{'='*60}")
        print(f"Tissue {tissue_id}")
        print(f"{'='*60}")

        # Load data
        Y, genes_Y, true_props = load_silver_data(tissue_id, SAMPLE_ID)
        X, cell_types, genes_X = load_reference_data(tissue_id)
        Y_al, X_al, common_genes = align_genes(Y, genes_Y, X, genes_X)
        coords = make_coords(Y_al.shape[0])

        print(
            f"  Spots={Y_al.shape[0]}, Genes={len(common_genes)}, "
            f"CellTypes={len(cell_types)}"
        )

        for condition, lambda_val in [("no_spatial", 0.0), ("auto", "auto")]:
            print(f"\n  --- {condition} (lambda_spatial={lambda_val}) ---")
            t0 = time.time()

            model = FlashDeconv(
                sketch_dim=512,
                lambda_spatial=lambda_val,
                n_hvg=2000,
                n_markers_per_type=50,
                random_state=0,
            )
            pred = model.fit_transform(
                Y_al, X_al, coords, cell_type_names=cell_types
            )
            elapsed = time.time() - t0
            lambda_used = model.lambda_used_

            print(f"  lambda_used={lambda_used:.4f}, time={elapsed:.2f}s")

            # Per-cell-type metrics
            ct_rows = evaluate_per_celltype(pred, true_props, cell_types)
            for row in ct_rows:
                row["tissue"] = tissue_id
                row["condition"] = condition
                row["lambda_used"] = lambda_used
                all_rows.append(row)

            # Print summary
            df_ct = pd.DataFrame(ct_rows)
            for cat in ["rare", "moderate", "abundant"]:
                sub = df_ct[df_ct["category"] == cat]
                if len(sub) > 0:
                    print(
                        f"    {cat:>10s} ({len(sub)} types): "
                        f"AUPRC={sub['auprc'].mean():.4f}  "
                        f"Pearson={sub['pearson'].mean():.4f}  "
                        f"Prec={sub['precision'].mean():.4f}  "
                        f"Rec={sub['recall'].mean():.4f}"
                    )

    # ------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------
    df_all = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, "laplacian_ablation_per_celltype.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nPer-cell-type results saved to {csv_path}")

    # ------------------------------------------------------------------
    # Summary: paired comparison (auto vs no_spatial)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PAIRED COMPARISON: auto lambda vs no spatial (lambda=0)")
    print("=" * 80)

    df_auto = df_all[df_all["condition"] == "auto"].set_index(
        ["tissue", "cell_type"]
    )
    df_none = df_all[df_all["condition"] == "no_spatial"].set_index(
        ["tissue", "cell_type"]
    )

    common_idx = df_auto.index.intersection(df_none.index)
    paired = pd.DataFrame(
        {
            "category": df_auto.loc[common_idx, "category"].values,
            "mean_abundance": df_auto.loc[common_idx, "mean_abundance"].values,
            "auprc_auto": df_auto.loc[common_idx, "auprc"].values,
            "auprc_none": df_none.loc[common_idx, "auprc"].values,
            "pearson_auto": df_auto.loc[common_idx, "pearson"].values,
            "pearson_none": df_none.loc[common_idx, "pearson"].values,
            "prec_auto": df_auto.loc[common_idx, "precision"].values,
            "prec_none": df_none.loc[common_idx, "precision"].values,
            "rec_auto": df_auto.loc[common_idx, "recall"].values,
            "rec_none": df_none.loc[common_idx, "recall"].values,
        },
        index=common_idx,
    )

    paired["delta_auprc"] = paired["auprc_auto"] - paired["auprc_none"]
    paired["delta_pearson"] = paired["pearson_auto"] - paired["pearson_none"]
    paired["delta_prec"] = paired["prec_auto"] - paired["prec_none"]
    paired["delta_rec"] = paired["rec_auto"] - paired["rec_none"]

    paired_csv = os.path.join(OUTPUT_DIR, "laplacian_ablation_paired.csv")
    paired.to_csv(paired_csv)
    print(f"Paired results saved to {paired_csv}")

    # Group by category
    print(f"\n{'Category':>12s}  {'N':>3s}  {'ΔAUPRC':>8s}  {'ΔPearson':>8s}  "
          f"{'ΔPrec':>8s}  {'ΔRecall':>8s}")
    print("-" * 65)

    for cat in ["rare", "moderate", "abundant", "ALL"]:
        if cat == "ALL":
            sub = paired
        else:
            sub = paired[paired["category"] == cat]

        if len(sub) == 0:
            continue

        print(
            f"{cat:>12s}  {len(sub):>3d}  "
            f"{sub['delta_auprc'].mean():>+8.4f}  "
            f"{sub['delta_pearson'].mean():>+8.4f}  "
            f"{sub['delta_prec'].mean():>+8.4f}  "
            f"{sub['delta_rec'].mean():>+8.4f}"
        )

    # ------------------------------------------------------------------
    # R2 hypothesis test: does Laplacian increase recall but decrease
    # precision specifically for rare cell types?
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REVIEWER 2 HYPOTHESIS: Laplacian spreading for rare types")
    print("=" * 80)

    rare = paired[paired["category"] == "rare"].dropna(
        subset=["delta_prec", "delta_rec"]
    )
    if len(rare) > 0:
        n_prec_down = (rare["delta_prec"] < 0).sum()
        n_rec_up = (rare["delta_rec"] > 0).sum()
        n_total = len(rare)

        print(f"Rare cell types: {n_total}")
        print(
            f"  Precision DECREASED with Laplacian: "
            f"{n_prec_down}/{n_total} ({100*n_prec_down/n_total:.0f}%)"
        )
        print(
            f"  Recall INCREASED with Laplacian:    "
            f"{n_rec_up}/{n_total} ({100*n_rec_up/n_total:.0f}%)"
        )
        print(
            f"  Both (spreading pattern):           "
            f"{((rare['delta_prec'] < 0) & (rare['delta_rec'] > 0)).sum()}"
            f"/{n_total}"
        )
        print(f"\n  Mean ΔPrecision (rare): {rare['delta_prec'].mean():+.4f}")
        print(f"  Mean ΔRecall   (rare): {rare['delta_rec'].mean():+.4f}")
        print(f"  Mean ΔAUPRC    (rare): {rare['delta_auprc'].mean():+.4f}")

    # Per-type detail for rare types
    print(f"\n{'Tissue':>6s} {'CellType':>20s} {'Abund':>6s} "
          f"{'P_auto':>7s} {'P_none':>7s} {'R_auto':>7s} {'R_none':>7s} "
          f"{'AUPRC_a':>7s} {'AUPRC_n':>7s}")
    print("-" * 100)

    for (tissue, ct), row in rare.iterrows():
        print(
            f"{tissue:>6d} {ct:>20s} {row['mean_abundance']:>6.3f} "
            f"{row['prec_auto']:>7.3f} {row['prec_none']:>7.3f} "
            f"{row['rec_auto']:>7.3f} {row['rec_none']:>7.3f} "
            f"{row['auprc_auto']:>7.3f} {row['auprc_none']:>7.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    run_ablation()
