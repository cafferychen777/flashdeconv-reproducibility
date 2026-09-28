"""
Cross-cohort spatial evidence for neutrophil-LAMP3-positive dendritic-cell
proximity using Marteau et al. CRC Atlas Xenium data (15 patients, 380 genes).

Data source: BioImage Archive S-BIAD2208
Paper: Marteau V et al., Cancer Cell (2025). DOI: 10.1016/j.ccell.2025.12.003

Strategy:
  1. Load processed Xenium h5ad (cell-level, annotated)
  2. Identify neutrophil cells (celltype annotation only; S100A9 gate stored as reference)
  3. Identify LAMP3+ mRegDC cells (LAMP3 > threshold within DC population)
  4. Compute spatial neighborhood enrichment per patient
  5. Permutation testing for significance
  6. Distance analysis: nearest LAMP3+ cell from each neutrophil cluster
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE = Path("/scratch/user/cafferychen777/FlashDeconv/analysis/cross_cohort_evidence")
DATA = BASE / "data"
RESULTS = BASE / "marteau_xenium" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

H5AD_PATH = DATA / "crca_xenium.h5ad"

NEUTROPHIL_MARKERS = ["S100A9", "CXCR1", "CXCR2"]
MREGDC_MARKERS = ["LAMP3", "CCR7", "CD274", "IDO1"]
MACROPHAGE_MARKERS = ["CD68", "CD163"]
MAST_MARKERS = ["KIT", "CPA3"]
TUMOR_MARKERS = ["EPCAM"]
DC_MARKERS = ["ITGAX"]

NEIGHBOR_RADII = [25, 50, 100, 200]  # µm
N_PERMUTATIONS = 1000
MIN_NEUTROPHILS_PER_PATIENT = 20
DBSCAN_EPS_UM = 50
DBSCAN_MIN_SAMPLES = 5


def load_xenium_data():
    """Load processed Xenium AnnData and extract essentials."""
    import anndata as ad

    print(f"Loading {H5AD_PATH} ...")
    adata = ad.read_h5ad(H5AD_PATH, backed="r")

    print(f"  Total cells: {adata.n_obs:,}")
    print(f"  Total genes: {adata.n_vars:,}")

    obs_cols = list(adata.obs.columns)
    print(f"  obs columns: {obs_cols[:20]}")

    if "celltype" in obs_cols:
        print("\n  Cell type distribution:")
        ct_counts = adata.obs["celltype"].value_counts()
        for ct, n in ct_counts.items():
            print(f"    {ct}: {n:,}")

    patient_col = None
    for candidate in ["patient", "sample", "donor", "patient_id", "sample_id",
                       "slide", "region", "batch"]:
        if candidate in obs_cols:
            patient_col = candidate
            break
    if patient_col:
        print(f"\n  Patient column: '{patient_col}'")
        print(f"  Patients: {adata.obs[patient_col].nunique()}")
        print(f"  Values: {sorted(adata.obs[patient_col].unique())}")

    spatial_keys = [c for c in obs_cols if c in
                    ["x_centroid", "y_centroid", "X", "Y",
                     "x", "y", "spatial_x", "spatial_y",
                     "cell_centroid_x", "cell_centroid_y"]]
    if not spatial_keys and "spatial" in adata.obsm:
        spatial_keys = ["obsm['spatial']"]

    print(f"\n  Spatial coordinate columns: {spatial_keys}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")

    return adata, patient_col, obs_cols


def explore_and_save_metadata(adata, patient_col, obs_cols):
    """Save metadata summary for planning."""
    summary = {}
    summary["n_cells"] = adata.n_obs
    summary["n_genes"] = adata.n_vars

    gene_names = list(adata.var_names)
    markers_found = {}
    all_markers = (NEUTROPHIL_MARKERS + MREGDC_MARKERS + MACROPHAGE_MARKERS +
                   MAST_MARKERS + TUMOR_MARKERS + DC_MARKERS)
    for m in all_markers:
        markers_found[m] = m in gene_names

    summary["markers_found"] = markers_found
    summary["patient_col"] = patient_col

    if patient_col:
        summary["patients"] = sorted(adata.obs[patient_col].unique().tolist())
    summary["obs_columns"] = obs_cols
    summary["obsm_keys"] = list(adata.obsm.keys())

    import json
    with open(RESULTS / "xenium_metadata_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nMetadata summary saved.")
    return summary


def extract_cell_data(adata, patient_col):
    """Extract cell coordinates, types, and gene expression for analysis."""
    obs = adata.obs.copy()

    # Spatial coordinates
    if "spatial" in adata.obsm:
        coords = np.array(adata.obsm["spatial"])
        obs["x"] = coords[:, 0]
        obs["y"] = coords[:, 1]
    else:
        for xc, yc in [("x_centroid", "y_centroid"),
                        ("cell_centroid_x", "cell_centroid_y"),
                        ("X", "Y"), ("x", "y")]:
            if xc in obs.columns and yc in obs.columns:
                obs["x"] = obs[xc].values
                obs["y"] = obs[yc].values
                break

    if "x" not in obs.columns:
        raise ValueError("Cannot find spatial coordinates in the data")

    # Gene expression for key markers
    gene_names = list(adata.var_names)
    all_markers = list(set(
        NEUTROPHIL_MARKERS + MREGDC_MARKERS + MACROPHAGE_MARKERS +
        MAST_MARKERS + TUMOR_MARKERS + DC_MARKERS
    ))
    available_markers = [m for m in all_markers if m in gene_names]

    print(f"\nExtracting expression for {len(available_markers)} markers...")
    marker_indices = [gene_names.index(m) for m in available_markers]

    # Handle backed mode - read in chunks
    import scipy.sparse as sp
    n = adata.n_obs
    chunk_size = 50000
    expr_data = {}
    for m in available_markers:
        expr_data[m] = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = adata.X[start:end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
        for i, m in enumerate(available_markers):
            midx = gene_names.index(m)
            expr_data[m][start:end] = chunk[:, midx]
        if (start // chunk_size) % 10 == 0:
            print(f"  Processed {end:,}/{n:,} cells")

    for m in available_markers:
        obs[m] = expr_data[m]

    print(f"  Done. Shape: {obs.shape}")
    return obs


def classify_cells(df):
    """Classify cells into functional categories."""
    celltype_col = "celltype" if "celltype" in df.columns else None

    # Neutrophil: annotated + marker confirmation
    if celltype_col:
        is_neutrophil_ann = df[celltype_col].str.lower().str.contains(
            "neutrophil", na=False)
    else:
        is_neutrophil_ann = pd.Series(False, index=df.index)

    # S100A9-based gating as independent confirmation
    if "S100A9" in df.columns:
        s100a9_thresh = np.percentile(df["S100A9"][df["S100A9"] > 0], 75) \
            if (df["S100A9"] > 0).sum() > 100 else 1.0
        is_neutrophil_expr = df["S100A9"] > s100a9_thresh
    else:
        is_neutrophil_expr = pd.Series(False, index=df.index)

    # Use the annotation-only gate for the primary direct-label spatial analysis.
    # S100A9 gate kept as reference but NOT used for downstream analysis
    # (S100A9 is also expressed in monocytes/macrophages, risking contamination).
    df["is_neutrophil"] = is_neutrophil_ann
    df["is_neutrophil_annotated"] = is_neutrophil_ann
    df["is_neutrophil_s100a9"] = is_neutrophil_expr
    df["is_neutrophil_broad"] = is_neutrophil_ann | is_neutrophil_expr

    # mRegDC: LAMP3+ within DC population (or all cells)
    if celltype_col:
        is_dc = df[celltype_col].str.lower().str.contains(
            "dendritic|dc", na=False)
    else:
        is_dc = pd.Series(True, index=df.index)

    if "LAMP3" in df.columns:
        lamp3_thresh = 0  # any LAMP3 expression = LAMP3+
        df["is_lamp3_positive"] = df["LAMP3"] > lamp3_thresh
        df["is_mregdc"] = is_dc & df["is_lamp3_positive"]
        # Also define LAMP3+ cells regardless of DC annotation
        df["is_lamp3_any"] = df["LAMP3"] > lamp3_thresh
    else:
        df["is_mregdc"] = pd.Series(False, index=df.index)
        df["is_lamp3_positive"] = pd.Series(False, index=df.index)
        df["is_lamp3_any"] = pd.Series(False, index=df.index)

    # Macrophage
    if celltype_col:
        df["is_macrophage"] = df[celltype_col].str.lower().str.contains(
            "macrophage", na=False)
    else:
        df["is_macrophage"] = pd.Series(False, index=df.index)

    # Mast cell
    if celltype_col:
        df["is_mast"] = df[celltype_col].str.lower().str.contains(
            "mast", na=False)
    else:
        df["is_mast"] = pd.Series(False, index=df.index)

    # Endothelial
    if celltype_col:
        df["is_endothelial"] = df[celltype_col].str.lower().str.contains(
            "endothelial", na=False)
    else:
        df["is_endothelial"] = pd.Series(False, index=df.index)

    # Tumor / epithelial
    if celltype_col:
        df["is_tumor"] = df[celltype_col].str.lower().str.contains(
            "cancer|tumor|epithelial", na=False)
    else:
        df["is_tumor"] = pd.Series(False, index=df.index)

    print("\n=== Cell classification summary ===")
    for label in ["is_neutrophil", "is_neutrophil_broad", "is_mregdc", "is_lamp3_any",
                   "is_macrophage", "is_mast", "is_endothelial", "is_tumor"]:
        if label in df.columns:
            print(f"  {label}: {df[label].sum():,} ({100*df[label].mean():.2f}%)")

    return df


def cluster_neutrophils(coords, eps=DBSCAN_EPS_UM, min_samples=DBSCAN_MIN_SAMPLES):
    """DBSCAN clustering of neutrophil spatial positions."""
    from sklearn.cluster import DBSCAN

    if len(coords) < min_samples:
        return np.full(len(coords), -1)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(coords)
    return labels


def neighborhood_enrichment(df, neutrophil_idx, radius, cell_type_col):
    """Compute enrichment of each cell type within radius of neutrophils."""
    coords_all = df[["x", "y"]].values
    coords_neut = coords_all[neutrophil_idx]

    tree = cKDTree(coords_all)

    neighbors = tree.query_ball_point(coords_neut, r=radius)
    neighbor_idx = np.unique(np.concatenate(neighbors))
    # Exclude neutrophils themselves
    neighbor_idx = np.setdiff1d(neighbor_idx, neutrophil_idx)

    if len(neighbor_idx) == 0:
        return {}

    # Cell type composition in neighborhood
    neighbor_types = df.iloc[neighbor_idx][cell_type_col]
    neighbor_counts = neighbor_types.value_counts(normalize=True)

    # Global composition (excluding neutrophils)
    non_neut_idx = np.setdiff1d(np.arange(len(df)), neutrophil_idx)
    global_types = df.iloc[non_neut_idx][cell_type_col]
    global_counts = global_types.value_counts(normalize=True)

    enrichment = {}
    for ct in global_counts.index:
        local_frac = neighbor_counts.get(ct, 0)
        global_frac = global_counts[ct]
        if global_frac > 0:
            enrichment[ct] = {
                "local_fraction": local_frac,
                "global_fraction": global_frac,
                "fold_enrichment": local_frac / global_frac,
                "log2_enrichment": np.log2(local_frac / global_frac)
                    if local_frac > 0 else -10,
            }

    return enrichment


def permutation_test_enrichment(df, neutrophil_idx, target_mask, radius,
                                 n_perm=N_PERMUTATIONS):
    """Permutation test: is target cell type enriched near neutrophils?"""
    coords_all = df[["x", "y"]].values
    coords_neut = coords_all[neutrophil_idx]

    tree = cKDTree(coords_all)

    # Observed: count target cells within radius of neutrophils
    neighbors = tree.query_ball_point(coords_neut, r=radius)
    neighbor_idx = np.unique(np.concatenate(neighbors))
    neighbor_idx = np.setdiff1d(neighbor_idx, neutrophil_idx)
    observed = target_mask.values[neighbor_idx].sum()

    # Permutation: randomly relabel target cells
    n_target = target_mask.sum()
    n_total = len(df)
    perm_counts = np.zeros(n_perm)

    rng = np.random.default_rng(42)
    for i in range(n_perm):
        perm_mask = np.zeros(n_total, dtype=bool)
        perm_mask[rng.choice(n_total, size=n_target, replace=False)] = True
        perm_counts[i] = perm_mask[neighbor_idx].sum()

    p_value = (np.sum(perm_counts >= observed) + 1) / (n_perm + 1)
    effect_size = (observed - perm_counts.mean()) / (perm_counts.std() + 1e-10)

    return {
        "observed": int(observed),
        "expected_mean": float(perm_counts.mean()),
        "expected_std": float(perm_counts.std()),
        "fold_over_expected": float(observed / (perm_counts.mean() + 1e-10)),
        "z_score": float(effect_size),
        "p_value": float(p_value),
    }


def nearest_distance_analysis(df, source_mask, target_mask):
    """Compute distances from source cells to nearest target cell."""
    source_coords = df.loc[source_mask, ["x", "y"]].values
    target_coords = df.loc[target_mask, ["x", "y"]].values

    if len(source_coords) == 0 or len(target_coords) == 0:
        return None

    tree = cKDTree(target_coords)
    dists, _ = tree.query(source_coords, k=1)

    return {
        "median_distance_um": float(np.median(dists)),
        "mean_distance_um": float(np.mean(dists)),
        "q25_distance_um": float(np.percentile(dists, 25)),
        "q75_distance_um": float(np.percentile(dists, 75)),
        "min_distance_um": float(np.min(dists)),
        "pct_within_50um": float(100 * np.mean(dists <= 50)),
        "pct_within_100um": float(100 * np.mean(dists <= 100)),
        "pct_within_200um": float(100 * np.mean(dists <= 200)),
        "n_source": len(source_coords),
        "n_target": len(target_coords),
    }


def analyze_one_patient(df_patient, patient_id):
    """Full analysis pipeline for one patient/region."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {patient_id}")
    print(f"  Total cells: {len(df_patient):,}")

    results = {"patient": patient_id, "n_cells": len(df_patient)}

    # Cell counts
    for label in ["is_neutrophil", "is_neutrophil_broad", "is_mregdc", "is_lamp3_any",
                   "is_macrophage", "is_mast", "is_endothelial", "is_tumor"]:
        if label in df_patient.columns:
            results[f"n_{label}"] = int(df_patient[label].sum())

    n_neut = df_patient["is_neutrophil"].sum()
    n_mregdc = df_patient["is_mregdc"].sum()
    print(f"  Neutrophils: {n_neut:,}")
    print(f"  mRegDC (LAMP3+DC): {n_mregdc:,}")
    print(f"  LAMP3+ (any): {df_patient['is_lamp3_any'].sum():,}")

    if n_neut < MIN_NEUTROPHILS_PER_PATIENT:
        print(f"  SKIP: too few neutrophils ({n_neut} < {MIN_NEUTROPHILS_PER_PATIENT})")
        results["skipped"] = True
        results["skip_reason"] = "too_few_neutrophils"
        return results

    results["skipped"] = False

    neut_idx = np.where(df_patient["is_neutrophil"].values)[0]

    # Cluster neutrophils
    neut_coords = df_patient.iloc[neut_idx][["x", "y"]].values
    cluster_labels = cluster_neutrophils(neut_coords)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    clustered = (cluster_labels >= 0).sum()
    results["n_neutrophil_clusters"] = n_clusters
    results["pct_clustered"] = float(100 * clustered / len(neut_idx)) if len(neut_idx) > 0 else 0

    print(f"  Neutrophil clusters: {n_clusters} ({results['pct_clustered']:.1f}% clustered)")

    # Neighborhood enrichment at multiple radii
    celltype_col = "celltype" if "celltype" in df_patient.columns else None
    for radius in NEIGHBOR_RADII:
        if celltype_col:
            enrich = neighborhood_enrichment(
                df_patient, neut_idx, radius, celltype_col)
            for ct, vals in enrich.items():
                results[f"enrichment_r{radius}_{ct}_fold"] = vals["fold_enrichment"]
                results[f"enrichment_r{radius}_{ct}_log2"] = vals["log2_enrichment"]

    # Permutation tests for key cell types
    perm_targets = {
        "mregdc": df_patient["is_mregdc"],
        "lamp3_any": df_patient["is_lamp3_any"],
        "macrophage": df_patient["is_macrophage"],
        "mast": df_patient["is_mast"],
        "endothelial": df_patient["is_endothelial"],
        "tumor": df_patient["is_tumor"],
    }

    for radius in [50, 100]:
        for target_name, target_mask in perm_targets.items():
            if target_mask.sum() < 10:
                continue
            print(f"  Permutation test: {target_name} at {radius}µm ...", end=" ")
            perm = permutation_test_enrichment(
                df_patient, neut_idx, target_mask, radius,
                n_perm=N_PERMUTATIONS)
            for k, v in perm.items():
                results[f"perm_r{radius}_{target_name}_{k}"] = v
            print(f"fold={perm['fold_over_expected']:.2f}, p={perm['p_value']:.4f}")

    # Distance analysis: neutrophils to nearest LAMP3+ cell
    if df_patient["is_lamp3_any"].sum() >= 5:
        dist_neut_lamp3 = nearest_distance_analysis(
            df_patient, df_patient["is_neutrophil"], df_patient["is_lamp3_any"])
        if dist_neut_lamp3:
            for k, v in dist_neut_lamp3.items():
                results[f"dist_neut_to_lamp3_{k}"] = v
            print(f"  Neutrophil→LAMP3+ distance: median={dist_neut_lamp3['median_distance_um']:.1f}µm, "
                  f"{dist_neut_lamp3['pct_within_100um']:.1f}% within 100µm")

    # Distance: neutrophils to nearest macrophage
    if df_patient["is_macrophage"].sum() >= 5:
        dist_neut_mac = nearest_distance_analysis(
            df_patient, df_patient["is_neutrophil"], df_patient["is_macrophage"])
        if dist_neut_mac:
            for k, v in dist_neut_mac.items():
                results[f"dist_neut_to_mac_{k}"] = v

    # Gene expression in neutrophil neighborhoods vs background
    gene_cols = [c for c in df_patient.columns
                 if c in NEUTROPHIL_MARKERS + MREGDC_MARKERS]
    if gene_cols:
        coords_all = df_patient[["x", "y"]].values
        tree = cKDTree(coords_all)
        neighbors_100 = tree.query_ball_point(
            df_patient.iloc[neut_idx][["x", "y"]].values, r=100)
        neigh_idx = np.unique(np.concatenate(neighbors_100))
        neigh_idx = np.setdiff1d(neigh_idx, neut_idx)
        bg_idx = np.setdiff1d(np.arange(len(df_patient)), neut_idx)
        bg_idx = np.setdiff1d(bg_idx, neigh_idx)

        for gene in gene_cols:
            neigh_expr = df_patient.iloc[neigh_idx][gene].mean()
            bg_expr = df_patient.iloc[bg_idx][gene].mean()
            if bg_expr > 0:
                results[f"gene_enrichment_100um_{gene}_fold"] = float(
                    neigh_expr / bg_expr)
            results[f"gene_enrichment_100um_{gene}_neigh_mean"] = float(neigh_expr)
            results[f"gene_enrichment_100um_{gene}_bg_mean"] = float(bg_expr)

    return results


def main():
    print("=" * 60)
    print("Marteau Xenium Neutrophil-mRegDC Co-localization Analysis")
    print("=" * 60)

    # Step 1: Load data
    adata, patient_col, obs_cols = load_xenium_data()

    # Step 2: Save metadata summary
    summary = explore_and_save_metadata(adata, patient_col, obs_cols)

    # Step 3: Extract cell data
    df = extract_cell_data(adata, patient_col)

    # Step 4: Classify cells
    df = classify_cells(df)

    # Save classified cell data (without expression to save space)
    save_cols = ["x", "y", "is_neutrophil", "is_mregdc", "is_lamp3_any",
                 "is_macrophage", "is_mast", "is_endothelial", "is_tumor"]
    if "celltype" in df.columns:
        save_cols.insert(0, "celltype")
    if patient_col and patient_col in df.columns:
        save_cols.insert(0, patient_col)
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_parquet(RESULTS / "classified_cells.parquet", index=False)
    print(f"\nClassified cells saved to {RESULTS / 'classified_cells.parquet'}")

    # Step 5: Per-patient analysis
    if patient_col and patient_col in df.columns:
        patients = sorted(df[patient_col].unique())
    else:
        patients = ["all"]

    all_results = []
    for pid in patients:
        if pid == "all":
            df_p = df
        else:
            df_p = df[df[patient_col] == pid].copy()
            df_p = df_p.reset_index(drop=True)
        result = analyze_one_patient(df_p, pid)
        all_results.append(result)

    # Step 6: Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS / "per_patient_results.csv", index=False)
    print(f"\nPer-patient results saved to {RESULTS / 'per_patient_results.csv'}")

    # Step 7: Summary statistics across patients
    analyzed = results_df[results_df.get("skipped", True) == False]
    if len(analyzed) > 0:
        print("\n" + "=" * 60)
        print("SUMMARY ACROSS PATIENTS")
        print("=" * 60)
        print(f"Patients analyzed: {len(analyzed)}/{len(results_df)}")

        # mRegDC enrichment near neutrophils
        for radius in [50, 100]:
            col_fold = f"perm_r{radius}_mregdc_fold_over_expected"
            col_p = f"perm_r{radius}_mregdc_p_value"
            if col_fold in analyzed.columns:
                folds = analyzed[col_fold].dropna()
                pvals = analyzed[col_p].dropna()
                print(f"\n  mRegDC enrichment at {radius}µm:")
                print(f"    Fold enrichment: {folds.median():.2f} "
                      f"(range {folds.min():.2f}-{folds.max():.2f})")
                sig = (pvals < 0.05).sum()
                print(f"    Significant (p<0.05): {sig}/{len(pvals)} patients")

            col_fold_l = f"perm_r{radius}_lamp3_any_fold_over_expected"
            col_p_l = f"perm_r{radius}_lamp3_any_p_value"
            if col_fold_l in analyzed.columns:
                folds = analyzed[col_fold_l].dropna()
                pvals = analyzed[col_p_l].dropna()
                print(f"\n  LAMP3+ cell enrichment at {radius}µm:")
                print(f"    Fold enrichment: {folds.median():.2f} "
                      f"(range {folds.min():.2f}-{folds.max():.2f})")
                sig = (pvals < 0.05).sum()
                print(f"    Significant (p<0.05): {sig}/{len(pvals)} patients")

        # Distance summary
        dist_col = "dist_neut_to_lamp3_median_distance_um"
        if dist_col in analyzed.columns:
            dists = analyzed[dist_col].dropna()
            print(f"\n  Neutrophil → LAMP3+ median distance:")
            print(f"    Across patients: {dists.median():.1f}µm "
                  f"(range {dists.min():.1f}-{dists.max():.1f})")

        # Save summary
        summary_stats = {
            "n_patients_total": len(results_df),
            "n_patients_analyzed": len(analyzed),
            "n_patients_skipped": len(results_df) - len(analyzed),
        }
        for radius in [50, 100]:
            col_fold = f"perm_r{radius}_mregdc_fold_over_expected"
            col_p = f"perm_r{radius}_mregdc_p_value"
            if col_fold in analyzed.columns:
                summary_stats[f"mregdc_r{radius}_median_fold"] = float(
                    analyzed[col_fold].dropna().median())
                summary_stats[f"mregdc_r{radius}_n_significant"] = int(
                    (analyzed[col_p].dropna() < 0.05).sum())

        import json
        with open(RESULTS / "summary_statistics.json", "w") as f:
            json.dump(summary_stats, f, indent=2)

    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    main()
