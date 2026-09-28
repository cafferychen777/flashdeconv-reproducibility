"""
Schürch et al. 2020 CODEX CRC dataset: CD15(granulocyte)-DC neighborhood analysis.

Paper: Schürch CM et al., Cell 182(5):1341-1359 (2020).
Data: TCIA CRC_FFPE-CODEX_CellNeighs

Strategy:
  1. Load cell-level data (coordinates, phenotypes, marker intensities)
  2. Identify granulocyte cells (CD15+ / cell type annotation)
  3. Identify DC cells (CD11c+/HLA-DR+)
  4. Compute neighborhood enrichment of DCs near granulocytes
  5. Compare with other cell type neighborhoods
  6. Per-patient analysis (35 patients)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path("/scratch/user/cafferychen777/FlashDeconv/analysis/cross_cohort_evidence")
DATA = BASE / "data" / "schurch_codex"
RESULTS = BASE / "schurch_codex" / "results"
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

NEIGHBOR_RADII = [30, 50, 100, 200]
N_PERMUTATIONS = 1000

# Cell type mapping from Schürch et al.
GRANULOCYTE_TYPES = ["Granulocyte", "granulocyte", "Neutrophil", "neutrophil"]
DC_TYPES = ["DC", "Dendritic", "dendritic cell", "DC/Mono"]
MACROPHAGE_TYPES = ["Macrophage", "macrophage", "M1 Macrophage", "M2 Macrophage"]
MAST_TYPES = ["Mast"]
ENDOTHELIAL_TYPES = ["Endothelial", "endothelial"]
TUMOR_TYPES = ["Tumor", "tumor", "Cancer", "Epithelial"]


def find_data_files():
    """Find Schürch CODEX data files."""
    candidates = list(DATA.glob("*.csv")) + list(DATA.glob("*.tsv"))
    candidates += list(DATA.glob("**/*.csv")) + list(DATA.glob("**/*.tsv"))

    if candidates:
        print(f"Found {len(candidates)} data files:")
        for f in candidates[:10]:
            print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    return candidates


def load_codex_data(data_path):
    """Load CODEX cell-level data."""
    print(f"Loading {data_path.name} ...")

    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix == ".tsv":
        df = pd.read_csv(data_path, sep="\t")
    else:
        df = pd.read_csv(data_path)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:30]}")

    return df


def explore_codex_structure(df):
    """Explore the structure of the CODEX data."""
    print("\n=== Data exploration ===")

    # Find coordinate columns
    coord_cols = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("x", "x:x", "x_centroid", "cell_centroid_x"):
            coord_cols["x"] = c
        elif "x" in cl and ("coord" in cl or "centroid" in cl or "position" in cl):
            coord_cols["x"] = c
        if cl in ("y", "y:y", "y_centroid", "cell_centroid_y"):
            coord_cols["y"] = c
        elif "y" in cl and ("coord" in cl or "centroid" in cl or "position" in cl):
            coord_cols["y"] = c
    print(f"  Coordinate columns: {coord_cols}")

    # Find cell type column
    celltype_col = None
    for c in df.columns:
        cl = c.lower()
        if c == "ClusterName" or any(kw in cl for kw in [
                "celltype", "cell_type", "phenotype",
                "clustername", "cluster_name", "annotation"]):
            celltype_col = c
            break
    if celltype_col:
        print(f"\n  Cell type column: {celltype_col}")
        ct_counts = df[celltype_col].value_counts()
        print(f"  Cell types ({len(ct_counts)}):")
        for ct, n in ct_counts.items():
            print(f"    {ct}: {n:,}")

    # Find patient/region column
    patient_col = None
    for c in df.columns:
        cl = c.lower()
        if any(kw in cl for kw in ["patient", "sample", "region", "group",
                                     "tissue", "donor"]):
            patient_col = c
            break
    if patient_col:
        print(f"\n  Patient column: {patient_col}")
        print(f"  N patients/regions: {df[patient_col].nunique()}")

    return coord_cols, celltype_col, patient_col


def classify_cells(df, celltype_col):
    """Map cell types to functional categories."""
    ct_lower = df[celltype_col].str.lower()

    df["is_granulocyte"] = ct_lower.str.contains(
        "granulocyte|neutrophil", na=False)
    df["is_dc"] = ct_lower.str.contains(
        "dendritic|cd11c.*dc|\\bdc[s]?\\b", na=False, regex=True)
    df["is_macrophage"] = ct_lower.str.contains(
        "macrophage|monocyte|cd68|cd163|cd11b.*mono", na=False)
    df["is_mast"] = ct_lower.str.contains("mast", na=False)
    df["is_endothelial"] = ct_lower.str.contains(
        "endothelial|vasculature|lymphatic", na=False)
    df["is_tumor"] = ct_lower.str.contains(
        "tumor|cancer", na=False)

    print("\n=== Cell classification ===")
    for label in ["is_granulocyte", "is_dc", "is_macrophage",
                   "is_mast", "is_endothelial", "is_tumor"]:
        print(f"  {label}: {df[label].sum():,} ({100*df[label].mean():.2f}%)")

    return df


def neighborhood_enrichment_codex(df, source_mask, target_mask, xcol, ycol,
                                    radius=100, n_perm=N_PERMUTATIONS):
    """Compute enrichment of target cells near source cells with permutation test."""
    coords_all = df[[xcol, ycol]].values
    source_idx = np.where(source_mask.values)[0]

    if len(source_idx) < 5:
        return None

    tree = cKDTree(coords_all)

    # Get neighbors of source cells
    neighbors = tree.query_ball_point(coords_all[source_idx], r=radius)
    neigh_idx = np.unique(np.concatenate(neighbors))
    neigh_idx = np.setdiff1d(neigh_idx, source_idx)

    if len(neigh_idx) == 0:
        return None

    # Observed target count in neighborhood
    observed = target_mask.values[neigh_idx].sum()
    total_neighbors = len(neigh_idx)
    observed_frac = observed / total_neighbors if total_neighbors > 0 else 0

    # Global target fraction (excluding source cells)
    non_source = np.setdiff1d(np.arange(len(df)), source_idx)
    global_frac = target_mask.values[non_source].mean()

    fold = observed_frac / global_frac if global_frac > 0 else np.nan

    # Permutation test
    n_target = target_mask.sum()
    perm_counts = np.zeros(n_perm)
    rng = np.random.default_rng(42)

    for i in range(n_perm):
        perm_mask = np.zeros(len(df), dtype=bool)
        perm_mask[rng.choice(len(df), size=n_target, replace=False)] = True
        perm_counts[i] = perm_mask[neigh_idx].sum()

    p_value = (np.sum(perm_counts >= observed) + 1) / (n_perm + 1)

    return {
        "observed": int(observed),
        "total_neighbors": total_neighbors,
        "observed_fraction": float(observed_frac),
        "global_fraction": float(global_frac),
        "fold_enrichment": float(fold),
        "log2_fold": float(np.log2(fold)) if fold > 0 and not np.isnan(fold) else np.nan,
        "p_value": float(p_value),
        "n_source": len(source_idx),
    }


def analyze_per_patient(df, patient_col, xcol, ycol):
    """Per-patient neighborhood enrichment analysis."""
    patients = sorted(df[patient_col].unique())
    results = []

    for pid in patients:
        mask = df[patient_col] == pid
        df_p = df[mask]

        n_gran = df_p["is_granulocyte"].sum()
        n_dc = df_p["is_dc"].sum()

        row = {
            "patient": pid,
            "n_cells": len(df_p),
            "n_granulocyte": int(n_gran),
            "n_dc": int(n_dc),
            "n_macrophage": int(df_p["is_macrophage"].sum()),
        }

        if n_gran < 5:
            row["skipped"] = True
            results.append(row)
            continue

        row["skipped"] = False

        # DC enrichment near granulocytes
        for radius in [50, 100]:
            for target, target_name in [
                ("is_dc", "dc"),
                ("is_macrophage", "macrophage"),
                ("is_mast", "mast"),
                ("is_endothelial", "endothelial"),
                ("is_tumor", "tumor"),
            ]:
                if df_p[target].sum() < 5:
                    continue
                enrich = neighborhood_enrichment_codex(
                    df_p, df_p["is_granulocyte"], df_p[target],
                    xcol, ycol, radius=radius, n_perm=500)
                if enrich:
                    for k, v in enrich.items():
                        row[f"r{radius}_{target_name}_{k}"] = v

        results.append(row)
        if not row.get("skipped"):
            dc_fold = row.get(f"r100_dc_fold_enrichment", "N/A")
            dc_p = row.get(f"r100_dc_p_value", "N/A")
            print(f"  {pid}: gran={n_gran}, DC fold@100µm={dc_fold}, p={dc_p}")

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Schürch CODEX CRC: Granulocyte-DC Neighborhood Analysis")
    print("=" * 60)

    # Find data
    files = find_data_files()
    if not files:
        print("No data files found. Please download first.")
        print(f"Expected location: {DATA}")
        print("\nTo download from TCIA:")
        print("  Look for processed cell-level CSV in supplementary data")
        print("  or use TCIA NBIA Data Retriever")
        return

    # Load the largest CSV (likely the main data)
    main_file = max(files, key=lambda f: f.stat().st_size)
    df = load_codex_data(main_file)

    # Explore structure
    coord_cols, celltype_col, patient_col = explore_codex_structure(df)

    if not coord_cols or not celltype_col:
        print("ERROR: Cannot identify coordinate or cell type columns")
        return

    xcol = coord_cols.get("x", list(coord_cols.values())[0])
    ycol = coord_cols.get("y", list(coord_cols.values())[1] if len(coord_cols) > 1 else None)

    if ycol is None:
        print("ERROR: Cannot find Y coordinate column")
        return

    # Classify cells
    df = classify_cells(df, celltype_col)

    # Overall enrichment
    print("\n=== Global neighborhood enrichment ===")
    for radius in [50, 100, 200]:
        for target, name in [("is_dc", "DC"), ("is_macrophage", "Macrophage"),
                              ("is_mast", "Mast"), ("is_endothelial", "Endothelial")]:
            if df[target].sum() < 5 or df["is_granulocyte"].sum() < 5:
                continue
            enrich = neighborhood_enrichment_codex(
                df, df["is_granulocyte"], df[target], xcol, ycol,
                radius=radius, n_perm=N_PERMUTATIONS)
            if enrich:
                print(f"  {name} at {radius}µm: "
                      f"fold={enrich['fold_enrichment']:.2f}, "
                      f"p={enrich['p_value']:.4f}")

    # Per-patient analysis
    if patient_col:
        print("\n=== Per-patient analysis ===")
        patient_results = analyze_per_patient(df, patient_col, xcol, ycol)
        patient_results.to_csv(RESULTS / "per_patient_granulocyte_dc.csv", index=False)
        print(f"\nSaved: {RESULTS / 'per_patient_granulocyte_dc.csv'}")

        # Summary
        analyzed = patient_results[patient_results.get("skipped", True) == False]
        if len(analyzed) > 0:
            dc_fold_col = "r100_dc_fold_enrichment"
            if dc_fold_col in analyzed.columns:
                folds = analyzed[dc_fold_col].dropna()
                print(f"\n  DC enrichment at 100µm (n={len(folds)} patients):")
                print(f"    Median fold: {folds.median():.2f}")
                print(f"    Range: {folds.min():.2f} - {folds.max():.2f}")
                sig = analyzed.get("r100_dc_p_value", pd.Series()).dropna()
                if len(sig) > 0:
                    print(f"    Significant (p<0.05): {(sig < 0.05).sum()}/{len(sig)}")

    print(f"\n✓ CODEX analysis complete.")


if __name__ == "__main__":
    main()
