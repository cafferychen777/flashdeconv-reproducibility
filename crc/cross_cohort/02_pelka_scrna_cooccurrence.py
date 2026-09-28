"""
Pelka et al. 2021 CRC scRNA-seq Atlas: neutrophil/granulocyte-mRegDC co-occurrence.

Paper: Pelka K et al., Cell 184(18):4734-4752 (2021).
Data: GEO GSE178341

Key clusters:
  cM09 (mregDC): 1,596 cells — mature regulatory DCs
  cM10 (Granulocyte): 2,043 cells — granulocytes/neutrophils
  62 patients, 370K cells total
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

warnings.filterwarnings("ignore")

BASE = Path("/scratch/user/cafferychen777/FlashDeconv/analysis/cross_cohort_evidence")
DATA = BASE / "data" / "pelka"
RESULTS = BASE / "pelka_scrna" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

H5_PATH = DATA / "GSE178341_crc10x_full_c295v4_submit.h5"
CLUSTER_PATH = DATA / "GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz"
META_PATH = DATA / "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"

TARGET_GENES = [
    "LAMP3", "CCR7", "CD274", "IDO1", "FSCN1",
    "S100A8", "S100A9", "FCGR3B", "CSF3R", "CXCR1", "CXCR2",
    "CD68", "CD163", "AGER", "TLR4",
    "EPCAM", "KIT", "CPA3", "PECAM1",
]


def load_data():
    """Load cluster annotations, metadata, and gene expression."""
    print("Loading cluster annotations...")
    clusters = pd.read_csv(CLUSTER_PATH)
    print(f"  Cells: {len(clusters):,}")

    print("Loading metadata...")
    meta = pd.read_csv(META_PATH)
    print(f"  Metadata: {meta.shape}")

    # Merge
    df = clusters.merge(meta, left_on="sampleID", right_on="cellID", how="left")
    print(f"  Merged: {df.shape}")

    # Extract patient ID from PatientTypeID (e.g., "C103_T" -> "C103")
    if "PatientTypeID" in df.columns:
        df["patient"] = df["PatientTypeID"].str.extract(r"(C\d+)", expand=False)
        df["tissue_type"] = df["PatientTypeID"].str.extract(r"_(\w+)$", expand=False)
        print(f"  Patients: {df['patient'].nunique()}")
        print(f"  Tumor samples: {(df['tissue_type'] == 'T').sum():,}")
        print(f"  Normal samples: {(df['tissue_type'] == 'N').sum():,}")
    elif "PID" in df.columns:
        df["patient"] = df["PID"]
        df["tissue_type"] = df.get("SPECIMEN_TYPE", "T")

    return df


def load_expression(n_cells):
    """Load gene expression from H5 file for target genes."""
    import h5py
    import scipy.sparse as sp

    print(f"\nLoading expression from {H5_PATH.name}...")
    with h5py.File(H5_PATH, "r") as f:
        # Get gene names
        features = f["matrix"]["features"]
        if "name" in features:
            gene_names = [g.decode() if isinstance(g, bytes) else g
                          for g in features["name"][:]]
        elif "id" in features:
            gene_names = [g.decode() if isinstance(g, bytes) else g
                          for g in features["id"][:]]
        else:
            gene_names = [f"gene_{i}" for i in range(features["id"].shape[0])]

        print(f"  Total genes: {len(gene_names)}")

        # Find target gene indices
        available = []
        gene_idx_map = {}
        for g in TARGET_GENES:
            if g in gene_names:
                gene_idx_map[g] = gene_names.index(g)
                available.append(g)

        print(f"  Target genes found: {len(available)}/{len(TARGET_GENES)}")
        missing = [g for g in TARGET_GENES if g not in gene_names]
        if missing:
            print(f"  Missing: {missing}")

        # Load sparse matrix
        data = f["matrix"]["data"][:]
        indices = f["matrix"]["indices"][:]
        indptr = f["matrix"]["indptr"][:]
        shape = tuple(f["matrix"]["shape"][:])

        mat = sp.csc_matrix((data, indices, indptr), shape=shape)
        print(f"  Matrix shape: {mat.shape} (genes x cells)")

        # Extract target gene expression
        expr_data = {}
        for g, idx in gene_idx_map.items():
            expr_data[g] = np.array(mat[idx, :].todense()).flatten()

    return pd.DataFrame(expr_data), available


def analyze_cooccurrence(df, expr_df):
    """Per-patient co-occurrence of mRegDC and granulocytes."""
    # Tumor samples only (most relevant for neutrophil microdomains)
    tumor_mask = df["tissue_type"] == "T"
    df_tumor = df[tumor_mask].copy()
    print(f"\nTumor samples: {len(df_tumor):,} cells")

    # Define cell populations using cl295v11SubFull
    full_col = "cl295v11SubFull"
    short_col = "cl295v11SubShort"

    df["is_mregdc"] = df[short_col] == "cM09"
    df["is_granulocyte"] = df[short_col] == "cM10"
    df["is_macrophage"] = df[short_col] == "cM02"
    df["is_monocyte"] = df[short_col] == "cM01"
    df["is_dc_any"] = df["clMidwayPr"] == "DC"
    df["is_mast"] = df[short_col] == "cMA01"

    print("\n=== Cell populations (all tissues) ===")
    for pop in ["is_mregdc", "is_granulocyte", "is_macrophage",
                "is_monocyte", "is_dc_any", "is_mast"]:
        print(f"  {pop}: {df[pop].sum():,}")

    # Per-patient analysis (tumor only)
    patients = sorted(df_tumor["patient"].unique())
    patient_results = []

    for pid in patients:
        mask = (df["patient"] == pid) & (df["tissue_type"] == "T")
        df_p = df[mask]
        n = len(df_p)
        if n < 50:
            continue

        row = {
            "patient": pid,
            "n_cells": n,
            "n_mregdc": int(df_p["is_mregdc"].sum()),
            "n_granulocyte": int(df_p["is_granulocyte"].sum()),
            "n_macrophage": int(df_p["is_macrophage"].sum()),
            "n_monocyte": int(df_p["is_monocyte"].sum()),
            "n_dc_any": int(df_p["is_dc_any"].sum()),
            "n_mast": int(df_p["is_mast"].sum()),
            "frac_mregdc": float(df_p["is_mregdc"].mean()),
            "frac_granulocyte": float(df_p["is_granulocyte"].mean()),
            "frac_macrophage": float(df_p["is_macrophage"].mean()),
        }

        # Expression of key genes in this patient
        p_idx = df_p.index
        for gene in ["LAMP3", "S100A8", "S100A9", "AGER", "TLR4"]:
            if gene in expr_df.columns:
                row[f"mean_{gene}"] = float(expr_df.loc[p_idx, gene].mean())
                # Mean in myeloid cells only
                myeloid_mask_p = df_p["clTopLevel"] == "Myeloid"
                if myeloid_mask_p.sum() > 0:
                    myeloid_idx = df_p[myeloid_mask_p].index
                    row[f"mean_{gene}_myeloid"] = float(
                        expr_df.loc[myeloid_idx, gene].mean())

        patient_results.append(row)

    patient_df = pd.DataFrame(patient_results)
    print(f"\n  Analyzed {len(patient_df)} tumor patients")

    return patient_df


def correlation_analysis(patient_df):
    """Test correlation between mRegDC and granulocyte fractions."""
    results = {}

    # Patients with at least some myeloid cells
    valid = patient_df[
        (patient_df["n_mregdc"] + patient_df["n_granulocyte"]) > 0
    ].copy()

    print(f"\n=== Correlation Analysis (n={len(valid)} patients) ===")

    # Core test: granulocyte vs mRegDC fraction
    if valid["frac_granulocyte"].std() > 0 and valid["frac_mregdc"].std() > 0:
        r, p = stats.spearmanr(valid["frac_granulocyte"], valid["frac_mregdc"])
        results["granulocyte_vs_mregdc_rho"] = float(r)
        results["granulocyte_vs_mregdc_p"] = float(p)
        results["granulocyte_vs_mregdc_n"] = len(valid)
        print(f"  Granulocyte% vs mRegDC%: rho={r:.3f}, p={p:.4f}")

    # Granulocyte vs macrophage
    if valid["frac_granulocyte"].std() > 0 and valid["frac_macrophage"].std() > 0:
        r, p = stats.spearmanr(valid["frac_granulocyte"], valid["frac_macrophage"])
        results["granulocyte_vs_macrophage_rho"] = float(r)
        results["granulocyte_vs_macrophage_p"] = float(p)
        print(f"  Granulocyte% vs Macrophage%: rho={r:.3f}, p={p:.4f}")

    # Gene expression correlations
    for g1, g2 in [("mean_S100A8_myeloid", "mean_LAMP3_myeloid"),
                    ("mean_S100A9_myeloid", "mean_LAMP3_myeloid"),
                    ("mean_S100A8", "mean_LAMP3"),
                    ("mean_S100A9", "mean_LAMP3")]:
        if g1 in valid.columns and g2 in valid.columns:
            x = valid[g1].dropna()
            y = valid[g2].dropna()
            common = x.index.intersection(y.index)
            if len(common) >= 5:
                r, p = stats.spearmanr(x[common], y[common])
                results[f"{g1}_vs_{g2}_rho"] = float(r)
                results[f"{g1}_vs_{g2}_p"] = float(p)
                print(f"  {g1} vs {g2}: rho={r:.3f}, p={p:.4f}")

    # Patients with both populations present
    both_present = valid[
        (valid["n_granulocyte"] > 0) & (valid["n_mregdc"] > 0)]
    results["n_patients_both_present"] = len(both_present)
    results["n_patients_granulocyte_only"] = len(
        valid[(valid["n_granulocyte"] > 0) & (valid["n_mregdc"] == 0)])
    results["n_patients_mregdc_only"] = len(
        valid[(valid["n_granulocyte"] == 0) & (valid["n_mregdc"] > 0)])
    results["n_patients_neither"] = len(
        valid[(valid["n_granulocyte"] == 0) & (valid["n_mregdc"] == 0)])

    print(f"\n  Patients with both granulocyte AND mRegDC: {results['n_patients_both_present']}")
    print(f"  Granulocyte only: {results['n_patients_granulocyte_only']}")
    print(f"  mRegDC only: {results['n_patients_mregdc_only']}")
    print(f"  Neither: {results['n_patients_neither']}")

    # Fisher's exact test for co-occurrence
    a = results["n_patients_both_present"]
    b = results["n_patients_granulocyte_only"]
    c = results["n_patients_mregdc_only"]
    d = results["n_patients_neither"]
    table = np.array([[a, b], [c, d]])
    odds_ratio, fisher_p = stats.fisher_exact(table)
    results["fisher_odds_ratio"] = float(odds_ratio)
    results["fisher_p"] = float(fisher_p)
    print(f"\n  Fisher's exact test for co-occurrence:")
    print(f"    Odds ratio: {odds_ratio:.2f}, p={fisher_p:.4f}")

    return results


def lr_expression_analysis(df, expr_df):
    """Analyze S100A8/A9 and AGER/TLR4 expression by cell type."""
    short_col = "cl295v11SubFull"

    myeloid_clusters = ["cM09 (mregDC)", "cM10 (Granulocyte)",
                         "cM02 (Macrophage-like)", "cM01 (Monocyte)",
                         "cM03 (DC1)", "cM04 (DC2)", "cM05 (DC2 C1Q+)"]

    results = []
    for cluster in myeloid_clusters:
        mask = df[short_col] == cluster
        if mask.sum() == 0:
            continue
        idx = df[mask].index
        row = {"cluster": cluster, "n_cells": int(mask.sum())}
        for gene in ["S100A8", "S100A9", "LAMP3", "CCR7", "CD274",
                      "IDO1", "AGER", "TLR4", "CD68", "CD163"]:
            if gene in expr_df.columns:
                vals = expr_df.loc[idx, gene]
                row[f"{gene}_mean"] = float(vals.mean())
                row[f"{gene}_pct_positive"] = float(100 * (vals > 0).mean())
        results.append(row)

    results_df = pd.DataFrame(results)
    return results_df


def main():
    print("=" * 60)
    print("Pelka 2021 CRC Atlas: Granulocyte-mRegDC Co-occurrence")
    print("=" * 60)

    # Load data
    df = load_data()

    # Load expression
    expr_df, available_genes = load_expression(len(df))

    # Attach expression to main df index
    assert len(expr_df) == len(df), f"Length mismatch: {len(expr_df)} vs {len(df)}"

    # Per-patient co-occurrence
    patient_df = analyze_cooccurrence(df, expr_df)
    patient_df.to_csv(RESULTS / "per_patient_cooccurrence.csv", index=False)
    print(f"\nSaved: {RESULTS / 'per_patient_cooccurrence.csv'}")

    # Correlation analysis
    corr_results = correlation_analysis(patient_df)
    with open(RESULTS / "correlation_results.json", "w") as f:
        json.dump(corr_results, f, indent=2)
    print(f"Saved: {RESULTS / 'correlation_results.json'}")

    # L-R expression by cell type
    lr_df = lr_expression_analysis(df, expr_df)
    lr_df.to_csv(RESULTS / "lr_expression_by_cluster.csv", index=False)
    print(f"\nSaved: {RESULTS / 'lr_expression_by_cluster.csv'}")

    print("\n=== L-R Expression Summary ===")
    for _, row in lr_df.iterrows():
        s100 = row.get("S100A8_mean", 0) + row.get("S100A9_mean", 0)
        lamp3 = row.get("LAMP3_mean", 0)
        ager = row.get("AGER_mean", 0)
        tlr4 = row.get("TLR4_mean", 0)
        print(f"  {row['cluster']:30s}: S100A8+A9={s100:.3f}, "
              f"LAMP3={lamp3:.3f}, AGER={ager:.3f}, TLR4={tlr4:.3f}")

    # Save summary
    summary = {
        "data_source": "Pelka et al. 2021 (GSE178341)",
        "n_cells_total": len(df),
        "n_mregdc": int(df["is_mregdc"].sum()),
        "n_granulocyte": int(df["is_granulocyte"].sum()),
        "n_patients": int(df["patient"].nunique()),
        "n_tumor_patients": int(
            df[df["tissue_type"] == "T"]["patient"].nunique()),
        "genes_available": available_genes,
        "correlation": corr_results,
    }
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Pelka analysis complete.")


if __name__ == "__main__":
    main()
