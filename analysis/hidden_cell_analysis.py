#!/usr/bin/env python
"""
Hidden Cell Type Analysis
==========================

Identify cell types whose markers would be missed by HVG selection but
captured by leverage-based gene selection.

This analysis demonstrates FlashDeconv's ability to detect rare cell types
like Tuft (brush) cells that have low-variance markers.

Requirements:
    - Haber et al. intestine scRNA-seq reference
    - Multiscale proportions from resolution_horizon_analysis.py

Usage:
    python hidden_cell_analysis.py --data_dir ./data --output_dir ./results

Output:
    - cell_type_visibility.csv (HVG blindness scores per cell type)
    - hidden_genes.csv (genes with high leverage but low variance)
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from pathlib import Path


def compute_leverage_scores(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage scores for genes in reference matrix.

    Parameters
    ----------
    X : np.ndarray
        Reference matrix (K x G), cell types by genes

    Returns
    -------
    np.ndarray
        Leverage scores for each gene (G,)
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    leverage = np.sum(Vt ** 2, axis=0)
    return leverage


def load_reference_data(data_dir):
    """Load intestine scRNA-seq reference."""
    # Try different possible reference file names
    for ref_name in ['haber_intestine_matched.h5ad', 'haber_processed.h5ad']:
        ref_path = Path(data_dir) / ref_name
        if ref_path.exists():
            print(f"Loading reference data ({ref_name})...")
            ref = sc.read_h5ad(ref_path)
            print(f"  Reference: {ref.n_obs} cells, {ref.n_vars} genes")

            ct_col = 'celltype1' if 'celltype1' in ref.obs else 'cell_type'
            cell_types = ref.obs[ct_col].value_counts()
            print(f"\n  Cell type abundances ({ct_col}):")
            for ct, count in cell_types.items():
                pct = count / len(ref) * 100
                rarity = "**RARE**" if pct < 5 else ""
                print(f"    {ct}: {count} ({pct:.1f}%) {rarity}")

            return ref, ct_col

    raise FileNotFoundError(
        f"Reference data not found in {data_dir}. "
        "Please download haber_processed.h5ad from Zenodo."
    )


def build_signature_matrix(adata, ct_col):
    """Build cell type signature matrix."""
    print("\nBuilding signature matrix...")
    cell_types = sorted(adata.obs[ct_col].unique())
    n_genes = adata.n_vars

    X = np.zeros((len(cell_types), n_genes))

    for i, ct in enumerate(cell_types):
        mask = adata.obs[ct_col] == ct
        if hasattr(adata.X, 'toarray'):
            expr = adata[mask].X.toarray()
        else:
            expr = np.asarray(adata[mask].X)
        X[i, :] = expr.mean(axis=0)

    # Normalize to CPM-like scale
    X = X / (X.sum(axis=1, keepdims=True) + 1e-10) * 1e4

    print(f"  Signature matrix: {X.shape[0]} cell types x {X.shape[1]} genes")
    return X, cell_types


def compute_marker_visibility(X, cell_types, gene_names, n_markers=30):
    """
    For each cell type, analyze how visible its markers are
    to variance-based vs leverage-based gene selection.
    """
    print("\nComputing marker visibility analysis...")

    leverage = compute_leverage_scores(X)
    gene_var = np.var(X, axis=0)

    leverage_rank = np.argsort(np.argsort(-leverage))
    variance_rank = np.argsort(np.argsort(-gene_var))

    n_genes = len(gene_names)
    results = []

    for ct_idx, ct_name in enumerate(cell_types):
        ct_expr = X[ct_idx, :]
        other_mask = np.ones(len(cell_types), dtype=bool)
        other_mask[ct_idx] = False
        other_expr = X[other_mask].mean(axis=0)

        # Log fold change
        fc = np.log2((ct_expr + 1) / (other_expr + 1))

        # Get top markers
        marker_idx = np.argsort(-fc)[:n_markers]
        marker_genes = [gene_names[i] for i in marker_idx]
        marker_fc = fc[marker_idx]

        # Check ranks
        marker_leverage_ranks = leverage_rank[marker_idx]
        marker_variance_ranks = variance_rank[marker_idx]

        # Percentiles
        marker_leverage_pct = np.mean(marker_leverage_ranks) / n_genes * 100
        marker_variance_pct = np.mean(marker_variance_ranks) / n_genes * 100

        # HVG blindness
        hvg_blindness = marker_variance_pct - marker_leverage_pct

        results.append({
            'cell_type': ct_name,
            'marker_leverage_pct': marker_leverage_pct,
            'marker_variance_pct': marker_variance_pct,
            'hvg_blindness': hvg_blindness,
            'top_marker': marker_genes[0],
            'top_marker_fc': marker_fc[0],
            'top_5_markers': ', '.join(marker_genes[:5]),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('hvg_blindness', ascending=False)

    print("\n  Cell types ranked by 'HVG blindness' (higher = HVG more likely to miss):")
    print("  " + "-" * 70)
    for _, row in results_df.iterrows():
        print(f"  {row['cell_type']:<30} HVG blindness: {row['hvg_blindness']:>6.1f}%")

    return results_df


def find_hidden_genes(X, gene_names, top_n=50):
    """Find genes with high leverage but low variance."""
    print("\n" + "=" * 60)
    print("Finding 'hidden' genes (high leverage, low variance)...")
    print("=" * 60)

    leverage = compute_leverage_scores(X)
    gene_var = np.var(X, axis=0)

    n_genes = len(gene_names)

    leverage_pct = np.argsort(np.argsort(-leverage)) / n_genes * 100
    variance_pct = np.argsort(np.argsort(-gene_var)) / n_genes * 100

    hidden_score = variance_pct - leverage_pct

    hidden_idx = np.argsort(-hidden_score)[:top_n]

    results = []
    for idx in hidden_idx:
        results.append({
            'gene': gene_names[idx],
            'leverage': leverage[idx],
            'variance': gene_var[idx],
            'leverage_pct': leverage_pct[idx],
            'variance_pct': variance_pct[idx],
            'hidden_score': hidden_score[idx],
        })

    results_df = pd.DataFrame(results)
    print(f"\n  Top 20 'hidden' genes (high leverage, low variance):")
    print(results_df.head(20).to_string())

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze hidden cell types in intestinal reference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing reference data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HIDDEN CELL TYPE ANALYSIS")
    print("=" * 60)

    # Load reference
    ref, ct_col = load_reference_data(args.data_dir)

    # Build signature matrix
    X, cell_types = build_signature_matrix(ref, ct_col)
    gene_names = ref.var_names.tolist()

    # Analyze marker visibility
    visibility_df = compute_marker_visibility(X, cell_types, gene_names)
    visibility_df.to_csv(output_dir / 'cell_type_visibility.csv', index=False)

    # Find hidden genes
    hidden_genes_df = find_hidden_genes(X, gene_names)
    hidden_genes_df.to_csv(output_dir / 'hidden_genes.csv', index=False)

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    hidden_types = visibility_df[visibility_df['hvg_blindness'] > 5].copy()
    if len(hidden_types) > 0:
        print("\nCell types with markers that HVG would likely MISS (>5% blindness):")
        for _, row in hidden_types.iterrows():
            print(f"  - {row['cell_type']}")
            print(f"    HVG blindness: {row['hvg_blindness']:.1f}%")
            print(f"    Top 5 markers: {row['top_5_markers']}")
    else:
        print("\nNo cell types show strong HVG blindness (>5% difference)")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
