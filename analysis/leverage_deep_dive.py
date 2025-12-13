#!/usr/bin/env python
"""
Leverage Deep Dive Analysis
============================

This script analyzes how leverage scores decouple biological identity from
population abundance, generating intermediate data for Figure 2.

Key experiments:
1. Abundance invariance test - how gene rankings change with cell type abundance
2. Gene quadrant analysis - categorizing genes by variance vs leverage
3. GO enrichment submission for GOLD genes

Requirements:
    - Mouse brain scRNA-seq reference (scrna_reference.h5ad)
    - Cell annotation file (cell_annotation.csv)

Data sources:
    Download from Cell2location tutorial:
    https://cell2location.cog.sanger.ac.uk/tutorial/

Usage:
    python leverage_deep_dive.py --data_dir ./data --output_dir ./results
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


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
    K, G = X.shape
    # Centering
    X_centered = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Leverage = sum of squared right singular vectors
    leverage = np.sum(Vt ** 2, axis=0)
    return leverage


def load_mouse_brain_data(data_dir):
    """Load mouse brain scRNA-seq reference data."""
    data_path = Path(data_dir) / 'mouse_brain'

    print("Loading scRNA-seq reference...")
    sc_adata = sc.read_h5ad(data_path / 'scrna_reference.h5ad')
    anno = pd.read_csv(data_path / 'cell_annotation.csv', index_col=0)

    # Subset to cells with annotation
    common_cells = list(set(sc_adata.obs_names) & set(anno.index))
    sc_adata = sc_adata[common_cells].copy()
    sc_adata.obs['cell_type'] = anno.loc[sc_adata.obs_names, 'annotation_1'].values

    print(f"  Loaded {sc_adata.n_obs} cells, {sc_adata.n_vars} genes")
    print(f"  Cell types: {sc_adata.obs['cell_type'].nunique()}")

    return sc_adata


def build_reference_matrix(sc_adata, cell_type_col='cell_type'):
    """Build cell type reference matrix from scRNA-seq data."""
    print("Building reference matrix...")

    cell_types = sorted(sc_adata.obs[cell_type_col].unique())
    n_genes = sc_adata.n_vars

    X = np.zeros((len(cell_types), n_genes))

    for i, ct in enumerate(cell_types):
        mask = (sc_adata.obs[cell_type_col] == ct).values
        if issparse(sc_adata.X):
            X[i] = sc_adata.X[mask].toarray().mean(axis=0)
        else:
            X[i] = sc_adata.X[mask].mean(axis=0)

    # Log-normalize
    X = np.log1p(X / (X.sum(axis=1, keepdims=True) + 1e-10) * 1e4)

    print(f"  Reference shape: {X.shape}")
    return X, cell_types, list(sc_adata.var_names)


def experiment1_abundance_invariance(X, cell_types, gene_names, output_dir):
    """
    Test how leverage vs variance rankings change with abundance.

    Simulates downsampling the dominant cell type and tracking marker gene ranks.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Abundance Invariance Test")
    print("=" * 60)

    # Find dominant cell type (highest mean expression across genes)
    dominant_idx = np.argmax(X.sum(axis=1))
    dominant_type = cell_types[dominant_idx]
    print(f"  Dominant type: {dominant_type}")

    # Find marker genes for dominant type (high fold change)
    dom_expr = X[dominant_idx]
    other_expr = np.delete(X, dominant_idx, axis=0).mean(axis=0)
    fc = dom_expr - other_expr  # log fold change
    marker_genes_idx = np.argsort(-fc)[:50]

    results = []
    abundances = [100, 80, 60, 40, 20, 10, 5]

    for pct in abundances:
        # Simulate reduced abundance by scaling down the dominant type
        X_sim = X.copy()
        X_sim[dominant_idx] *= (pct / 100)

        # Compute variance and leverage
        gene_var = np.var(X_sim, axis=0)
        leverage = compute_leverage_scores(X_sim)

        # Rank genes
        var_rank = np.argsort(np.argsort(-gene_var)) + 1
        lev_rank = np.argsort(np.argsort(-leverage)) + 1

        # Average rank of marker genes
        avg_var_rank = np.mean(var_rank[marker_genes_idx])
        avg_lev_rank = np.mean(lev_rank[marker_genes_idx])

        results.append({
            'abundance_dominant_pct': pct,
            'avg_var_rank_dominant': avg_var_rank,
            'avg_lev_rank_dominant': avg_lev_rank,
        })

        print(f"  {pct}%: Var rank = {avg_var_rank:.0f}, Lev rank = {avg_lev_rank:.0f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'experiment1_abundance_invariance.csv', index=False)
    print(f"  Saved: experiment1_abundance_invariance.csv")

    return df


def experiment2_gene_quadrant(X, gene_names, output_dir):
    """
    Categorize genes into quadrants by variance vs leverage.

    Quadrants:
    - GOLD: Low variance, high leverage (missed by HVG, captured by leverage)
    - NOISE: High variance, low leverage (technical noise, captured by HVG)
    - High/High: Important by both metrics
    - Low/Low: Not important by either metric
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Gene Quadrant Analysis")
    print("=" * 60)

    gene_var = np.var(X, axis=0)
    leverage = compute_leverage_scores(X)

    log_var = np.log10(gene_var + 1e-10)
    log_lev = np.log10(leverage + 1e-10)

    var_thresh = np.median(log_var)
    lev_thresh = np.median(log_lev)

    quadrants = []
    for g, (v, l) in enumerate(zip(log_var, log_lev)):
        if v < var_thresh and l > lev_thresh:
            q = 'Low Var / High Lev'
        elif v > var_thresh and l < lev_thresh:
            q = 'High Var / Low Lev'
        elif v > var_thresh and l > lev_thresh:
            q = 'High Var / High Lev'
        else:
            q = 'Low Var / Low Lev'
        quadrants.append({
            'gene': gene_names[g],
            'log_variance': v,
            'log_leverage': l,
            'quadrant': q,
        })

    df = pd.DataFrame(quadrants)
    df.to_csv(output_dir / 'experiment2_gene_quadrant_all.csv', index=False)

    # Save GOLD and NOISE gene lists for GO enrichment
    gold_genes = df[df['quadrant'] == 'Low Var / High Lev']['gene'].tolist()
    noise_genes = df[df['quadrant'] == 'High Var / Low Lev']['gene'].tolist()

    pd.DataFrame({'symbol': gold_genes}).to_csv(
        output_dir / 'experiment2_gold_genes_symbols.csv', index=False
    )
    pd.DataFrame({'symbol': noise_genes}).to_csv(
        output_dir / 'experiment2_noise_genes_symbols.csv', index=False
    )

    print(f"  GOLD genes (low var, high lev): {len(gold_genes)}")
    print(f"  NOISE genes (high var, low lev): {len(noise_genes)}")
    print(f"  Saved gene quadrant analysis and gene lists")

    # Print instructions for GO enrichment
    print("\n  To perform GO enrichment on GOLD genes:")
    print("  1. Visit https://maayanlab.cloud/Enrichr/")
    print("  2. Upload experiment2_gold_genes_symbols.csv")
    print("  3. Download 'GO_Biological_Process_2021' results")
    print("  4. Save to: results/leverage_deep_dive/enrichr_gold/")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Leverage Deep Dive Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing mouse brain data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / 'leverage_deep_dive'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'enrichr_gold').mkdir(exist_ok=True)

    print("=" * 60)
    print("LEVERAGE DEEP DIVE ANALYSIS")
    print("=" * 60)

    # Load data
    sc_adata = load_mouse_brain_data(args.data_dir)

    # Build reference
    X, cell_types, gene_names = build_reference_matrix(sc_adata)

    # Run experiments
    experiment1_abundance_invariance(X, cell_types, gene_names, output_dir)
    experiment2_gene_quadrant(X, gene_names, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
