#!/usr/bin/env python
"""
Cortex Lamination Deconvolution
================================

Run FlashDeconv on Cell2location's paired mouse brain Visium data
to demonstrate cortical layer organization.

Requirements:
    - FlashDeconv (pip install flashdeconv)
    - Cell2location mouse brain data (scRNA-seq reference + Visium ST8059048)

Data download:
    See README.md Part 3 for download instructions.

Usage:
    python cortex_deconvolution.py --data_dir ./data --output_dir ./results

Output:
    - level2_v3_data.npz (for figure generation)
    - level2_v3_proportions.csv
    - level2_v3_correlations.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr
from pathlib import Path
from collections import Counter

try:
    from flashdeconv import FlashDeconv
except ImportError:
    print("FlashDeconv not installed. Run: pip install flashdeconv")
    raise


def load_paired_data(data_dir):
    """Load Cell2location's paired Visium and scRNA-seq data."""
    data_path = Path(data_dir) / 'mouse_brain'

    print("=" * 60)
    print("Loading Cell2location PAIRED Mouse Brain Data")
    print("=" * 60)

    # Load scRNA-seq reference
    print("\n1. Loading scRNA-seq reference...")
    sc_adata = sc.read_h5ad(data_path / 'scrna_reference.h5ad')
    anno = pd.read_csv(data_path / 'cell_annotation.csv', index_col=0)

    # Subset to cells with annotation
    common_cells = list(set(sc_adata.obs_names) & set(anno.index))
    sc_adata = sc_adata[common_cells].copy()
    sc_adata.obs['cell_type'] = anno.loc[sc_adata.obs_names, 'annotation_1'].values
    print(f"   scRNA: {sc_adata.shape}, {sc_adata.obs['cell_type'].nunique()} cell types")

    # Convert Ensembl to Symbol
    print("\n2. Converting gene IDs to symbols...")
    symbols = sc_adata.var['SYMBOL'].astype(str).values
    has_symbol = (symbols != '') & (symbols != 'nan') & (symbols != 'None')
    sc_adata = sc_adata[:, has_symbol].copy()

    # Make unique names
    new_names = symbols[has_symbol]
    name_counts = Counter()
    unique_names = []
    for name in new_names:
        if name_counts[name] > 0:
            unique_names.append(f"{name}-{name_counts[name]}")
        else:
            unique_names.append(name)
        name_counts[name] += 1
    sc_adata.var_names = pd.Index(unique_names)
    print(f"   After symbol conversion: {sc_adata.shape}")

    # Load PAIRED Visium spatial data (ST8059048)
    print("\n3. Loading PAIRED Visium spatial data (ST8059048)...")
    st_path = data_path / 'C2L' / 'ST' / '48'

    sp_adata = sc.read_10x_h5(st_path / 'ST8059048_filtered_feature_bc_matrix.h5')
    sp_adata.var_names_make_unique()

    # Load spatial coordinates
    coords_df = pd.read_csv(
        st_path / 'spatial' / 'tissue_positions_list.csv',
        header=None,
        index_col=0
    )
    coords_df.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']

    # Filter to in_tissue spots
    in_tissue = coords_df[coords_df['in_tissue'] == 1].index
    common_spots = list(set(sp_adata.obs_names) & set(in_tissue))
    sp_adata = sp_adata[common_spots].copy()

    # Add spatial coordinates
    sp_adata.obsm['spatial'] = coords_df.loc[sp_adata.obs_names, ['pxl_row', 'pxl_col']].values
    print(f"   Spatial: {sp_adata.shape}, {len(common_spots)} spots in tissue")

    # Find common genes
    common_genes = list(set(sc_adata.var_names) & set(sp_adata.var_names))
    print(f"   Common genes: {len(common_genes)}")

    # Subset to common genes
    sc_adata = sc_adata[:, common_genes].copy()
    sp_adata = sp_adata[:, common_genes].copy()

    return sc_adata, sp_adata, common_genes


def create_reference_signatures(sc_adata):
    """Create cell type reference signatures from scRNA-seq data."""
    print("\n4. Creating reference signatures...")

    cell_types = sc_adata.obs['cell_type'].unique()
    n_types = len(cell_types)
    n_genes = sc_adata.n_vars

    X_ref = np.zeros((n_types, n_genes))

    for i, ct in enumerate(cell_types):
        mask = (sc_adata.obs['cell_type'] == ct).values
        if issparse(sc_adata.X):
            X_ref[i] = sc_adata.X[mask].toarray().mean(axis=0)
        else:
            X_ref[i] = sc_adata.X[mask].mean(axis=0)

    print(f"   Reference shape: {X_ref.shape}")
    print(f"   Cell types: {n_types}")

    return X_ref, list(cell_types)


def run_flashdeconv(sp_adata, X_ref, cell_types):
    """Run FlashDeconv deconvolution."""
    print("\n" + "=" * 60)
    print("Running FlashDeconv")
    print("=" * 60)

    if issparse(sp_adata.X):
        Y = sp_adata.X.toarray()
    else:
        Y = np.array(sp_adata.X)

    coords = sp_adata.obsm['spatial']

    t0 = time.time()
    model = FlashDeconv(
        sketch_dim=512,
        lambda_spatial=5000.0,
        rho_sparsity=0.01,
        preprocess="log_cpm",
        n_hvg=2000,
        max_iter=100,
        verbose=True,
        random_state=42,
    )

    proportions = model.fit_transform(Y, X_ref, coords)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")

    prop_df = pd.DataFrame(
        proportions,
        index=sp_adata.obs_names,
        columns=cell_types
    )

    return prop_df, elapsed


def evaluate_with_markers(sp_adata, prop_df, sc_adata):
    """Evaluate predictions using marker gene correlation."""
    print("\n" + "=" * 60)
    print("Evaluating Predictions (Marker Gene Correlation)")
    print("=" * 60)

    sc_adata_eval = sc_adata.copy()
    sc.pp.normalize_total(sc_adata_eval, target_sum=1e4)
    sc.pp.log1p(sc_adata_eval)

    try:
        sc.tl.rank_genes_groups(sc_adata_eval, 'cell_type', method='wilcoxon', n_genes=20)
        markers_available = True
    except Exception as e:
        print(f"Marker finding failed: {e}")
        markers_available = False

    results = []

    if markers_available:
        for ct in prop_df.columns:
            try:
                markers = sc_adata_eval.uns['rank_genes_groups']['names'][ct][:10]
                markers = [g for g in markers if g in sp_adata.var_names]

                if len(markers) < 2:
                    continue

                marker_idx = [list(sp_adata.var_names).index(g) for g in markers]
                if issparse(sp_adata.X):
                    marker_expr = sp_adata.X[:, marker_idx].toarray().mean(axis=1)
                else:
                    marker_expr = sp_adata.X[:, marker_idx].mean(axis=1)

                pred = prop_df[ct].values
                r, p = pearsonr(marker_expr.flatten(), pred.flatten())

                results.append({
                    'cell_type': ct,
                    'correlation': r,
                    'p_value': p,
                    'n_markers': len(markers)
                })
            except Exception:
                continue

    if results:
        results_df = pd.DataFrame(results).sort_values('correlation', ascending=False)
        print(f"\nTop 15 cell types by marker correlation:")
        print(results_df.head(15).to_string(index=False))
        return results_df
    else:
        print("Could not compute marker correlations")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run FlashDeconv on Cell2location paired mouse brain data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing mouse brain data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PAIRED data
    sc_adata, sp_adata, common_genes = load_paired_data(args.data_dir)

    # Create reference
    X_ref, cell_types = create_reference_signatures(sc_adata)

    # Run FlashDeconv
    prop_df, elapsed = run_flashdeconv(sp_adata, X_ref, cell_types)

    # Evaluate
    results_df = evaluate_with_markers(sp_adata, prop_df, sc_adata)

    # Save results
    prop_df.to_csv(output_dir / 'level2_v3_proportions.csv')
    if results_df is not None:
        results_df.to_csv(output_dir / 'level2_v3_correlations.csv', index=False)

    # Save npz for figure generation
    np.savez(
        output_dir / 'level2_v3_data.npz',
        proportions=prop_df.values,
        coordinates=sp_adata.obsm['spatial'],
        spot_names=np.array(prop_df.index.tolist(), dtype=object),
        cell_types=np.array(cell_types, dtype=object)
    )
    print(f"\nSaved: {output_dir / 'level2_v3_data.npz'}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)

    if results_df is not None:
        mean_r = results_df['correlation'].mean()
        top_r = results_df.head(10)['correlation'].mean()
        print(f"Mean marker correlation (all types): {mean_r:.3f}")
        print(f"Mean marker correlation (top 10):    {top_r:.3f}")
        print(f"Time: {elapsed:.1f}s")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
