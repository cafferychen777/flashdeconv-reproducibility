#!/usr/bin/env python
"""
Resolution Horizon Analysis on Visium HD Mouse Small Intestine.

This script demonstrates FlashDeconv's scale-space analysis capability,
showing how different cell types have characteristic spatial scales
and identifying the "resolution horizon" (8-16μm threshold).

Reproduces results from FlashDeconv paper Section 2.4 and Figure visium_hd.

Requirements:
    - FlashDeconv (pip install flashdeconv)
    - scanpy, pandas, numpy, matplotlib
    - Visium HD Mouse Small Intestine data (from 10x Genomics)
    - Haber et al. intestine scRNA-seq reference

Data download:
    # Visium HD data
    curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz
    tar -xzf Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz

    # Haber et al. reference (pre-processed)
    curl -L -o haber_processed.h5ad "https://zenodo.org/records/4447233/files/haber_processed.h5ad?download=1"

Usage:
    python resolution_horizon_analysis.py --data_dir ./data --output_dir ./results
    python resolution_horizon_analysis.py --bins 8,16,32,64  # Specific scales
    python resolution_horizon_analysis.py --bins 64,128      # Fast test

Author: FlashDeconv Team
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import sparse

import scanpy as sc
from flashdeconv import FlashDeconv
from flashdeconv.io.loader import prepare_data

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.linewidth': 0.5,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
})


def load_visium_hd_data(data_dir, bin_size='016um'):
    """
    Load Visium HD data at specified bin size.

    Parameters
    ----------
    data_dir : Path
        Directory containing Visium_HD_Mouse_Small_Intestine_binned_outputs/
    bin_size : str
        One of '002um', '008um', '016um'

    Returns
    -------
    AnnData
        Spatial transcriptomics data with coords in obsm['spatial']
    """
    import pyarrow.parquet as pq

    data_path = data_dir / 'Visium_HD_Mouse_Small_Intestine_binned_outputs'
    bin_path = data_path / f'square_{bin_size}'

    h5_path = bin_path / 'filtered_feature_bc_matrix.h5'

    print(f"  Loading Visium HD {bin_size} data...")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # Load spatial coordinates
    positions_path = bin_path / 'spatial' / 'tissue_positions.parquet'
    if positions_path.exists():
        positions = pq.read_table(positions_path).to_pandas()
        positions = positions.set_index('barcode')

        common_barcodes = adata.obs_names.intersection(positions.index)
        adata = adata[common_barcodes].copy()

        coords = positions.loc[common_barcodes, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
        adata.obsm['spatial'] = coords
    else:
        raise FileNotFoundError(f"Positions file not found: {positions_path}")

    print(f"    -> {adata.n_obs:,} spots, {adata.n_vars:,} genes")
    return adata


def load_reference_data(data_dir):
    """Load mouse intestine scRNA-seq reference."""
    # Try matched reference first, then fall back to original
    for ref_name in ['haber_intestine_matched.h5ad', 'haber_processed.h5ad']:
        ref_path = data_dir / ref_name
        if ref_path.exists():
            print(f"  Loading reference data ({ref_name})...")
            adata_ref = sc.read_h5ad(ref_path)
            adata_ref.var_names_make_unique()

            # Determine cell type column
            ct_col = 'celltype1' if 'celltype1' in adata_ref.obs else 'cell_type'

            print(f"    -> {adata_ref.n_obs:,} cells, {adata_ref.n_vars:,} genes")
            print(f"    -> {adata_ref.obs[ct_col].nunique()} cell types")
            return adata_ref, ct_col

    raise FileNotFoundError(
        f"Reference data not found in {data_dir}. "
        "Please download haber_processed.h5ad from Zenodo."
    )


def create_custom_bins(adata, target_bin_size, original_bin_size=16):
    """
    Create custom bin sizes by aggregating spots.

    Parameters
    ----------
    adata : AnnData
        Original data at base resolution
    target_bin_size : int
        Target bin size in μm
    original_bin_size : int
        Original bin size in μm (default 16)

    Returns
    -------
    AnnData
        Aggregated data at target resolution
    """
    scale_factor = target_bin_size // original_bin_size

    if scale_factor == 1:
        return adata.copy()

    coords = adata.obsm['spatial']

    print(f"    Creating {target_bin_size}μm bins (aggregating {scale_factor}x{scale_factor} spots)...")

    # Estimate pixel spacing from data
    tree = cKDTree(coords[:min(10000, len(coords))])
    distances, _ = tree.query(coords[:min(10000, len(coords))], k=2)
    pixel_spacing = np.median(distances[:, 1])

    grid_size_pixels = pixel_spacing * scale_factor

    # Assign each spot to a grid cell
    min_x, min_y = coords.min(axis=0)
    grid_x = ((coords[:, 0] - min_x) / grid_size_pixels).astype(int)
    grid_y = ((coords[:, 1] - min_y) / grid_size_pixels).astype(int)

    grid_ids = grid_x.astype(str) + '_' + grid_y.astype(str)
    unique_grids = np.unique(grid_ids)
    n_new_spots = len(unique_grids)

    print(f"      {adata.n_obs:,} spots -> {n_new_spots:,} bins")

    # Build aggregation matrix
    grid_to_idx = {g: i for i, g in enumerate(unique_grids)}
    row_indices = [grid_to_idx[g] for g in grid_ids]
    col_indices = np.arange(adata.n_obs)
    data = np.ones(adata.n_obs)

    agg_matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_new_spots, adata.n_obs)
    )

    # Sum counts
    if sparse.issparse(adata.X):
        new_X = agg_matrix @ adata.X
    else:
        new_X = agg_matrix @ sparse.csr_matrix(adata.X)

    # Compute new coordinates (centroid)
    new_coords = np.zeros((n_new_spots, 2))
    for i, g in enumerate(unique_grids):
        mask = grid_ids == g
        new_coords[i] = coords[mask].mean(axis=0)

    # Create new AnnData
    adata_new = sc.AnnData(X=new_X, var=adata.var.copy())
    adata_new.obs_names = pd.Index(unique_grids)
    adata_new.obsm['spatial'] = new_coords
    adata_new.uns['bin_size'] = target_bin_size

    return adata_new


def compute_morans_i(values, coords, k=10):
    """Compute Moran's I spatial autocorrelation."""
    n = len(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)

    y = values - values.mean()
    lag = np.array([y[indices[i, 1:]].mean() for i in range(n)])

    numerator = np.sum(y * lag)
    denominator = np.sum(y ** 2)

    morans_i = (n / (k * n)) * numerator / (denominator + 1e-10)
    return morans_i


def run_multiscale_analysis(data_dir, output_dir, bin_sizes):
    """
    Run FlashDeconv at multiple spatial scales.

    Parameters
    ----------
    data_dir : Path
        Directory containing data files
    output_dir : Path
        Directory for output files
    bin_sizes : list
        List of bin sizes in μm

    Returns
    -------
    dict
        Results at each scale
    """
    results = {}
    cell_types = None

    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load reference
    adata_ref, ct_col = load_reference_data(data_dir)

    # Load base resolution data (16μm is fastest)
    print(f"\n  Loading base spatial data (016um)...")
    adata_base = load_visium_hd_data(data_dir, '016um')
    base_bin_size = 16

    # Initialize FlashDeconv
    model = FlashDeconv(
        sketch_dim=512,
        lambda_spatial=5000.0,
        n_hvg=2000,
        n_markers_per_type=50,
        k_neighbors=6,
        max_iter=200,
        tol=1e-4,
        verbose=False,
        random_state=42,
    )

    print("\n" + "=" * 60)
    print("RUNNING MULTI-SCALE DECONVOLUTION")
    print("=" * 60)

    for scale_idx, bin_size in enumerate(bin_sizes):
        print(f"\n  [{scale_idx + 1}/{len(bin_sizes)}] Processing {bin_size}μm resolution")
        print("  " + "-" * 40)

        # Get data at this resolution
        if bin_size == base_bin_size:
            adata_st = adata_base.copy()
        elif bin_size < base_bin_size:
            print(f"    Skipping {bin_size}μm (smaller than base {base_bin_size}μm)")
            # Would need 8μm base data
            continue
        else:
            adata_st = create_custom_bins(adata_base, bin_size, base_bin_size)

        # Prepare data
        Y, X, coords, cell_type_names, gene_names = prepare_data(
            adata_st, adata_ref, cell_type_key=ct_col
        )
        print(f"    Y: {Y.shape}, X: {X.shape}, common genes: {len(gene_names)}")

        # Run deconvolution
        print(f"    Running FlashDeconv on {Y.shape[0]:,} spots...")
        start_time = time.time()

        try:
            proportions_raw = model.fit_transform(Y, X, coords)
            elapsed = time.time() - start_time

            proportions = pd.DataFrame(proportions_raw, columns=cell_type_names)

            if cell_types is None:
                cell_types = list(proportions.columns)

            results[bin_size] = {
                'proportions': proportions,
                'n_spots': Y.shape[0],
                'time': elapsed,
                'coords': coords.copy(),
            }

            print(f"    Completed in {elapsed:.2f}s ({Y.shape[0] / elapsed:,.0f} spots/sec)")

        except Exception as e:
            print(f"    Error: {e}")
            results[bin_size] = None

        gc.collect()

    return results, cell_types


def compute_resolution_metrics(results, cell_types):
    """Compute metrics to characterize resolution horizon."""
    print("\n" + "=" * 60)
    print("COMPUTING RESOLUTION METRICS")
    print("=" * 60)

    metrics = []

    for bin_size, res in results.items():
        if res is None:
            continue

        props = res['proportions']
        coords = res['coords']

        for cell_type in props.columns:
            p = props[cell_type].values
            p = np.clip(p, 1e-10, 1)

            # Signal metrics
            cv = np.std(p) / (np.mean(p) + 1e-10)
            max_prop = np.max(p)
            pct_detectable = (p > 0.01).mean() * 100  # >1%
            pct_high = (p > 0.1).mean() * 100  # >10%

            # Spatial autocorrelation
            morans_i = compute_morans_i(p, coords, k=min(10, len(p) - 1))

            metrics.append({
                'bin_size': bin_size,
                'cell_type': cell_type,
                'cv': cv,
                'max_prop': max_prop,
                'pct_detectable': pct_detectable,
                'pct_high': pct_high,
                'morans_i': morans_i,
                'mean_prop': np.mean(p),
            })

    return pd.DataFrame(metrics)


def create_resolution_figure(results, metrics_df, output_dir):
    """Create publication figure showing resolution horizon."""
    print("\n  Generating figure...")

    fig = plt.figure(figsize=(7, 6))

    # Panel A: Spatial maps at different scales
    bin_sizes_to_show = sorted([bs for bs in results.keys() if results[bs] is not None])[:4]

    # Use brush cell (Tuft) to show rare cell type signal loss
    rare_ct = None
    for ct in ['brush cell', 'Tuft', 'enteroendocrine cell']:
        if ct in results[bin_sizes_to_show[0]]['proportions'].columns:
            rare_ct = ct
            break

    if rare_ct:
        for i, bin_size in enumerate(bin_sizes_to_show):
            ax = fig.add_subplot(2, len(bin_sizes_to_show), i + 1)
            res = results[bin_size]
            props = res['proportions'][rare_ct].values
            coords = res['coords']

            sc_plot = ax.scatter(coords[:, 0], coords[:, 1],
                                 c=props, cmap='Reds', s=max(0.5, 5 - i),
                                 vmin=0, vmax=0.3, alpha=0.8, rasterized=True)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'{bin_size}μm', fontsize=9, fontweight='bold')

    # Panel B: Resolution sensitivity for rare cell types
    ax_b = fig.add_subplot(2, 2, 3)

    rare_types = ['brush cell', 'enteroendocrine cell']
    colors = ['#e74c3c', '#3498db']

    for ct, color in zip(rare_types, colors):
        if ct not in metrics_df['cell_type'].values:
            continue
        subset = metrics_df[metrics_df['cell_type'] == ct].sort_values('bin_size')
        if len(subset) > 1:
            ax_b.plot(subset['bin_size'], subset['max_prop'] * 100,
                      'o-', label=ct.replace(' cell', ''), color=color,
                      markersize=5, linewidth=1.5)

    ax_b.set_xlabel('Resolution (μm)')
    ax_b.set_ylabel('Max proportion (%)')
    ax_b.set_xscale('log', base=2)
    ax_b.legend(frameon=False, fontsize=7)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Panel C: Moran's I across scales
    ax_c = fig.add_subplot(2, 2, 4)

    top_cts = metrics_df.groupby('cell_type')['mean_prop'].mean().nlargest(4).index
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_cts)))

    for ct, color in zip(top_cts, colors):
        subset = metrics_df[metrics_df['cell_type'] == ct].sort_values('bin_size')
        if len(subset) > 1:
            ax_c.plot(subset['bin_size'], subset['morans_i'],
                      'o-', label=ct.replace(' cell', '')[:15],
                      color=color, markersize=5, linewidth=1.5)

    ax_c.set_xlabel('Resolution (μm)')
    ax_c.set_ylabel("Moran's I")
    ax_c.set_xscale('log', base=2)
    ax_c.legend(frameon=False, fontsize=6)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'resolution_horizon_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'resolution_horizon_figure.pdf', bbox_inches='tight')
    plt.close()

    print(f"    Saved to: {output_dir / 'resolution_horizon_figure.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Resolution Horizon Analysis on Visium HD',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output files')
    parser.add_argument('--bins', type=str, default='16,32,64,128',
                        help='Comma-separated bin sizes in μm')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_sizes = [int(x.strip()) for x in args.bins.split(',')]
    bin_sizes = sorted(bin_sizes)

    print("=" * 60)
    print("RESOLUTION HORIZON ANALYSIS")
    print("FlashDeconv on Visium HD Mouse Small Intestine")
    print("=" * 60)
    print(f"\nBin sizes: {bin_sizes} μm")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Run analysis
    results, cell_types = run_multiscale_analysis(data_dir, output_dir, bin_sizes)

    # Check results
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("\nERROR: No successful deconvolution runs!")
        return

    # Compute metrics
    metrics_df = compute_resolution_metrics(results, cell_types)
    metrics_df.to_csv(output_dir / 'resolution_metrics.csv', index=False)

    # Create figure
    create_resolution_figure(results, metrics_df, output_dir)

    # Save proportions
    all_data = []
    for bin_size, res in results.items():
        if res is None:
            continue
        props = res['proportions']
        coords = res['coords']
        for i in range(len(props)):
            row = {'bin_size': bin_size, 'coord_x': coords[i, 0], 'coord_y': coords[i, 1]}
            for ct in props.columns:
                row[ct] = props.iloc[i][ct]
            all_data.append(row)

    if all_data:
        pd.DataFrame(all_data).to_csv(output_dir / 'multiscale_proportions.csv', index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nResolution horizon analysis for rare cell types:")
    for rare_ct in ['brush cell', 'enteroendocrine cell']:
        if rare_ct not in metrics_df['cell_type'].values:
            continue
        subset = metrics_df[metrics_df['cell_type'] == rare_ct].sort_values('bin_size')
        print(f"\n  {rare_ct}:")
        for _, row in subset.iterrows():
            print(f"    {row['bin_size']:3d}μm: max={row['max_prop']*100:.1f}%, "
                  f"detectable={row['pct_detectable']:.1f}%")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
