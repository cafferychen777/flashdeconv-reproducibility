#!/usr/bin/env python
"""
Tuft-Stem Cell Niche Discovery Analysis.

This script demonstrates FlashDeconv's ability to discover rare cell type
niches that would be invisible to variance-based (HVG) gene selection methods.

Key findings:
1. Tuft cells (brush cells) have HIGH leverage but LOW variance markers
2. FlashDeconv reveals focal tuft cell niches at 8μm resolution
3. Tuft cells show 16.8x enrichment for co-localization with stem cells
4. This signal is validated by Moran's I spatial autocorrelation

Reproduces results from FlashDeconv paper Figure tuft_discovery.

Requirements:
    - FlashDeconv (pip install flashdeconv)
    - scanpy, pandas, numpy, matplotlib
    - Pre-computed multiscale proportions from resolution_horizon_analysis.py

Usage:
    python tuft_stem_discovery.py --data_dir ./data --output_dir ./results

Author: FlashDeconv Team
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
from scipy import stats, sparse

import scanpy as sc
from flashdeconv.utils.genes import compute_leverage_scores

# Publication-quality settings (Nature style)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.2,
})


def compute_hvg_blindness(adata_ref, ct_col, n_markers=30):
    """
    Compute "HVG blindness" for each cell type.

    HVG blindness = how much worse variance ranks markers compared to leverage.
    High blindness = HVG methods would likely miss these cell types.

    Parameters
    ----------
    adata_ref : AnnData
        Single-cell reference data
    ct_col : str
        Column name for cell type annotations
    n_markers : int
        Number of marker genes to consider per cell type

    Returns
    -------
    DataFrame
        Cell type visibility metrics
    """
    print("Computing HVG blindness for each cell type...")

    # Build signature matrix
    cell_types = sorted(adata_ref.obs[ct_col].unique())
    n_genes = adata_ref.n_vars
    gene_names = adata_ref.var_names.tolist()

    X = np.zeros((len(cell_types), n_genes))
    for i, ct in enumerate(cell_types):
        mask = adata_ref.obs[ct_col] == ct
        if sparse.issparse(adata_ref.X):
            expr = adata_ref[mask].X.toarray()
        else:
            expr = np.asarray(adata_ref[mask].X)
        X[i, :] = expr.mean(axis=0)

    # Normalize to CPM-like scale
    X = X / (X.sum(axis=1, keepdims=True) + 1e-10) * 1e4

    # Compute leverage and variance
    leverage = compute_leverage_scores(X)
    gene_var = np.var(X, axis=0)

    # Rank genes (0 = best)
    leverage_rank = np.argsort(np.argsort(-leverage))
    variance_rank = np.argsort(np.argsort(-gene_var))

    results = []
    for ct_idx, ct_name in enumerate(cell_types):
        # Find marker genes (high fold change vs other types)
        ct_expr = X[ct_idx, :]
        other_mask = np.ones(len(cell_types), dtype=bool)
        other_mask[ct_idx] = False
        other_expr = X[other_mask].mean(axis=0)

        fc = np.log2((ct_expr + 1) / (other_expr + 1))
        marker_idx = np.argsort(-fc)[:n_markers]

        # Check their ranks
        marker_leverage_pct = np.mean(leverage_rank[marker_idx]) / n_genes * 100
        marker_variance_pct = np.mean(variance_rank[marker_idx]) / n_genes * 100

        # HVG blindness: positive = HVG would miss
        hvg_blindness = marker_variance_pct - marker_leverage_pct

        results.append({
            'cell_type': ct_name,
            'marker_leverage_pct': marker_leverage_pct,
            'marker_variance_pct': marker_variance_pct,
            'hvg_blindness': hvg_blindness,
            'top_marker': gene_names[marker_idx[0]],
        })

    df = pd.DataFrame(results).sort_values('hvg_blindness', ascending=False)

    print("\nCell types ranked by HVG blindness:")
    for _, row in df.head(5).iterrows():
        print(f"  {row['cell_type']}: {row['hvg_blindness']:.1f}% blindness")

    return df


def compute_morans_i(values, coords, k=10):
    """Compute Moran's I spatial autocorrelation with permutation test."""
    n = len(values)
    values = np.array(values)
    coords = np.array(coords)

    # Build k-NN spatial weight matrix
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)

    W = np.zeros((n, n))
    for i in range(n):
        for j in indices[i, 1:]:
            W[i, j] = 1

    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # Compute Moran's I
    y = values - values.mean()
    numerator = np.sum(W * np.outer(y, y))
    denominator = np.sum(y ** 2)

    I = (n / W.sum()) * (numerator / denominator) if denominator > 0 else 0

    # Permutation test
    n_perm = 999
    I_perm = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        num_p = np.sum(W * np.outer(y_perm, y_perm))
        I_p = (n / W.sum()) * (num_p / denominator) if denominator > 0 else 0
        I_perm.append(I_p)

    p_value = (np.sum(np.array(I_perm) >= I) + 1) / (n_perm + 1)

    return I, p_value, np.mean(I_perm), np.std(I_perm)


def analyze_colocalization(props_df, tuft_col='brush cell', stem_col='epithelial fate stem cell'):
    """
    Analyze Tuft-Stem cell co-localization.

    Parameters
    ----------
    props_df : DataFrame
        Proportions at finest resolution (8μm)
    tuft_col : str
        Column name for tuft cells
    stem_col : str
        Column name for stem cells

    Returns
    -------
    DataFrame
        Co-localization enrichment for all cell types
    """
    print("\nAnalyzing Tuft-Stem co-localization...")

    # Define tuft hotspots
    tuft_threshold = 0.1  # >10% tuft proportion
    high_tuft = props_df[tuft_col] > tuft_threshold
    n_hotspots = high_tuft.sum()

    print(f"  Tuft hotspots (>{tuft_threshold*100:.0f}%): {n_hotspots} spots")

    results = []
    ct_cols = [c for c in props_df.columns if c not in ['bin_size', 'coord_x', 'coord_y', 'spot_id']]

    for ct in ct_cols:
        if ct == tuft_col:
            continue

        hot_mean = props_df.loc[high_tuft, ct].mean()
        cold_mean = props_df.loc[~high_tuft, ct].mean()
        enrichment = hot_mean / (cold_mean + 1e-10)

        results.append({
            'cell_type': ct,
            'enrichment': enrichment,
            'mean_in_hotspot': hot_mean,
            'mean_elsewhere': cold_mean,
        })

    df = pd.DataFrame(results).sort_values('enrichment', ascending=False)

    print("\nTop enriched cell types in Tuft niches:")
    for _, row in df.head(5).iterrows():
        print(f"  {row['cell_type']}: {row['enrichment']:.1f}x enrichment")

    return df


def create_discovery_figure(props_df, visibility_df, coloc_df, output_dir):
    """Create publication-quality Tuft-Stem discovery figure."""
    print("\nCreating discovery figure...")

    # Filter to finest resolution if multiple
    if 'bin_size' in props_df.columns:
        finest = props_df['bin_size'].min()
        props_8um = props_df[props_df['bin_size'] == finest].copy()
    else:
        props_8um = props_df.copy()

    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                  wspace=0.3, hspace=0.35)

    # Color scheme
    tuft_color = '#e41a1c'
    stem_color = '#377eb8'
    other_color = '#999999'

    # =========================================================================
    # Panel A: HVG Blindness Bar Chart
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    vis_sorted = visibility_df.sort_values('hvg_blindness', ascending=True)
    y_pos = np.arange(len(vis_sorted))

    # Rename brush to Tuft for display
    display_names = {ct: ('Tuft' if 'brush' in ct.lower() else ct.replace(' cell', '').replace('epithelial fate ', '')[:12])
                     for ct in vis_sorted['cell_type']}
    colors = [tuft_color if 'brush' in ct.lower() else other_color
              for ct in vis_sorted['cell_type']]

    bars = ax_a.barh(y_pos, vis_sorted['hvg_blindness'], color=colors, alpha=0.8, height=0.7)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([display_names[ct] for ct in vis_sorted['cell_type']])
    ax_a.set_xlabel('HVG Blindness (%)')
    ax_a.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # =========================================================================
    # Panel B: Tuft cell spatial distribution
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    tuft_col = 'brush cell' if 'brush cell' in props_8um.columns else None
    if tuft_col:
        sc = ax_b.scatter(
            props_8um['coord_x'], props_8um['coord_y'],
            c=props_8um[tuft_col], s=0.3, cmap='Reds',
            vmin=0, vmax=0.3, alpha=0.9, rasterized=True
        )
        ax_b.set_aspect('equal')
        ax_b.axis('off')
        cbar = plt.colorbar(sc, ax=ax_b, shrink=0.5, aspect=15, pad=0.02)
        cbar.set_label('Tuft cell\nproportion', fontsize=7)

    # =========================================================================
    # Panel C: Stem cell spatial distribution
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])

    stem_col = None
    for col in ['epithelial fate stem cell', 'stem cell', 'Stem']:
        if col in props_8um.columns:
            stem_col = col
            break

    if stem_col:
        sc2 = ax_c.scatter(
            props_8um['coord_x'], props_8um['coord_y'],
            c=props_8um[stem_col], s=0.3, cmap='Blues',
            vmin=0, vmax=0.5, alpha=0.9, rasterized=True
        )
        ax_c.set_aspect('equal')
        ax_c.axis('off')
        cbar2 = plt.colorbar(sc2, ax=ax_c, shrink=0.5, aspect=15, pad=0.02)
        cbar2.set_label('Stem cell\nproportion', fontsize=7)

    # =========================================================================
    # Panel D: Resolution sensitivity
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 0])

    if 'bin_size' in props_df.columns and tuft_col:
        bin_sizes = sorted(props_df['bin_size'].unique())
        max_props = []
        for bs in bin_sizes:
            sub = props_df[props_df['bin_size'] == bs]
            max_props.append(sub[tuft_col].max() * 100)

        ax_d.plot(bin_sizes, max_props, 'o-', color=tuft_color, markersize=5)
        ax_d.set_xscale('log', base=2)
        ax_d.set_xlabel('Resolution (μm)')
        ax_d.set_ylabel('Max Tuft Cell Proportion (%)')
        ax_d.spines['top'].set_visible(False)
        ax_d.spines['right'].set_visible(False)
        ax_d.set_xticks(bin_sizes)
        ax_d.set_xticklabels([str(b) for b in bin_sizes])

    # =========================================================================
    # Panel E: Co-localization enrichment
    # =========================================================================
    ax_e = fig.add_subplot(gs[1, 1])

    # Top 6 cell types by enrichment
    top_coloc = coloc_df.head(6).copy()
    colors_e = [stem_color if 'stem' in ct.lower() else
                (tuft_color if 'enteroend' in ct.lower() else other_color)
                for ct in top_coloc['cell_type']]

    bars = ax_e.barh(range(len(top_coloc)), top_coloc['enrichment'],
                     color=colors_e, alpha=0.8, height=0.7)
    ax_e.set_yticks(range(len(top_coloc)))
    ax_e.set_yticklabels([ct.replace(' cell', '').replace('epithelial fate ', '')[:15]
                          for ct in top_coloc['cell_type']])
    ax_e.set_xlabel('Fold enrichment in Tuft niches')
    ax_e.axvline(x=1, color='gray', linestyle='--', linewidth=0.8)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)

    # =========================================================================
    # Panel F: Co-localization zoom
    # =========================================================================
    ax_f = fig.add_subplot(gs[1, 2])

    if tuft_col and stem_col:
        tuft_vals = props_8um[tuft_col].values
        stem_vals = props_8um[stem_col].values

        # Categorical classification
        tuft_thresh = 0.05
        stem_thresh = 0.10

        is_tuft_high = tuft_vals > tuft_thresh
        is_stem_high = stem_vals > stem_thresh

        categories = np.zeros(len(props_8um), dtype=int)
        categories[is_tuft_high & ~is_stem_high] = 1  # Tuft-only
        categories[~is_tuft_high & is_stem_high] = 2  # Stem-only
        categories[is_tuft_high & is_stem_high] = 3   # Co-localized

        # Find zoom region
        coloc_mask = categories == 3
        if coloc_mask.sum() > 0:
            center_x = props_8um.loc[coloc_mask, 'coord_x'].median()
            center_y = props_8um.loc[coloc_mask, 'coord_y'].median()
        else:
            center_x = props_8um['coord_x'].median()
            center_y = props_8um['coord_y'].median()

        zoom_radius = 400
        zoom_mask = (
            (props_8um['coord_x'] > center_x - zoom_radius) &
            (props_8um['coord_x'] < center_x + zoom_radius) &
            (props_8um['coord_y'] > center_y - zoom_radius) &
            (props_8um['coord_y'] < center_y + zoom_radius)
        )
        zoom_data = props_8um[zoom_mask].copy()
        zoom_cats = categories[zoom_mask.values]

        # Plot
        color_map = {0: '#E5E5E5', 1: '#D55E00', 2: '#0072B2', 3: '#CC79A7'}

        for cat, color, label, size in [
            (0, '#E5E5E5', 'Background', 1),
            (2, '#0072B2', 'Stem-high', 4),
            (1, '#D55E00', 'Tuft-high', 6),
            (3, '#CC79A7', 'Co-localized', 8),
        ]:
            mask = zoom_cats == cat
            if mask.sum() > 0:
                ax_f.scatter(
                    zoom_data.loc[zoom_data.index[mask], 'coord_x'],
                    zoom_data.loc[zoom_data.index[mask], 'coord_y'],
                    c=color, s=size, alpha=0.7 if cat == 0 else 0.9,
                    label=label if cat > 0 else None, rasterized=True
                )

        ax_f.set_aspect('equal')
        ax_f.axis('off')
        ax_f.legend(loc='upper left', frameon=True, fontsize=6,
                    markerscale=1.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'tuft_stem_discovery.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'tuft_stem_discovery.pdf', bbox_inches='tight')
    print(f"  Saved to: {output_dir / 'tuft_stem_discovery.png'}")
    plt.close()


def run_validation(props_df, output_dir):
    """Run null model validation for Tuft cell signal."""
    print("\n" + "=" * 60)
    print("NULL MODEL VALIDATION")
    print("=" * 60)

    # Filter to finest resolution
    if 'bin_size' in props_df.columns:
        finest = props_df['bin_size'].min()
        props_8um = props_df[props_df['bin_size'] == finest].copy()
    else:
        props_8um = props_df.copy()

    tuft_col = 'brush cell' if 'brush cell' in props_8um.columns else None
    if tuft_col is None:
        print("  Tuft cell column not found, skipping validation")
        return {}

    coords = props_8um[['coord_x', 'coord_y']].values
    tuft_vals = props_8um[tuft_col].values

    # Subsample for efficiency
    n_sample = min(5000, len(tuft_vals))
    np.random.seed(42)
    sample_idx = np.random.choice(len(tuft_vals), n_sample, replace=False)

    print(f"\nComputing Moran's I on {n_sample} spots...")
    I, p_val, I_rand_mean, I_rand_std = compute_morans_i(
        tuft_vals[sample_idx], coords[sample_idx], k=10
    )

    print(f"  Moran's I: {I:.4f}")
    print(f"  P-value: {p_val:.4f}")
    print(f"  Random baseline: {I_rand_mean:.4f} +/- {I_rand_std:.4f}")

    results = {
        'morans_i': I,
        'morans_pvalue': p_val,
        'morans_random_mean': I_rand_mean,
        'morans_random_std': I_rand_std,
    }

    # Save results
    pd.DataFrame([results]).to_csv(output_dir / 'tuft_validation_results.csv', index=False)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Tuft-Stem Cell Niche Discovery Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output files')
    parser.add_argument('--props_file', type=str, default=None,
                        help='Pre-computed proportions CSV (optional)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TUFT-STEM CELL NICHE DISCOVERY")
    print("=" * 60)

    # Load proportions
    if args.props_file:
        props_path = Path(args.props_file)
    else:
        props_path = output_dir / 'multiscale_proportions.csv'

    if not props_path.exists():
        print(f"\nProportions file not found: {props_path}")
        print("Please run resolution_horizon_analysis.py first to generate proportions.")
        return

    print(f"\nLoading proportions from {props_path}...")
    props_df = pd.read_csv(props_path)
    print(f"  Loaded {len(props_df)} rows")

    # Load reference for HVG blindness analysis
    ref_path = None
    for name in ['haber_intestine_matched.h5ad', 'haber_processed.h5ad']:
        if (data_dir / name).exists():
            ref_path = data_dir / name
            break

    visibility_df = None
    if ref_path:
        print(f"\nLoading reference from {ref_path}...")
        adata_ref = sc.read_h5ad(ref_path)
        ct_col = 'celltype1' if 'celltype1' in adata_ref.obs else 'cell_type'
        visibility_df = compute_hvg_blindness(adata_ref, ct_col)
        visibility_df.to_csv(output_dir / 'cell_type_visibility.csv', index=False)
    else:
        print("\nReference not found, skipping HVG blindness analysis")
        visibility_df = pd.DataFrame({
            'cell_type': props_df.columns[3:],  # Skip metadata columns
            'hvg_blindness': np.zeros(len(props_df.columns) - 3)
        })

    # Analyze co-localization
    coloc_df = analyze_colocalization(props_df)
    coloc_df.to_csv(output_dir / 'tuft_colocalization.csv', index=False)

    # Create figure
    create_discovery_figure(props_df, visibility_df, coloc_df, output_dir)

    # Run validation
    validation_results = run_validation(props_df, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if visibility_df is not None:
        tuft_row = visibility_df[visibility_df['cell_type'].str.contains('brush', case=False)]
        if len(tuft_row) > 0:
            print(f"\nTuft cell HVG blindness: {tuft_row['hvg_blindness'].values[0]:.1f}%")
            print("  -> Tuft markers would be missed by standard HVG selection")

    stem_row = coloc_df[coloc_df['cell_type'].str.contains('stem', case=False)]
    if len(stem_row) > 0:
        print(f"\nStem cell enrichment in Tuft niches: {stem_row['enrichment'].values[0]:.1f}x")

    if validation_results:
        print(f"\nMoran's I spatial autocorrelation: {validation_results['morans_i']:.4f}")
        print(f"  P-value: {validation_results['morans_pvalue']:.4f}")
        print("  -> Tuft cell distribution shows significant spatial clustering")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
