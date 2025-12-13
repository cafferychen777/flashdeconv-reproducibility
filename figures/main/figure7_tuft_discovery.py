#!/usr/bin/env python
"""
Figure 7: Tuft-Stem Cell Niche Discovery
=========================================

This script generates Figure 7, demonstrating FlashDeconv's ability to discover
rare cell type niches that would be missed by HVG-based gene selection.

Panels:
- A: HVG Blindness bar chart - shows which cell types' markers would be missed
- B: Tuft cell spatial distribution at 8um resolution
- C: Stem cell spatial distribution at 8um resolution
- D: Resolution sensitivity - max tuft proportion across scales
- E: Co-localization enrichment - fold enrichment of other cell types in tuft niches
- F: Zoom into tuft-stem co-localization region

Requirements:
    Run analysis/multiscale_analysis.py first to generate:
    - multiscale_proportions.csv
    - cell_type_visibility.csv

Usage:
    python figure7_tuft_discovery.py --results_dir ./results --output_dir ./figures
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Style for Nature
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_data(data_dir):
    """Load multiscale analysis results."""
    data_path = Path(data_dir)

    # Proportions from multiscale analysis
    props_path = data_path / 'multiscale_proportions.csv'
    if not props_path.exists():
        raise FileNotFoundError(
            f"Proportions file not found: {props_path}\n"
            "Run analysis/multiscale_analysis.py first."
        )
    props = pd.read_csv(props_path)

    # Visibility analysis
    vis_path = data_path / 'cell_type_visibility.csv'
    if not vis_path.exists():
        raise FileNotFoundError(
            f"Visibility file not found: {vis_path}\n"
            "Run analysis/hidden_cell_analysis.py first."
        )
    visibility = pd.read_csv(vis_path)

    return props, visibility


def create_figure(props, visibility, output_dir):
    """Create the tuft-stem discovery figure."""

    fig = plt.figure(figsize=(7, 6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                  wspace=0.3, hspace=0.35)

    tuft_color = '#e41a1c'
    stem_color = '#377eb8'
    other_color = '#999999'

    # Panel A: HVG Blindness
    ax_a = fig.add_subplot(gs[0, 0])

    vis_sorted = visibility.sort_values('hvg_blindness', ascending=True)
    y_pos = np.arange(len(vis_sorted))

    display_names = {ct: ('Tuft' if 'brush' in ct.lower() else
                          ct.replace(' cell', '').replace('epithelial fate ', ''))
                     for ct in vis_sorted['cell_type']}
    colors = [tuft_color if 'brush' in ct.lower() else other_color
              for ct in vis_sorted['cell_type']]

    bars = ax_a.barh(y_pos, vis_sorted['hvg_blindness'], color=colors, alpha=0.8, height=0.7)

    for bar, ct in zip(bars, vis_sorted['cell_type']):
        if 'brush' in ct.lower():
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([display_names[ct] for ct in vis_sorted['cell_type']])
    ax_a.set_xlabel('HVG Blindness (%)')
    ax_a.text(-0.2, 1.02, 'a', transform=ax_a.transAxes, fontsize=10, fontweight='bold')

    # Panel B: Tuft cell spatial distribution
    ax_b = fig.add_subplot(gs[0, 1])

    subset_8 = props[props['bin_size'] == 8].copy() if 8 in props['bin_size'].values else props.copy()

    if 'brush cell' in subset_8.columns:
        sc = ax_b.scatter(
            subset_8['coord_x'],
            subset_8['coord_y'],
            c=subset_8['brush cell'],
            s=0.3,
            cmap='Reds',
            vmin=0,
            vmax=0.3,
            alpha=0.9,
            rasterized=True
        )
        cbar = plt.colorbar(sc, ax=ax_b, shrink=0.5, aspect=15, pad=0.02)
        cbar.set_label('Tuft cell\nproportion', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    ax_b.set_aspect('equal')
    ax_b.axis('off')
    ax_b.text(-0.05, 1.02, 'b', transform=ax_b.transAxes, fontsize=10, fontweight='bold')
    ax_b.text(0.5, -0.05, 'Tuft cell distribution (8um)',
              transform=ax_b.transAxes, ha='center', fontsize=8)

    # Panel C: Stem cell distribution
    ax_c = fig.add_subplot(gs[0, 2])

    stem_col = 'epithelial fate stem cell' if 'epithelial fate stem cell' in subset_8.columns else None
    if stem_col:
        sc2 = ax_c.scatter(
            subset_8['coord_x'],
            subset_8['coord_y'],
            c=subset_8[stem_col],
            s=0.3,
            cmap='Blues',
            vmin=0,
            vmax=0.5,
            alpha=0.9,
            rasterized=True
        )
        cbar2 = plt.colorbar(sc2, ax=ax_c, shrink=0.5, aspect=15, pad=0.02)
        cbar2.set_label('Stem cell\nproportion', fontsize=7)
        cbar2.ax.tick_params(labelsize=6)

    ax_c.set_aspect('equal')
    ax_c.axis('off')
    ax_c.text(-0.05, 1.02, 'c', transform=ax_c.transAxes, fontsize=10, fontweight='bold')
    ax_c.text(0.5, -0.05, 'Stem cell distribution (8um)',
              transform=ax_c.transAxes, ha='center', fontsize=8)

    # Panel D: Resolution sensitivity
    ax_d = fig.add_subplot(gs[1, 0])

    if 'brush cell' in props.columns:
        bin_sizes = sorted(props['bin_size'].unique())
        max_prop = []
        for bs in bin_sizes:
            sub = props[props['bin_size'] == bs]
            max_prop.append(sub['brush cell'].max() * 100)

        ax_d.plot(bin_sizes, max_prop, 'o-', color=tuft_color, markersize=5)
        ax_d.set_xscale('log', base=2)
        ax_d.set_xlabel('Resolution (um)')
        ax_d.set_ylabel('Max Tuft Cell Proportion (%)')
        ax_d.set_xticks(bin_sizes)
        ax_d.set_xticklabels([str(b) for b in bin_sizes])

    ax_d.text(-0.2, 1.02, 'd', transform=ax_d.transAxes, fontsize=10, fontweight='bold')

    # Panel E: Co-localization enrichment
    ax_e = fig.add_subplot(gs[1, 1])

    if 'brush cell' in subset_8.columns:
        high_brush = subset_8['brush cell'] > 0.1
        enrichment_data = []

        test_types = ['epithelial fate stem cell', 'enteroendocrine cell',
                      'transit amplifying cell', 'paneth cell', 'enterocyte', 'goblet cell']

        for ct in test_types:
            if ct in subset_8.columns:
                hot_mean = subset_8.loc[high_brush, ct].mean()
                cold_mean = subset_8.loc[~high_brush, ct].mean()
                ratio = hot_mean / cold_mean if cold_mean > 0 else 0
                enrichment_data.append({
                    'cell_type': ct.replace(' cell', '').replace('epithelial fate ', ''),
                    'enrichment': ratio
                })

        if enrichment_data:
            enrich_df = pd.DataFrame(enrichment_data).sort_values('enrichment', ascending=True)

            colors_e = [stem_color if 'stem' in ct.lower() else
                        (tuft_color if 'enteroend' in ct.lower() else other_color)
                        for ct in enrich_df['cell_type']]

            ax_e.barh(range(len(enrich_df)), enrich_df['enrichment'], color=colors_e, alpha=0.8, height=0.7)
            ax_e.set_yticks(range(len(enrich_df)))
            ax_e.set_yticklabels(enrich_df['cell_type'])
            ax_e.set_xlabel('Fold enrichment in Tuft niches')
            ax_e.axvline(x=1, color='gray', linestyle='--', linewidth=0.8)

    ax_e.text(-0.2, 1.02, 'e', transform=ax_e.transAxes, fontsize=10, fontweight='bold')

    # Panel F: Co-localization zoom
    ax_f = fig.add_subplot(gs[1, 2])

    if 'brush cell' in subset_8.columns and stem_col:
        tuft_thresh = 0.05
        stem_thresh = 0.10

        high_tuft_mask = subset_8['brush cell'] > tuft_thresh
        high_stem_mask = subset_8[stem_col] > stem_thresh
        coloc_mask = high_tuft_mask & high_stem_mask

        if coloc_mask.sum() > 0:
            coloc_data = subset_8[coloc_mask]
            center_x = coloc_data['coord_x'].median()
            center_y = coloc_data['coord_y'].median()
        else:
            center_x = subset_8['coord_x'].median()
            center_y = subset_8['coord_y'].median()

        zoom_radius = 400
        zoom_mask = (
            (subset_8['coord_x'] > center_x - zoom_radius) &
            (subset_8['coord_x'] < center_x + zoom_radius) &
            (subset_8['coord_y'] > center_y - zoom_radius) &
            (subset_8['coord_y'] < center_y + zoom_radius)
        )
        zoom_data = subset_8[zoom_mask].copy()

        tuft_vals = zoom_data['brush cell'].values
        stem_vals = zoom_data[stem_col].values

        is_tuft_high = tuft_vals > tuft_thresh
        is_stem_high = stem_vals > stem_thresh

        categories = np.zeros(len(zoom_data), dtype=int)
        categories[is_tuft_high & ~is_stem_high] = 1
        categories[~is_tuft_high & is_stem_high] = 2
        categories[is_tuft_high & is_stem_high] = 3

        # Plot background
        bg_mask = categories == 0
        ax_f.scatter(
            zoom_data.loc[zoom_data.index[bg_mask], 'coord_x'],
            zoom_data.loc[zoom_data.index[bg_mask], 'coord_y'],
            c='#E5E5E5', s=1, alpha=0.3, rasterized=True
        )

        # Plot signal spots
        for cat, color, label, size in [
            (2, '#0072B2', 'Stem-high', 4),
            (1, '#D55E00', 'Tuft-high', 6),
            (3, '#CC79A7', 'Co-localized', 8),
        ]:
            mask = categories == cat
            if mask.sum() > 0:
                ax_f.scatter(
                    zoom_data.loc[zoom_data.index[mask], 'coord_x'],
                    zoom_data.loc[zoom_data.index[mask], 'coord_y'],
                    c=color, s=size, alpha=0.9, label=label, rasterized=True
                )

        leg = ax_f.legend(loc='upper left', frameon=True, fontsize=6, markerscale=1.5,
                          handletextpad=0.3, framealpha=0.9, edgecolor='none')
        leg.get_frame().set_facecolor('white')

    ax_f.set_aspect('equal')
    ax_f.axis('off')
    ax_f.text(-0.05, 1.02, 'f', transform=ax_f.transAxes, fontsize=10, fontweight='bold')
    ax_f.text(0.5, -0.08, 'Tuft-Stem co-localization (zoom)',
              transform=ax_f.transAxes, ha='center', fontsize=8)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path / 'figure7_tuft_discovery.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'figure7_tuft_discovery.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path / 'figure7_tuft_discovery.pdf'}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 7: Tuft-Stem Cell Niche Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing multiscale analysis results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Directory for output figures')
    args = parser.parse_args()

    print("Loading data...")
    props, visibility = load_data(args.results_dir)

    print("Creating figure...")
    create_figure(props, visibility, args.output_dir)

    print("Done!")


if __name__ == '__main__':
    main()
