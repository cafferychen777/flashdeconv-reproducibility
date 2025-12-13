#!/usr/bin/env python
"""
Figure 2: Leverage scores decouple biological identity from population abundance
================================================================================

This script generates Figure 2, demonstrating the key mechanism behind FlashDeconv:
- Panel A: Abundance invariance - leverage scores remain stable as cell type
           abundances change, while variance-based gene ranking degrades
- Panel B: Gene quadrant analysis - categorizing genes by variance vs leverage
- Panel C: GO enrichment of GOLD genes (low variance, high leverage)
- Panel D: Spatial expression comparison of GOLD vs NOISE genes

Requirements:
    Run leverage_deep_dive.py first to generate intermediate data:
    python analysis/leverage_deep_dive.py --data_dir ./data --output_dir ./results

Usage:
    python figure2_leverage_mechanism.py --results_dir ./results --output_dir ./figures
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Nature-style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
})

# Colors
COLORS = {
    'variance': '#3498db',
    'leverage': '#e74c3c',
    'gold': '#27ae60',
    'quadrant_gold': '#27ae60',
    'quadrant_noise': '#e74c3c',
    'quadrant_high_high': '#3498db',
    'quadrant_low_low': '#ecf0f1',
}


def load_data(results_dir):
    """Load all required data from leverage analysis results."""
    data = {}
    results_path = Path(results_dir) / 'leverage_deep_dive'

    # Panel A: Abundance invariance
    abundance_file = results_path / 'experiment1_abundance_invariance.csv'
    if abundance_file.exists():
        data['abundance'] = pd.read_csv(abundance_file)
    else:
        raise FileNotFoundError(
            f"Abundance invariance data not found: {abundance_file}\n"
            "Run analysis/leverage_deep_dive.py first."
        )

    # Panel B: Gene quadrant
    quadrant_file = results_path / 'experiment2_gene_quadrant_all.csv'
    if quadrant_file.exists():
        data['quadrant'] = pd.read_csv(quadrant_file)

    # Panel C: GO enrichment
    go_file = results_path / 'enrichr_gold/GO_Biological_Process_2021.mouse.enrichr.reports.txt'
    if go_file.exists():
        data['go_gold'] = pd.read_csv(go_file, sep='\t')

    # Panel D: GOLD/NOISE gene lists
    gold_file = results_path / 'experiment2_gold_genes_symbols.csv'
    noise_file = results_path / 'experiment2_noise_genes_symbols.csv'
    if gold_file.exists():
        data['gold_genes'] = pd.read_csv(gold_file)['symbol'].tolist()
    if noise_file.exists():
        data['noise_genes'] = pd.read_csv(noise_file)['symbol'].tolist()

    return data


def plot_panel_a(fig, gs_a, data):
    """Panel A: Abundance invariance test."""
    df = data['abundance']

    ax = fig.add_subplot(gs_a)
    ax.plot(df['abundance_dominant_pct'], df['avg_var_rank_dominant'],
            'o-', color=COLORS['variance'], linewidth=1.5, markersize=5, label='Variance')
    ax.plot(df['abundance_dominant_pct'], df['avg_lev_rank_dominant'],
            's-', color=COLORS['leverage'], linewidth=1.5, markersize=5, label='Leverage')
    ax.set_xlabel('Dominant type abundance (%)')
    ax.set_ylabel('Marker gene rank')
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    ax.set_ylim(90, 260)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_facecolor('white')
    ax.grid(False)
    ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold')


def plot_panel_b(fig, gs_b, data):
    """Panel B: Gene quadrant analysis."""
    ax = fig.add_subplot(gs_b)
    df = data['quadrant']

    quadrant_colors = {
        'Low Var / High Lev (GOLD)': COLORS['quadrant_gold'],
        'High Var / Low Lev (NOISE)': COLORS['quadrant_noise'],
        'High Var / High Lev': COLORS['quadrant_high_high'],
        'Low Var / Low Lev': COLORS['quadrant_low_low'],
    }

    df['quadrant_mapped'] = df['quadrant'].replace({
        'Low Var / High Lev': 'Low Var / High Lev (GOLD)',
        'High Var / Low Lev': 'High Var / Low Lev (NOISE)'
    })

    for quadrant, color in quadrant_colors.items():
        mask = df['quadrant_mapped'] == quadrant
        if mask.sum() > 0:
            subset = df[mask]
            ax.scatter(subset['log_variance'], subset['log_leverage'],
                      c=color, s=3, alpha=0.5, label=f'{quadrant} ({mask.sum()})',
                      rasterized=True)

    var_thresh = df['log_variance'].median()
    lev_thresh = df['log_leverage'].median()
    ax.axvline(var_thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(lev_thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r'$\log_{10}$(Variance)')
    ax.set_ylabel(r'$\log_{10}$(Leverage score)')
    ax.legend(loc='upper left', frameon=False, fontsize=6, markerscale=2)
    ax.set_facecolor('white')
    ax.grid(False)
    ax.text(-0.08, 1.05, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold')


def plot_panel_c(fig, gs_c, data):
    """Panel C: GO enrichment barplot."""
    ax = fig.add_subplot(gs_c)

    if 'go_gold' not in data:
        ax.text(0.5, 0.5, 'GO data not available', ha='center', va='center',
                transform=ax.transAxes)
        ax.text(-0.02, 1.05, 'c', transform=ax.transAxes, fontsize=12, fontweight='bold')
        return

    go_df = data['go_gold']
    top_terms = go_df.nsmallest(8, 'Adjusted P-value').copy()

    def clean_term(x):
        term = x.split('(GO:')[0].strip()
        term = term.replace('response to transforming growth factor beta stimulus',
                           'response to TGF-beta stimulus')
        term = term.replace('lymphatic endothelial cell differentiation',
                           'lymphatic endothelial diff.')
        return term
    top_terms['Term_clean'] = top_terms['Term'].apply(clean_term)

    y_pos = np.arange(len(top_terms))
    ax.barh(y_pos, -np.log10(top_terms['Adjusted P-value']),
            color=COLORS['gold'], alpha=0.8, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_terms['Term_clean'], fontsize=8)
    ax.set_xlabel(r'$-\log_{10}$(Adjusted P-value)', fontsize=9)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(-np.log10(0.01), color='darkred', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.invert_yaxis()
    ax.set_facecolor('white')
    ax.grid(False)
    ax.text(-0.02, 1.05, 'c', transform=ax.transAxes, fontsize=12, fontweight='bold')


def plot_panel_d(fig, gs_d, data, visium_path):
    """Panel D: Spatial expression comparison (placeholder if no spatial data)."""
    ax = fig.add_subplot(gs_d)
    ax.text(0.5, 0.5, 'Spatial visualization\n(requires Visium data)',
            ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.axis('off')
    ax.text(-0.02, 1.02, 'd', transform=ax.transAxes, fontsize=12, fontweight='bold')


def create_figure(data, output_dir, visium_path=None):
    """Create the complete Figure 2."""
    fig = plt.figure(figsize=(14, 10))

    gs_main = GridSpec(2, 2, figure=fig,
                       height_ratios=[1, 1.2],
                       width_ratios=[1, 1],
                       hspace=0.25, wspace=0.25,
                       left=0.10, right=0.97, top=0.96, bottom=0.05)

    plot_panel_a(fig, gs_main[0, 0], data)
    plot_panel_b(fig, gs_main[0, 1], data)
    plot_panel_c(fig, gs_main[1, 0], data)
    plot_panel_d(fig, gs_main[1, 1], data, visium_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path / 'figure2_leverage_mechanism.png', dpi=300, facecolor='white')
    fig.savefig(output_path / 'figure2_leverage_mechanism.pdf', facecolor='white')
    plt.close()

    print(f"Saved: {output_path / 'figure2_leverage_mechanism.pdf'}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 2: Leverage Mechanism',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python figure2_leverage_mechanism.py --results_dir ./results --output_dir ./figures
        """
    )
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing leverage analysis results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Directory for output figures')
    parser.add_argument('--visium_path', type=str, default=None,
                        help='Path to Visium h5ad file (optional, for Panel D)')
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.results_dir)

    print("Creating figure...")
    create_figure(data, args.output_dir, args.visium_path)

    print("Done!")


if __name__ == '__main__':
    main()
