#!/usr/bin/env python
"""
Benchmark FlashDeconv on Liver case study.

This script reproduces the Liver case study results from the FlashDeconv paper.

Evaluation metrics (following Spotless paper):
1. AUPR: Area under precision-recall curve for portal/central vein EC detection
2. JSD: Jensen-Shannon Divergence vs snRNA-seq cell type proportions

Cell types (9 common types):
- Hepatocytes (dominant ~60%)
- Kupffer cells
- LSECs (liver sinusoidal endothelial cells)
- T cells, B cells
- Cholangiocytes
- Portal Vein Endothelial cells, Central Vein Endothelial cells
- Mesothelial cells

Usage:
    python benchmark_liver.py --data_dir ./data/spotless/converted

Requirements:
    - FlashDeconv (pip install flashdeconv)
    - Converted Spotless liver data (see scripts/convert_spotless_data.R)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_recall_curve, auc
import time
import warnings
warnings.filterwarnings('ignore')

from flashdeconv import FlashDeconv


# 9 cell types used in benchmark
CELLTYPES = [
    'B cells',
    'Central Vein Endothelial cells',
    'Cholangiocytes',
    'Hepatocytes',
    'Kupffer cells',
    'LSECs',
    'Mesothelial cells',
    'Portal Vein Endothelial cells',
    'T cells',
]

# snRNA-seq ground truth proportions (from Spotless paper)
SNRNASEQ_PROPORTIONS = {
    'Hepatocytes': 0.65,
    'Kupffer cells': 0.10,
    'LSECs': 0.08,
    'T cells': 0.04,
    'B cells': 0.03,
    'Cholangiocytes': 0.04,
    'Portal Vein Endothelial cells': 0.02,
    'Central Vein Endothelial cells': 0.02,
    'Mesothelial cells': 0.02,
}


def load_reference(data_dir, ref_name='liver_ref_9ct'):
    """Load reference scRNA-seq data."""
    prefix = os.path.join(data_dir, ref_name)

    counts = mmread(f"{prefix}_counts.mtx").T.tocsr()
    with open(f"{prefix}_genes.txt") as f:
        genes = [line.strip() for line in f]
    with open(f"{prefix}_celltypes.txt") as f:
        celltypes = [line.strip() for line in f]

    return {'counts': counts, 'genes': genes, 'celltypes': celltypes}


def load_visium_sample(data_dir, sample_name):
    """Load Visium spatial data with zonation annotations."""
    prefix = os.path.join(data_dir, sample_name)

    counts = mmread(f"{prefix}_counts.mtx").toarray()
    with open(f"{prefix}_genes.txt") as f:
        genes = [line.strip() for line in f]
    coords = pd.read_csv(f"{prefix}_coords.csv", index_col=0)
    coords_arr = coords[['x', 'y']].values if 'x' in coords.columns else coords.iloc[:, :2].values

    # Load metadata with zonation
    metadata = pd.read_csv(f"{prefix}_metadata.csv", index_col=0)

    return {
        'counts': counts,
        'genes': genes,
        'coords': coords_arr,
        'metadata': metadata,
    }


def build_signature_matrix(ref, target_genes, target_celltypes):
    """Build signature matrix from raw counts (average per cell type)."""
    ref_genes = list(ref['genes'])
    common_genes = sorted(set(ref_genes) & set(target_genes))

    print(f"  Common genes: {len(common_genes)}")

    ref_gene_idx = {g: i for i, g in enumerate(ref_genes)}
    target_gene_idx = {g: i for i, g in enumerate(target_genes)}

    common_ref_idx = [ref_gene_idx[g] for g in common_genes]
    common_target_idx = [target_gene_idx[g] for g in common_genes]

    ref_counts = ref['counts'][:, common_ref_idx]
    if hasattr(ref_counts, 'toarray'):
        ref_counts = ref_counts.toarray()

    ref_celltypes = np.array(ref['celltypes'])
    X_ref = np.zeros((len(target_celltypes), len(common_genes)))

    for i, ct in enumerate(target_celltypes):
        mask = ref_celltypes == ct
        if mask.sum() > 0:
            X_ref[i] = ref_counts[mask].mean(axis=0)
        else:
            print(f"  Warning: No cells found for {ct}")

    return X_ref, common_target_idx


def calculate_jsd(props_true, props_pred):
    """Calculate Jensen-Shannon Divergence."""
    props_true = np.array(props_true) + 1e-10
    props_pred = np.array(props_pred) + 1e-10
    props_true = props_true / props_true.sum()
    props_pred = props_pred / props_pred.sum()
    return jensenshannon(props_true, props_pred) ** 2


def calculate_aupr(y_true, y_score):
    """Calculate Area Under Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def run_benchmark(data_dir, output_dir):
    """Run FlashDeconv benchmark on liver data."""
    print("=" * 60)
    print("Liver Case Study Benchmark")
    print("=" * 60)

    # Load reference (combined 3 protocols)
    print("\nLoading reference data (9 cell types, all protocols)...")
    ref = load_reference(data_dir, 'liver_ref_9ct')
    print(f"  Cells: {ref['counts'].shape[0]}")
    print(f"  Genes: {len(ref['genes'])}")
    print(f"  Cell types: {len(set(ref['celltypes']))}")

    # Sort celltypes to match expected order
    unique_cts = sorted(set(ref['celltypes']))
    print(f"  Types: {unique_cts}")

    # Visium samples
    samples = [
        'liver_mouseVisium_JB01',
        'liver_mouseVisium_JB02',
        'liver_mouseVisium_JB03',
        'liver_mouseVisium_JB04',
    ]

    all_results = []

    for sample_name in samples:
        print(f"\n{'=' * 40}")
        print(f"Processing: {sample_name}")
        print('=' * 40)

        # Load spatial data
        try:
            sp = load_visium_sample(data_dir, sample_name)
        except FileNotFoundError:
            print(f"  Sample not found, skipping")
            continue

        print(f"  Spots: {sp['counts'].shape[0]}")
        print(f"  Genes: {len(sp['genes'])}")

        # Check zonation annotations
        zonation = sp['metadata']['zonationGroup'].values
        print(f"  Zonation groups: {np.unique(zonation)}")

        # Build signature matrix
        X_ref, gene_idx = build_signature_matrix(ref, sp['genes'], unique_cts)

        # Prepare spatial data
        Y = sp['counts'][:, gene_idx]
        coords = sp['coords']

        print(f"  Data shape: Y={Y.shape}, X_ref={X_ref.shape}")

        # FlashDeconv configurations
        configs = [
            {"name": "default", "lambda_spatial": "auto", "sketch_dim": 512, "n_hvg": 2000},
            {"name": "large", "lambda_spatial": 0, "sketch_dim": 2048, "n_hvg": 10000},
        ]

        for config in configs:
            print(f"\n  Config: {config['name']}")

            model = FlashDeconv(
                sketch_dim=min(config['sketch_dim'], Y.shape[1]),
                lambda_spatial=config['lambda_spatial'],
                preprocess="log_cpm",
                n_hvg=config.get('n_hvg', 2000),
                rho_sparsity=0,
                max_iter=200,
                verbose=False,
                random_state=42,
            )

            try:
                t0 = time.time()
                pred = model.fit_transform(Y, X_ref, coords)
                elapsed_time = time.time() - t0

                # Mean proportions
                mean_props = pred.mean(axis=0)

                # Calculate JSD vs snRNA-seq ground truth
                gt_vec = np.array([SNRNASEQ_PROPORTIONS.get(ct, 0) for ct in unique_cts])
                jsd = calculate_jsd(gt_vec, mean_props)

                # Calculate AUPR for Portal/Central vein EC zonation
                portal_idx = unique_cts.index('Portal Vein Endothelial cells')
                central_idx = unique_cts.index('Central Vein Endothelial cells')

                portal_pred = pred[:, portal_idx]
                central_pred = pred[:, central_idx]

                # AUPR calculation: use spots in Portal or Central zones
                aupr_mask = (zonation == 'Portal') | (zonation == 'Central')

                portal_label = (zonation[aupr_mask] == 'Portal').astype(int)
                central_label = (zonation[aupr_mask] == 'Central').astype(int)

                portal_pred_filtered = portal_pred[aupr_mask]
                central_pred_filtered = central_pred[aupr_mask]

                aupr_portal = calculate_aupr(portal_label, portal_pred_filtered)
                aupr_central = calculate_aupr(central_label, central_pred_filtered)
                aupr_mean = (aupr_portal + aupr_central) / 2

                all_results.append({
                    'sample': sample_name,
                    'config': config['name'],
                    'jsd': jsd,
                    'aupr_portal': aupr_portal,
                    'aupr_central': aupr_central,
                    'aupr_mean': aupr_mean,
                    'Hepatocytes_prop': mean_props[unique_cts.index('Hepatocytes')],
                    'time_sec': elapsed_time,
                })

                print(f"    JSD: {jsd:.4f}")
                print(f"    AUPR (mean): {aupr_mean:.4f}")
                print(f"    Time: {elapsed_time:.2f}s")

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark FlashDeconv on Liver case study')
    parser.add_argument('--data_dir', type=str, default='./data/spotless/converted',
                        help='Directory containing converted Spotless data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output results')
    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    results = run_benchmark(args.data_dir, args.output_dir)

    if results:
        df = pd.DataFrame(results)

        # Save results
        output_file = os.path.join(args.output_dir, 'liver_benchmark_results.csv')
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        summary = df.groupby('config').agg({
            'jsd': 'mean',
            'aupr_mean': 'mean',
        }).round(4)

        print("\nFlashDeconv Results (avg across 4 Visium slides):")
        print(summary)

        # Compare with other methods
        print("\n" + "-" * 40)
        print("Comparison with other methods (JSD, lower is better):")
        print("-" * 40)

        other_methods_jsd = {
            'rctd': 0.0334,
            'cell2location': 0.0352,
            'spatialdwls': 0.0646,
            'nnls': 0.1056,
        }

        flash_jsd = df[df['config'] == 'large']['jsd'].mean()
        all_jsd = {**other_methods_jsd, 'FlashDeconv': flash_jsd}

        for i, (method, val) in enumerate(sorted(all_jsd.items(), key=lambda x: x[1]), 1):
            marker = " <- FlashDeconv" if method == 'FlashDeconv' else ""
            print(f"  {i}. {method:15s}: {val:.4f}{marker}")


if __name__ == '__main__':
    main()
