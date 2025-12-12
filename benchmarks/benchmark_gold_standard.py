#!/usr/bin/env python
"""
Benchmark FlashDeconv on Gold Standard datasets.

This script reproduces the Gold Standard benchmark results from the FlashDeconv paper.

Gold Standard datasets:
1. STARMap (Wang2018): 108 spots, mouse visual cortex - subcellular resolution
2. seqFISH+ (Eng2019): 14 FOVs (<10 spots each), mouse cortex & olfactory bulb

Reference mapping:
- gold_ref_1: seqFISH+ cortex reference (906 cells) -> Eng2019_cortex_svz
- gold_ref_2: seqFISH+ olfactory bulb reference (2050 cells) -> Eng2019_ob
- gold_ref_3: STARMap reference (12824 cells) -> Wang2018

Usage:
    python benchmark_gold_standard.py --data_dir ./data/spotless/converted

Requirements:
    - FlashDeconv (pip install flashdeconv)
    - Converted Spotless Gold Standard data
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

from flashdeconv import FlashDeconv


def load_reference(prefix):
    """Load converted reference data."""
    counts = mmread(f"{prefix}_counts.mtx").T.tocsr()
    with open(f"{prefix}_genes.txt") as f:
        genes = [line.strip() for line in f]
    with open(f"{prefix}_cells.txt") as f:
        cells = [line.strip() for line in f]
    with open(f"{prefix}_celltypes.txt") as f:
        celltypes = [line.strip() for line in f]

    return {
        'counts': counts,
        'genes': genes,
        'cells': cells,
        'celltypes': celltypes
    }


def load_test_data(prefix):
    """Load Gold Standard test data."""
    counts = mmread(f"{prefix}_counts.mtx").toarray().T  # spots x genes
    with open(f"{prefix}_genes.txt") as f:
        genes = [line.strip() for line in f]

    coords_df = pd.read_csv(f"{prefix}_coords.csv", index_col=0)
    coords = coords_df[['x', 'y']].values

    props_df = pd.read_csv(f"{prefix}_proportions.csv", index_col=0)
    if 'spot_no' in props_df.columns:
        props_df = props_df.drop(columns=['spot_no'])

    return {
        'counts': counts,
        'genes': genes,
        'coords': coords,
        'proportions': props_df.values,
        'cell_types': list(props_df.columns)
    }


def normalize_celltype(ct):
    """Normalize cell type name for matching."""
    ct = ct.replace('.', ' ').replace('_', ' ').replace('/', ' ')
    ct = ct.lower().strip()
    ct = ' '.join(ct.split())
    return ct


def build_signature_matrix(ref, target_genes, target_celltypes):
    """Build cell type signature matrix from reference data."""
    ref_genes = list(ref['genes'])
    common_genes = sorted(set(ref_genes) & set(target_genes))

    if len(common_genes) < 50:
        return None, None, []

    ref_gene_idx = {g: i for i, g in enumerate(ref_genes)}
    target_gene_idx = {g: i for i, g in enumerate(target_genes)}

    common_ref_idx = [ref_gene_idx[g] for g in common_genes]
    common_target_idx = [target_gene_idx[g] for g in common_genes]

    ref_counts = ref['counts'][:, common_ref_idx]
    if hasattr(ref_counts, 'toarray'):
        ref_counts = ref_counts.toarray()

    ref_celltypes = np.array(ref['celltypes'])
    unique_ref_cts = list(set(ref_celltypes))
    ref_ct_norm = {normalize_celltype(rct): rct for rct in unique_ref_cts}

    X_ref = np.zeros((len(target_celltypes), len(common_genes)))

    for i, tct in enumerate(target_celltypes):
        if tct in unique_ref_cts:
            mask = ref_celltypes == tct
            X_ref[i] = ref_counts[mask].mean(axis=0)
        else:
            tct_norm = normalize_celltype(tct)
            if tct_norm in ref_ct_norm:
                rct = ref_ct_norm[tct_norm]
                mask = ref_celltypes == rct
                X_ref[i] = ref_counts[mask].mean(axis=0)
            else:
                matched = False
                for rct_norm, rct in ref_ct_norm.items():
                    if tct_norm in rct_norm or rct_norm in tct_norm:
                        mask = ref_celltypes == rct
                        X_ref[i] = ref_counts[mask].mean(axis=0)
                        matched = True
                        break
                if not matched:
                    X_ref[i] = ref_counts.mean(axis=0)

    return X_ref, common_target_idx, common_genes


def calculate_metrics(pred, true):
    """Calculate RMSE, JSD, AUPR, Pearson."""
    rmse = np.sqrt(np.mean((pred - true) ** 2))

    jsd_values = []
    for i in range(pred.shape[0]):
        p = pred[i] + 1e-10
        q = true[i] + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        jsd_values.append(jensenshannon(p, q) ** 2)
    jsd = np.mean(jsd_values)

    true_binary = (true.flatten() > 0.01).astype(int)
    pred_flat = pred.flatten()
    if true_binary.sum() > 0 and true_binary.sum() < len(true_binary):
        precision, recall, _ = precision_recall_curve(true_binary, pred_flat)
        aupr = auc(recall, precision)
    else:
        aupr = np.nan

    pearson_r = pearsonr(pred.flatten(), true.flatten())[0]

    return rmse, jsd, aupr, pearson_r


def run_benchmark(data_dir, output_dir):
    """Run benchmark on Gold Standard datasets."""

    ref_mapping = {
        'cortex_svz': 'gold_ref_1',  # seqFISH+ cortex
        'ob': 'gold_ref_2',          # seqFISH+ olfactory bulb
        'visp': 'gold_ref_3',        # STARMap
    }

    test_datasets = [
        # Eng2019 cortex_svz (7 FOVs)
        *[f"Eng2019_cortex_svz_fov{i}" for i in range(7)],
        # Eng2019 olfactory bulb (7 FOVs)
        *[f"Eng2019_ob_fov{i}" for i in range(7)],
        # Wang2018 STARMap
        "Wang2018_visp_rep0410",
    ]

    all_results = []

    # Load references
    refs = {}
    for key, ref_name in ref_mapping.items():
        ref_path = os.path.join(data_dir, ref_name)
        if os.path.exists(f"{ref_path}_counts.mtx"):
            refs[key] = load_reference(ref_path)
            print(f"Loaded {ref_name}: {refs[key]['counts'].shape[0]} cells, "
                  f"{len(set(refs[key]['celltypes']))} cell types")

    print("\n" + "=" * 70)
    print("Running Gold Standard Benchmark")
    print("=" * 70)

    for ds_name in test_datasets:
        test_path = os.path.join(data_dir, ds_name)
        if not os.path.exists(f"{test_path}_counts.mtx"):
            continue

        # Determine which reference to use
        if 'cortex_svz' in ds_name:
            ref_key = 'cortex_svz'
        elif '_ob_' in ds_name:
            ref_key = 'ob'
        else:
            ref_key = 'visp'

        ref = refs.get(ref_key)
        if ref is None:
            print(f"  Reference {ref_key} not found, skipping {ds_name}")
            continue

        try:
            test = load_test_data(test_path)
            print(f"\n{ds_name}:")
            print(f"  Test: {test['counts'].shape[0]} spots, {len(test['cell_types'])} cell types")

            X_ref, gene_idx, common_genes = build_signature_matrix(
                ref, test['genes'], test['cell_types']
            )

            if X_ref is None:
                print(f"  Insufficient common genes, skipping")
                continue

            Y = test['counts'][:, gene_idx]
            ground_truth = test['proportions']
            coords = test['coords']

            print(f"  Common genes: {len(common_genes)}")

            # Run FlashDeconv
            t0 = time.time()
            model = FlashDeconv(
                sketch_dim=min(256, Y.shape[1]),
                lambda_spatial=0.1,
                rho_sparsity=0.01,
                preprocess="log_cpm",
                n_hvg=min(500, Y.shape[1]),
                max_iter=100,
                verbose=False,
                random_state=42,
            )
            pred = model.fit_transform(Y, X_ref, coords)
            elapsed = time.time() - t0

            rmse, jsd, aupr, pearson_r = calculate_metrics(pred, ground_truth)

            all_results.append({
                'dataset': ds_name,
                'reference': ref_key,
                'n_spots': Y.shape[0],
                'n_celltypes': len(test['cell_types']),
                'n_common_genes': len(common_genes),
                'pearson': pearson_r,
                'rmse': rmse,
                'jsd': jsd,
                'aupr': aupr,
                'time': elapsed,
            })

            print(f"  Pearson={pearson_r:.4f}, RMSE={rmse:.4f}, Time={elapsed:.2f}s")

        except Exception as e:
            print(f"  Error: {e}")

    return pd.DataFrame(all_results)


def summarize_results(df):
    """Summarize results following Spotless methodology."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df['tissue'] = df['dataset'].apply(
        lambda x: 'cortex_svz' if 'cortex_svz' in x
        else ('ob' if '_ob_' in x else 'visp')
    )
    df['source'] = df['dataset'].apply(
        lambda x: 'Eng2019' if 'Eng2019' in x else 'Wang2018'
    )

    # seqFISH+ (Eng2019) results
    eng = df[df['source'] == 'Eng2019']
    if len(eng) > 0:
        print(f"\nseqFISH+ (Eng2019, n={len(eng)} FOVs):")
        print(f"  Pearson: {eng['pearson'].mean():.4f} ± {eng['pearson'].std():.4f}")
        print(f"  RMSE: {eng['rmse'].mean():.4f}")

    # STARMap (Wang2018) results
    wang = df[df['source'] == 'Wang2018']
    if len(wang) > 0:
        print(f"\nSTARMap (Wang2018, n={len(wang)}):")
        print(f"  Pearson: {wang['pearson'].mean():.4f}")
        print(f"  RMSE: {wang['rmse'].mean():.4f}")

    # Compare with published results
    print("\n" + "-" * 40)
    print("Comparison with Spotless paper (Pearson):")
    print("-" * 40)
    print("Wang2018: spatialdwls=0.712, music=0.663, rctd=0.568")
    print("Eng2019:  rctd=0.720, music=0.719, spatialdwls=0.636")

    if len(wang) > 0:
        print(f"\nFlashDeconv Wang2018: {wang['pearson'].mean():.4f}")
    if len(eng) > 0:
        print(f"FlashDeconv Eng2019:  {eng['pearson'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark FlashDeconv on Gold Standard')
    parser.add_argument('--data_dir', type=str, default='./data/spotless/converted',
                        help='Directory containing converted data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory for output results')
    args = parser.parse_args()

    print("=" * 70)
    print("Gold Standard Benchmark: STARMap + seqFISH+")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = run_benchmark(args.data_dir, args.output_dir)

    if len(df) == 0:
        print("No results!")
        return

    # Save results
    output_file = os.path.join(args.output_dir, 'gold_standard_results.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    summarize_results(df)


if __name__ == '__main__':
    main()
