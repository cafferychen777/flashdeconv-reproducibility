"""
Parameter tuning for Melanoma case study.

Goal: Find best parameters with Full reference and sketch_dim=512.

Parameters to tune:
- n_hvg: 2000, 3000, 5000, 10000
- n_markers_per_type: 50, 100, 200
- lambda_spatial: 0, "auto", 5000, 10000
- rho_sparsity: 0, 0.01, 0.05
- preprocess: "log_cpm", "pearson"
"""

import sys
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.spatial.distance import jensenshannon
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/apple/Research/FlashDeconv')
from flashdeconv import FlashDeconv

# Paths
DATA_DIR = Path("/Users/apple/Research/FlashDeconv/validation/benchmark_data/converted")

# Malignant states
MALIGNANT_STATES = [
    'melanocytic/oxphos', 'neural-like', 'immune-like', 'stem-like',
    'stress-like (hypoxia/UPR)', 'RNA-processing', 'mesenchymal'
]

NON_MALIGNANT = ['B cell', 'CAF', 'DC', 'EC', 'Monocyte/macrophage', 'pDC', 'Pericyte', 'T/NK cell']

# Ground truth
MC_GROUND_TRUTH = {
    'Bcell': 0.005, 'CAF': 0.012, 'EC': 0.032, 'Melanocytic': 0.848,
    'Mono/Mac': 0.039, 'Pericyte': 0.017, 'Tcell': 0.047,
}
EVAL_MAP = {
    'B cell': 'Bcell', 'CAF': 'CAF', 'EC': 'EC',
    'Monocyte/macrophage': 'Mono/Mac', 'Pericyte': 'Pericyte', 'T/NK cell': 'Tcell',
}
EVAL_CELLTYPES = list(MC_GROUND_TRUTH.keys())


def load_reference():
    prefix = DATA_DIR / "melanoma_ref"
    counts = mmread(f"{prefix}_counts.mtx")
    with open(f"{prefix}_genes.txt") as f:
        genes = [line.strip() for line in f]
    with open(f"{prefix}_celltypes.txt") as f:
        celltypes = [line.strip() for line in f]
    return counts, genes, celltypes


def load_spatial(sample_id):
    prefix = DATA_DIR / f"melanoma_visium_sample{sample_id:02d}"
    counts = mmread(f"{prefix}_counts.mtx").toarray()
    with open(f"{prefix}_genes.txt") as f:
        genes = [line.strip() for line in f]
    coords = pd.read_csv(f"{prefix}_coords.csv", index_col=0)
    coords_arr = coords[['x', 'y']].values if 'x' in coords.columns else coords.iloc[:, :2].values
    return counts, genes, coords_arr


def build_full_signature(counts_sparse, celltypes, genes):
    """Build signature with all 15 cell types."""
    X_csc = counts_sparse.tocsc()
    celltypes_arr = np.array(celltypes)
    full_cts = sorted(set(celltypes))
    signature = np.zeros((len(full_cts), X_csc.shape[0]), dtype=np.float64)
    for i, ct in enumerate(full_cts):
        mask = (celltypes_arr == ct)
        if mask.sum() > 0:
            signature[i] = X_csc[:, mask].mean(axis=1).A1
    return signature, genes, full_cts


def align_genes(Y, sp_genes, X, ref_genes):
    common = sorted(set(sp_genes) & set(ref_genes))
    sp_idx = {g: i for i, g in enumerate(sp_genes)}
    ref_idx = {g: i for i, g in enumerate(ref_genes)}
    return Y[:, [sp_idx[g] for g in common]], X[:, [ref_idx[g] for g in common]], common


def aggregate_to_eval(props, cell_types):
    result = {}
    mal_sum = 0
    for i, ct in enumerate(cell_types):
        if ct in MALIGNANT_STATES:
            mal_sum += props[i]
        elif ct in EVAL_MAP:
            result[EVAL_MAP[ct]] = props[i]
    result['Melanocytic'] = mal_sum
    total = sum(result.get(ct, 0) for ct in EVAL_CELLTYPES)
    if total > 0:
        for ct in EVAL_CELLTYPES:
            result[ct] = result.get(ct, 0) / total
    return np.array([result.get(ct, 0) for ct in EVAL_CELLTYPES])


def calculate_jsd(p_true, p_pred):
    p = np.array(p_true) + 1e-10
    q = np.array(p_pred) + 1e-10
    return jensenshannon(p / p.sum(), q / q.sum()) ** 2


def run_single_config(Y, X, coords, full_cts, gt_vec, config):
    """Run FlashDeconv with a single config."""
    try:
        model = FlashDeconv(
            sketch_dim=512,  # Fixed
            lambda_spatial=config['lambda_spatial'],
            preprocess=config['preprocess'],
            n_hvg=config['n_hvg'],
            n_markers_per_type=config['n_markers_per_type'],
            rho_sparsity=config['rho_sparsity'],
            max_iter=200,
            tol=1e-4,
            random_state=42,
            verbose=False,
        )
        props = model.fit_transform(Y, X, coords)
        mean_props = props.mean(axis=0)
        eval_props = aggregate_to_eval(mean_props, full_cts)
        jsd = calculate_jsd(gt_vec, eval_props)
        melanocytic = eval_props[EVAL_CELLTYPES.index('Melanocytic')]
        return jsd, melanocytic
    except Exception as e:
        print(f"    Error: {e}")
        return np.nan, np.nan


def main():
    print("=" * 70)
    print("MELANOMA PARAMETER TUNING")
    print("Fixed: sketch_dim=512, Full reference (15 types)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    ref_counts, ref_genes, ref_celltypes = load_reference()
    X_full, _, full_cts = build_full_signature(ref_counts, ref_celltypes, ref_genes)
    gt_vec = np.array([MC_GROUND_TRUTH[ct] for ct in EVAL_CELLTYPES])

    # Parameter grid
    param_grid = {
        'n_hvg': [2000, 5000, 10000],
        'n_markers_per_type': [50, 100],
        'lambda_spatial': [0, "auto", 5000],
        'rho_sparsity': [0, 0.01, 0.05],
        'preprocess': ["log_cpm", "pearson"],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    print(f"\nTotal configs to test: {len(combinations)}")

    # Test on all samples
    samples = [2, 3, 4]
    results = []

    for idx, values in enumerate(combinations):
        config = dict(zip(keys, values))
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        print(f"\n[{idx+1}/{len(combinations)}] {config_str}")

        sample_jsds = []
        sample_mels = []

        for sample_id in samples:
            Y_raw, sp_genes, coords = load_spatial(sample_id)
            Y, X, _ = align_genes(Y_raw, sp_genes, X_full, ref_genes)

            jsd, mel = run_single_config(Y, X, coords, full_cts, gt_vec, config)
            sample_jsds.append(jsd)
            sample_mels.append(mel)

        avg_jsd = np.nanmean(sample_jsds)
        avg_mel = np.nanmean(sample_mels)
        print(f"  Avg JSD: {avg_jsd:.4f}, Avg Melanocytic: {avg_mel*100:.1f}%")

        results.append({
            **config,
            'jsd_s2': sample_jsds[0],
            'jsd_s3': sample_jsds[1],
            'jsd_s4': sample_jsds[2],
            'avg_jsd': avg_jsd,
            'avg_melanocytic': avg_mel,
        })

    # Summary
    df = pd.DataFrame(results)
    df = df.sort_values('avg_jsd')

    print("\n" + "=" * 70)
    print("TOP 10 CONFIGS:")
    print("=" * 70)
    print(df.head(10).to_string(index=False))

    # Save results
    output_path = Path(__file__).parent / "tuning_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Best config
    best = df.iloc[0]
    print("\n" + "=" * 70)
    print("BEST CONFIG:")
    print("=" * 70)
    print(f"  n_hvg: {best['n_hvg']}")
    print(f"  n_markers_per_type: {best['n_markers_per_type']}")
    print(f"  lambda_spatial: {best['lambda_spatial']}")
    print(f"  rho_sparsity: {best['rho_sparsity']}")
    print(f"  preprocess: {best['preprocess']}")
    print(f"  Avg JSD: {best['avg_jsd']:.4f}")
    print(f"  Avg Melanocytic: {best['avg_melanocytic']*100:.1f}%")

    # Compare with baseline
    baseline_jsd = 0.0331  # Current Full reference result
    improvement = (baseline_jsd - best['avg_jsd']) / baseline_jsd * 100
    print(f"\n  Improvement over baseline (0.0331): {improvement:.1f}%")

    return df


if __name__ == "__main__":
    df = main()
