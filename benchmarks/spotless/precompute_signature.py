#!/usr/bin/env python3
"""
Precompute Signature Matrix from Reference Data.

This script computes the average expression profile per cell type from
the raw scRNA-seq reference data (133K cells) and saves it as a small
signature matrix (K × G). This allows benchmarks to measure FlashDeconv's
true memory footprint without the overhead of loading massive reference data.

Output: liver_signature.npz containing:
  - signature: (K, G) average expression per cell type
  - genes: gene names
  - celltypes: cell type names
"""

import numpy as np
from scipy.io import mmread
from pathlib import Path

DATA_DIR = Path(__file__).parent / "benchmark_data" / "converted"


def main():
    print("=" * 60)
    print("Precomputing Signature Matrix from Reference Data")
    print("=" * 60)

    # Load reference data
    print("\n[1] Loading reference data...")
    ref_name = 'liver_ref_9ct'
    prefix = DATA_DIR / ref_name

    counts = mmread(f"{prefix}_counts.mtx").T.tocsr()
    with open(f"{prefix}_genes.txt") as f:
        genes = [line.strip() for line in f]
    with open(f"{prefix}_celltypes.txt") as f:
        celltypes = [line.strip() for line in f]

    print(f"  Cells: {counts.shape[0]:,}")
    print(f"  Genes: {counts.shape[1]:,}")
    print(f"  Cell types: {len(set(celltypes))}")

    # Get unique cell types
    unique_celltypes = sorted(set(celltypes))
    celltypes_arr = np.array(celltypes)

    print(f"\n[2] Computing average expression per cell type...")

    # Compute signature matrix: average expression per cell type
    # Shape: (K, G) where K = number of cell types, G = number of genes
    K = len(unique_celltypes)
    G = counts.shape[1]
    signature = np.zeros((K, G), dtype=np.float64)

    for i, ct in enumerate(unique_celltypes):
        mask = celltypes_arr == ct
        n_cells = mask.sum()
        # Use sparse mean - efficient!
        ct_counts = counts[mask]
        signature[i] = np.array(ct_counts.mean(axis=0)).flatten()
        print(f"  {ct}: {n_cells:,} cells")

    # Save as compressed NPZ
    output_file = DATA_DIR / "liver_signature.npz"
    print(f"\n[3] Saving to {output_file}...")

    np.savez_compressed(
        output_file,
        signature=signature,
        genes=np.array(genes),
        celltypes=np.array(unique_celltypes)
    )

    # Report size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    memory_size_mb = signature.nbytes / (1024 * 1024)

    print(f"\n[4] Summary:")
    print(f"  Signature shape: {signature.shape}")
    print(f"  Memory size: {memory_size_mb:.2f} MB")
    print(f"  File size: {file_size_mb:.2f} MB (compressed)")
    print(f"\n  Compare to raw reference:")
    print(f"    Raw cells: 133,779")
    print(f"    Raw matrix (dense): ~15.5 GB")
    print(f"    Signature matrix: {memory_size_mb:.2f} MB")
    print(f"    Reduction: {15500 / memory_size_mb:.0f}x smaller!")

    print("\n" + "=" * 60)
    print("Done! Use liver_signature.npz in benchmark scripts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
