#!/usr/bin/env python
"""Cortex Lamination Deconvolution."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro.signatures import (
    create_reference_signatures,
    evaluate_with_markers,
    load_cortex_paired_data,
    run_cortex_flashdeconv,
    save_cortex_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run FlashDeconv on Cell2location paired mouse brain data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing mouse brain data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CORTEX LAMINATION DECONVOLUTION")
    print("=" * 60)

    sc_adata, sp_adata, common_genes = load_cortex_paired_data(args.data_dir)
    print(f"Common genes: {len(common_genes):,}")

    X_ref, cell_types = create_reference_signatures(sc_adata)
    prop_df, elapsed = run_cortex_flashdeconv(sp_adata, X_ref, cell_types)
    results_df = evaluate_with_markers(sp_adata, prop_df, sc_adata)
    save_cortex_outputs(output_dir, prop_df, sp_adata, cell_types, results_df)

    print("\nRESULT SUMMARY")
    if results_df is not None:
        print(f"Mean marker correlation: {results_df['correlation'].mean():.3f}")
        print(f"Top-10 marker correlation: {results_df.head(10)['correlation'].mean():.3f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
