#!/usr/bin/env python
"""Hidden Cell Type Analysis."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro.signatures import (
    build_signature_matrix,
    compute_marker_visibility,
    find_hidden_genes,
    load_intestine_reference,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hidden cell types in intestinal reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing reference data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HIDDEN CELL TYPE ANALYSIS")
    print("=" * 60)

    ref, ct_col = load_intestine_reference(args.data_dir)
    X, cell_types = build_signature_matrix(ref, ct_col, normalize="cpm")
    gene_names = ref.var_names.tolist()

    visibility_df = compute_marker_visibility(X, cell_types, gene_names)
    visibility_df.to_csv(output_dir / "cell_type_visibility.csv", index=False)

    hidden_genes_df = find_hidden_genes(X, gene_names)
    hidden_genes_df.to_csv(output_dir / "hidden_genes.csv", index=False)

    hidden_types = visibility_df[visibility_df["hvg_blindness"] > 5]
    if not hidden_types.empty:
        print("\nCell types with markers that HVG would likely miss:")
        for _, row in hidden_types.iterrows():
            print(f"  - {row['cell_type']}: {row['hvg_blindness']:.1f}%")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
