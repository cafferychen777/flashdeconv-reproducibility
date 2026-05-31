#!/usr/bin/env python
"""Leverage Deep Dive Analysis."""

import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro.signatures import (
    build_signature_matrix,
    experiment1_abundance_invariance,
    experiment2_gene_quadrant,
    load_mouse_brain_reference,
)


def main():
    parser = argparse.ArgumentParser(
        description="Leverage Deep Dive Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing mouse brain data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "leverage_deep_dive"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "enrichr_gold").mkdir(exist_ok=True)

    print("=" * 60)
    print("LEVERAGE DEEP DIVE ANALYSIS")
    print("=" * 60)

    sc_adata = load_mouse_brain_reference(args.data_dir)
    X, cell_types = build_signature_matrix(sc_adata, "cell_type", normalize="log_cpm")
    gene_names = list(sc_adata.var_names)

    abundance_df = experiment1_abundance_invariance(X, cell_types, gene_names, output_dir)
    quadrant_df = experiment2_gene_quadrant(X, gene_names, output_dir)

    print("\nSUMMARY")
    print(f"Abundance experiment rows: {len(abundance_df)}")
    print(f"Gene quadrant rows: {len(quadrant_df):,}")
    print("GO enrichment input files were written next to the quadrant table.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
