#!/usr/bin/env python
"""Tuft-Stem Cell Niche Discovery Analysis."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro.plotting import create_tuft_discovery_figure
from repro.signatures import compute_hvg_blindness, load_intestine_reference
from repro.visium_hd import analyze_colocalization, run_tuft_validation


def main():
    parser = argparse.ArgumentParser(
        description="Tuft-Stem Cell Niche Discovery Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output files")
    parser.add_argument("--props_file", type=str, default=None, help="Pre-computed proportions CSV (optional)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TUFT-STEM CELL NICHE DISCOVERY")
    print("=" * 60)

    props_path = Path(args.props_file) if args.props_file else output_dir / "multiscale_proportions.csv"
    if not props_path.exists():
        print(f"\nProportions file not found: {props_path}")
        print("Run analysis/resolution_horizon_analysis.py first.")
        return

    props_df = pd.read_csv(props_path)
    print(f"Loaded proportions: {len(props_df):,} rows")

    try:
        adata_ref, ct_col = load_intestine_reference(data_dir)
        visibility_df = compute_hvg_blindness(adata_ref, ct_col)
        visibility_df.to_csv(output_dir / "cell_type_visibility.csv", index=False)
    except FileNotFoundError:
        print("Reference not found; writing neutral visibility table.")
        cell_cols = [c for c in props_df.columns if c not in {"bin_size", "coord_x", "coord_y", "spot_id"}]
        visibility_df = pd.DataFrame(
            {
                "cell_type": cell_cols,
                "hvg_blindness": np.zeros(len(cell_cols)),
            }
        )
        visibility_df.to_csv(output_dir / "cell_type_visibility.csv", index=False)

    finest_props = props_df[props_df["bin_size"] == props_df["bin_size"].min()].copy() if "bin_size" in props_df else props_df
    coloc_df = analyze_colocalization(finest_props)
    coloc_df.to_csv(output_dir / "tuft_colocalization.csv", index=False)

    create_tuft_discovery_figure(props_df, visibility_df, output_dir, coloc_df=coloc_df, prefix="tuft_stem_discovery")
    validation_results = run_tuft_validation(props_df, output_dir)

    print("\nSUMMARY")
    tuft_row = visibility_df[visibility_df["cell_type"].str.contains("brush", case=False, na=False)]
    if not tuft_row.empty:
        print(f"Tuft cell HVG blindness: {tuft_row['hvg_blindness'].iloc[0]:.1f}%")
    stem_row = coloc_df[coloc_df["cell_type"].str.contains("stem", case=False, na=False)]
    if not stem_row.empty:
        print(f"Stem cell enrichment in Tuft niches: {stem_row['enrichment'].iloc[0]:.1f}x")
    if validation_results:
        print(f"Moran's I spatial autocorrelation: {validation_results['morans_i']:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
