#!/usr/bin/env python
"""Resolution Horizon Analysis on Visium HD Mouse Small Intestine."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro.plotting import create_resolution_figure
from repro.visium_hd import (
    compute_resolution_metrics,
    flatten_multiscale_proportions,
    run_multiscale_analysis,
)


def main():
    parser = argparse.ArgumentParser(
        description="Resolution Horizon Analysis on Visium HD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output files")
    parser.add_argument("--bins", type=str, default="16,32,64,128", help="Comma-separated bin sizes in um")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_sizes = sorted(int(x.strip()) for x in args.bins.split(","))

    print("=" * 60)
    print("RESOLUTION HORIZON ANALYSIS")
    print("=" * 60)
    print(f"Bin sizes: {bin_sizes} um")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    results, cell_types = run_multiscale_analysis(data_dir, output_dir, bin_sizes)
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        print("\nERROR: No successful deconvolution runs.")
        return

    metrics_df = compute_resolution_metrics(results, cell_types)
    metrics_df.to_csv(output_dir / "resolution_metrics.csv", index=False)
    create_resolution_figure(results, metrics_df, output_dir)

    props_df = flatten_multiscale_proportions(results)
    if not props_df.empty:
        props_df.to_csv(output_dir / "multiscale_proportions.csv", index=False)

    print("\nResolution horizon analysis for rare cell types:")
    for rare_ct in ["brush cell", "enteroendocrine cell"]:
        subset = metrics_df[metrics_df["cell_type"] == rare_ct].sort_values("bin_size")
        if subset.empty:
            continue
        print(f"\n  {rare_ct}:")
        for _, row in subset.iterrows():
            print(f"    {row['bin_size']:3d}um: max={row['max_prop'] * 100:.1f}%, detectable={row['pct_detectable']:.1f}%")

    print("\nDONE")


if __name__ == "__main__":
    main()
