#!/usr/bin/env python
"""Figure 2: Leverage scores decouple biological identity from population abundance."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repro.plotting import create_leverage_figure, load_leverage_figure_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 2: Leverage Mechanism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory containing leverage analysis results")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Directory for output figures")
    parser.add_argument("--visium_path", type=str, default=None, help="Path to Visium h5ad file (optional, for Panel D)")
    args = parser.parse_args()

    data = load_leverage_figure_data(args.results_dir)
    create_leverage_figure(data, args.output_dir, args.visium_path)
    print("Done")


if __name__ == "__main__":
    main()
