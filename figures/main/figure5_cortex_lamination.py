#!/usr/bin/env python
"""Figure 5: Cortical Layer Deconvolution."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repro.plotting import create_cortex_lamination_figure, load_cortex_figure_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 5: Cortex Lamination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory containing level2_v3_data.npz")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Directory for output figures")
    args = parser.parse_args()

    data = load_cortex_figure_data(args.results_dir)
    create_cortex_lamination_figure(data, args.output_dir)
    print("Done")


if __name__ == "__main__":
    main()
