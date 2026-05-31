#!/usr/bin/env python
"""Figure 7: Tuft-Stem Cell Niche Discovery."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repro.plotting import create_tuft_discovery_figure, load_tuft_figure_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 7: Tuft-Stem Cell Niche Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory containing multiscale analysis results")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Directory for output figures")
    args = parser.parse_args()

    props, visibility = load_tuft_figure_data(args.results_dir)
    create_tuft_discovery_figure(props, visibility, args.output_dir)
    print("Done")


if __name__ == "__main__":
    main()
