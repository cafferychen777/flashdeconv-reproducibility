#!/bin/bash
# Download Spotless Benchmark data from Zenodo
#
# This script downloads the Spotless benchmark datasets required to reproduce
# FlashDeconv's benchmark results.
#
# Data source: https://zenodo.org/records/10277187
#
# Usage:
#     bash download_spotless_data.sh [OUTPUT_DIR]
#
# Example:
#     bash download_spotless_data.sh ./data/spotless

set -e

OUTPUT_DIR="${1:-./data/spotless}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Downloading Spotless Benchmark Data"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Zenodo base URL
ZENODO_BASE="https://zenodo.org/records/10277187/files"

# Download Silver & Gold standards
echo ""
echo "[1/3] Downloading Silver & Gold standards (6.9 GB)..."
echo "This may take a while..."
curl -L -o "$OUTPUT_DIR/standards.tar.gz" "$ZENODO_BASE/standards.tar.gz?download=1"

# Download Liver datasets
echo ""
echo "[2/3] Downloading Liver case study data (4.5 GB)..."
curl -L -o "$OUTPUT_DIR/liver_datasets.tar.gz" "$ZENODO_BASE/liver_datasets.tar.gz?download=1"

# Download Melanoma datasets
echo ""
echo "[3/3] Downloading Melanoma case study data (1.3 GB)..."
curl -L -o "$OUTPUT_DIR/melanoma_datasets.tar.gz" "$ZENODO_BASE/melanoma_datasets.tar.gz?download=1"

echo ""
echo "============================================================"
echo "Download complete! Extracting files..."
echo "============================================================"

# Extract files
cd "$OUTPUT_DIR"

echo "Extracting standards.tar.gz..."
tar -xzf standards.tar.gz

echo "Extracting liver_datasets.tar.gz..."
tar -xzf liver_datasets.tar.gz

echo "Extracting melanoma_datasets.tar.gz..."
tar -xzf melanoma_datasets.tar.gz

echo ""
echo "============================================================"
echo "Done! Data is ready in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: Rscript scripts/convert_spotless_data.R $OUTPUT_DIR"
echo "  2. Run: python benchmarks/benchmark_silver_standards.py"
echo "============================================================"
