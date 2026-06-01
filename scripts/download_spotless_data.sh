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

set -euo pipefail

OUTPUT_DIR="${1:-./data/spotless}"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Downloading Spotless Benchmark Data"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Zenodo base URL
ZENODO_BASE="https://zenodo.org/records/10277187/files"

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "$dest" ]; then
        echo "  Already downloaded: $(basename "$dest"), skipping."
        return 0
    fi

    echo "  Downloading $desc..."
    if ! curl -fL --retry 3 --retry-delay 5 -o "$dest" "$url"; then
        echo "ERROR: Failed to download $desc from $url" >&2
        echo "  Check your internet connection and that the Zenodo record is accessible:" >&2
        echo "  https://zenodo.org/records/10277187" >&2
        rm -f "$dest"  # remove partial download
        return 1
    fi

    # Verify download is not empty / HTML error page
    local size
    size=$(wc -c < "$dest")
    if [ "$size" -lt 1000 ]; then
        echo "ERROR: Downloaded file is suspiciously small ($size bytes)." >&2
        echo "  The Zenodo record may have moved or require authentication." >&2
        rm -f "$dest"
        return 1
    fi
}

# Download all three archives
echo ""
echo "[1/3] Silver & Gold standards (standards.tar.gz, ~6.9 GB)..."
download_file "$ZENODO_BASE/standards.tar.gz?download=1" \
              "$OUTPUT_DIR/standards.tar.gz" \
              "Silver & Gold standards"

echo ""
echo "[2/3] Liver case study (liver_datasets.tar.gz, ~4.5 GB)..."
download_file "$ZENODO_BASE/liver_datasets.tar.gz?download=1" \
              "$OUTPUT_DIR/liver_datasets.tar.gz" \
              "Liver datasets"

echo ""
echo "[3/3] Melanoma case study (melanoma_datasets.tar.gz, ~1.3 GB)..."
download_file "$ZENODO_BASE/melanoma_datasets.tar.gz?download=1" \
              "$OUTPUT_DIR/melanoma_datasets.tar.gz" \
              "Melanoma datasets"

echo ""
echo "============================================================"
echo "Downloads complete. Extracting archives..."
echo "============================================================"

extract_archive() {
    local archive="$1"
    local name
    name="$(basename "$archive")"

    if [ ! -f "$archive" ]; then
        echo "WARNING: $name not found, skipping extraction." >&2
        return 1
    fi

    echo "  Extracting $name..."
    if ! tar -xzf "$archive" -C "$OUTPUT_DIR"; then
        echo "ERROR: Failed to extract $name. The file may be corrupted." >&2
        echo "  Delete it and re-run this script to re-download:" >&2
        echo "    rm $archive" >&2
        return 1
    fi
}

extract_archive "$OUTPUT_DIR/standards.tar.gz"
extract_archive "$OUTPUT_DIR/liver_datasets.tar.gz"
extract_archive "$OUTPUT_DIR/melanoma_datasets.tar.gz"

echo ""
echo "============================================================"
echo "Validating extracted directory structure..."
echo "============================================================"

ERRORS=0

check_dir() {
    if [ -d "$OUTPUT_DIR/$1" ]; then
        local count
        count=$(find "$OUTPUT_DIR/$1" -name "*.rds" | wc -l | tr -d ' ')
        echo "  OK  $1/ ($count RDS files)"
    else
        echo "  MISSING  $1/"
        ERRORS=$((ERRORS + 1))
    fi
}

check_dir "reference"

silver_count=0
if [ -d "$OUTPUT_DIR/test_silver_standard" ]; then
    legacy_count=$(find "$OUTPUT_DIR/test_silver_standard" -maxdepth 1 -name "*.rds" 2>/dev/null | wc -l | tr -d ' ')
    silver_count=$((silver_count + legacy_count))
fi
current_count=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type f -path "*/silver_standard_*-*/*.rds" 2>/dev/null | wc -l | tr -d ' ')
silver_count=$((silver_count + current_count))
if [ "$silver_count" -ge 56 ]; then
    echo "  OK  Silver Standard test datasets ($silver_count RDS files)"
else
    echo "  MISSING  Silver Standard test datasets (found $silver_count RDS files)"
    ERRORS=$((ERRORS + 1))
fi
check_dir "gold_standard_1"
check_dir "gold_standard_2"
check_dir "gold_standard_3"
check_dir "liver"
check_dir "melanoma"

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "WARNING: $ERRORS expected directories are missing." >&2
    echo "  The tarball structure may differ from what this script expects." >&2
    echo "  Please check the Zenodo page and inspect the extracted contents:" >&2
    echo "    ls $OUTPUT_DIR/" >&2
    echo "    tar -tzf $OUTPUT_DIR/standards.tar.gz | head -30" >&2
    echo ""
    echo "  If tarballs extract with a top-level directory (e.g., standards/)," >&2
    echo "  move the contents up:" >&2
    echo "    mv $OUTPUT_DIR/standards/* $OUTPUT_DIR/" >&2
else
    echo ""
    echo "All expected directories found."
fi

# --- Checksum verification ---

echo ""
echo "============================================================"
echo "Verifying checksums against Zenodo-published MD5 values..."
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_checksums.sh"
if [ -x "$VERIFY_SCRIPT" ]; then
    bash "$VERIFY_SCRIPT" "$OUTPUT_DIR"
else
    echo "  SKIP  verify_checksums.sh not found; checksums not verified."
fi

echo ""
echo "============================================================"
echo "Done! Data is ready in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Convert to Python-readable format:"
echo "     Rscript scripts/convert_spotless_data.R $OUTPUT_DIR"
echo "  2. Run benchmarks:"
echo "     python benchmarks/benchmark_silver_standards.py --data_dir $OUTPUT_DIR/converted"
echo "============================================================"
