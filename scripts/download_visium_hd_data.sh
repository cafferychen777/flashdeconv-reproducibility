#!/bin/bash
#
# Download Visium HD Mouse Small Intestine data for FlashDeconv analysis
#
# Usage:
#     bash download_visium_hd_data.sh [DATA_DIR]
#
# Example:
#     bash download_visium_hd_data.sh ./data
#
# This script downloads:
# 1. Visium HD Mouse Small Intestine binned outputs from 10x Genomics (~6 GB)
# 2. Haber et al. intestine scRNA-seq reference from Zenodo (~160 MB)
#

DATA_DIR=${1:-./data}
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "Downloading Visium HD and reference data"
echo "Output directory: $DATA_DIR"
echo "============================================================"

# Download Visium HD data from 10x Genomics
VISIUM_URL="https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz"
VISIUM_FILE="$DATA_DIR/Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz"

if [ -d "$DATA_DIR/Visium_HD_Mouse_Small_Intestine_binned_outputs" ]; then
    echo ""
    echo "Visium HD data already exists, skipping download..."
else
    echo ""
    echo "Downloading Visium HD Mouse Small Intestine data (~6 GB)..."
    echo "Source: 10x Genomics"
    echo "URL: $VISIUM_URL"
    echo ""

    if [ -f "$VISIUM_FILE" ]; then
        echo "Archive already downloaded, extracting..."
    else
        curl -L -o "$VISIUM_FILE" "$VISIUM_URL"
    fi

    echo "Extracting Visium HD data..."
    tar -xzf "$VISIUM_FILE" -C "$DATA_DIR"
    echo "Done!"
fi

# Download Haber et al. reference
HABER_URL="https://zenodo.org/records/4447233/files/haber_processed.h5ad?download=1"
HABER_FILE="$DATA_DIR/haber_processed.h5ad"

if [ -f "$HABER_FILE" ]; then
    echo ""
    echo "Haber reference already exists, skipping download..."
else
    echo ""
    echo "Downloading Haber et al. intestine scRNA-seq reference (~160 MB)..."
    echo "Source: Zenodo (Cell2location tutorial data)"
    echo "URL: $HABER_URL"
    echo ""

    curl -L -o "$HABER_FILE" "$HABER_URL"
    echo "Done!"
fi

echo ""
echo "============================================================"
echo "Download complete!"
echo "============================================================"
echo ""
echo "Downloaded files:"
ls -lh "$DATA_DIR"/*.h5ad 2>/dev/null || true
ls -ld "$DATA_DIR"/Visium_HD_* 2>/dev/null || true
echo ""
echo "Next steps:"
echo "1. Run: python scripts/prepare_haber_reference.py --data_dir $DATA_DIR"
echo "2. Run: python analysis/resolution_horizon_analysis.py --data_dir $DATA_DIR"
