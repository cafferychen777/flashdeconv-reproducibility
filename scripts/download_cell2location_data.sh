#!/bin/bash
#
# Download Cell2location mouse brain data used for cortex lamination and leverage analyses.
#
# Usage:
#     bash scripts/download_cell2location_data.sh [DATA_DIR]
#
# Example:
#     bash scripts/download_cell2location_data.sh ./data/mouse_brain

set -euo pipefail

DATA_DIR="${1:-./data/mouse_brain}"
mkdir -p "$DATA_DIR"
ANALYSIS_DATA_DIR="$(dirname "$DATA_DIR")"

SCRNA_URL="https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_snrna/regression_model/RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes/sc.h5ad"
VISIUM_URL="https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_visium_wo_cloupe_data.zip"

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "$dest" ]; then
        echo "  Already available: $(basename "$dest")"
        return 0
    fi

    echo "  Downloading $desc..."
    if ! curl -fL --retry 3 --retry-delay 5 -o "$dest" "$url"; then
        echo "ERROR: Failed to download $desc from $url" >&2
        rm -f "$dest"
        exit 1
    fi
}

echo "============================================================"
echo "Downloading Cell2location mouse brain data"
echo "Output directory: $DATA_DIR"
echo "============================================================"

download_file "$SCRNA_URL" "$DATA_DIR/scrna_reference.h5ad" "snRNA-seq reference with annotations"

VISIUM_ZIP="$DATA_DIR/mouse_brain_visium_wo_cloupe_data.zip"
VISIUM_DIR="$DATA_DIR/mouse_brain_visium_wo_cloupe_data"

if [ -d "$VISIUM_DIR" ]; then
    echo "  Spatial Visium archive already extracted."
else
    download_file "$VISIUM_URL" "$VISIUM_ZIP" "Visium spatial archive"
    echo "  Extracting Visium spatial archive..."
    python3 -m zipfile -e "$VISIUM_ZIP" "$DATA_DIR"
fi

MATRIX="$VISIUM_DIR/rawdata/ST8059048/filtered_feature_bc_matrix.h5"
POSITIONS="$VISIUM_DIR/rawdata/ST8059048/spatial/tissue_positions_list.csv"

if [ ! -f "$MATRIX" ] || [ ! -f "$POSITIONS" ]; then
    echo "ERROR: Expected ST8059048 files were not found after extraction." >&2
    echo "  Missing matrix:    $MATRIX" >&2
    echo "  Missing positions: $POSITIONS" >&2
    exit 1
fi

echo ""
echo "Downloaded files:"
echo "  $DATA_DIR/scrna_reference.h5ad"
echo "  $MATRIX"
echo "  $POSITIONS"
echo ""
echo "Next steps:"
echo "  python analysis/cortex_deconvolution.py --data_dir $ANALYSIS_DATA_DIR --output_dir ./results"
echo "  python analysis/leverage_deep_dive.py --data_dir $ANALYSIS_DATA_DIR --output_dir ./results"
