#!/bin/bash
# Download datasets for the cross-cohort evidence analyses.
# Run on arseven login node (downloads only, no computation)

set -euo pipefail

BASE="/scratch/user/cafferychen777/FlashDeconv/analysis/cross_cohort_evidence/data"
mkdir -p "$BASE"/{pelka,schurch_codex,tcga}

echo "=== Downloading datasets for cross-cohort evidence analyses ==="
echo "Date: $(date)"

# ---------------------------------------------------------------
# 1. Marteau Xenium (25 GB) — already started separately
# ---------------------------------------------------------------
if [ ! -f "$BASE/crca_xenium.h5ad" ] || [ $(stat -c%s "$BASE/crca_xenium.h5ad" 2>/dev/null || echo 0) -lt 1000000000 ]; then
    echo ""
    echo "[1/5] Marteau Xenium h5ad (25 GB) — downloading..."
    wget -q --show-progress -O "$BASE/crca_xenium.h5ad" \
        "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/208/S-BIAD2208/Files/xenium/processed/crca_xenium.h5ad"
else
    echo "[1/5] Marteau Xenium h5ad — already downloaded ($(du -sh "$BASE/crca_xenium.h5ad" | cut -f1))"
fi

# ---------------------------------------------------------------
# 2. Marteau scRNA-seq atlas (for L-R analysis on full atlas)
# ---------------------------------------------------------------
if [ ! -f "$BASE/MUI_Innsbruck-adata.h5ad" ]; then
    echo ""
    echo "[2/5] Marteau scRNA-seq atlas (2.2 GB)..."
    wget -q --show-progress -O "$BASE/MUI_Innsbruck-adata.h5ad" \
        "https://zenodo.org/records/16631519/files/MUI_Innsbruck-adata.h5ad?download=1"
else
    echo "[2/5] Marteau scRNA-seq atlas — already downloaded"
fi

# ---------------------------------------------------------------
# 3. Pelka 2021 scRNA-seq (from Broad SCP or GEO)
# ---------------------------------------------------------------
echo ""
echo "[3/5] Pelka 2021 CRC scRNA-seq..."
cd "$BASE/pelka"

# Try downloading from CellxGene / Broad SCP
# The Broad SCP1162 data requires login, so try GEO instead
# GEO GSE178341 has raw counts, but we need processed data

# Option A: Try downloading processed atlas from Broad SCP (public link)
if [ ! -f "pelka_crc_atlas.h5ad" ]; then
    echo "  Attempting download from Broad SCP..."
    # The SCP1162 data might need auth; try direct links
    wget -q --show-progress -O pelka_metadata.csv.gz \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl/GSE178341_all_cells_metadata.csv.gz" \
        2>/dev/null || echo "  GEO metadata download failed, will try alternatives"

    # Try count matrix
    wget -q --show-progress -O pelka_counts.h5 \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl/GSE178341_all_cells_raw_counts.h5" \
        2>/dev/null || echo "  GEO counts download failed"
else
    echo "  Already downloaded"
fi

# ---------------------------------------------------------------
# 4. Schürch 2020 CODEX (cell-level data from supplementary)
# ---------------------------------------------------------------
echo ""
echo "[4/5] Schürch 2020 CODEX cell data..."
cd "$BASE/schurch_codex"

# The paper's supplementary table has cell coordinates and phenotypes
# Table S1 from the Cell paper supplementary
if [ ! -f "CRC_clusters_neighborhoods_markers.csv" ]; then
    echo "  Downloading supplementary cell data..."
    # Try the direct Cell supplementary link
    wget -q --show-progress -O "schurch_table_s1.xlsx" \
        "https://ars.els-cdn.com/content/image/1-s2.0-S0092867420308709-mmc2.xlsx" \
        2>/dev/null || true

    # Alternative: processed cell data from GitHub/Zenodo
    wget -q --show-progress -O "CRC_clusters_neighborhoods_markers.csv" \
        "https://raw.githubusercontent.com/nolanlab/CRC_atlas/master/data/CRC_clusters_neighborhoods_markers.csv" \
        2>/dev/null || echo "  Direct CSV download failed, trying alternative..."

    # Another try - the Nolan lab GitHub
    if [ ! -f "CRC_clusters_neighborhoods_markers.csv" ]; then
        wget -q --show-progress -O "CRC_clusters_neighborhoods_markers.csv" \
            "https://zenodo.org/api/records/3634104/files/CRC_clusters_neighborhoods_markers.csv/content" \
            2>/dev/null || echo "  Zenodo download also failed"
    fi
else
    echo "  Already downloaded"
fi

# ---------------------------------------------------------------
# 5. TCGA-COAD (expression + survival)
# ---------------------------------------------------------------
echo ""
echo "[5/5] TCGA-COAD data..."
cd "$BASE/tcga"

if [ ! -f "tcga_coad_fpkm.tsv.gz" ]; then
    echo "  Downloading expression data..."
    wget -q --show-progress -O tcga_coad_fpkm.tsv.gz \
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-COAD.htseq_fpkm.tsv.gz" \
        2>/dev/null || echo "  Expression download failed"
fi

if [ ! -f "tcga_coad_survival.tsv" ]; then
    echo "  Downloading survival data..."
    wget -q --show-progress -O tcga_coad_survival.tsv \
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-COAD.survival.tsv" \
        2>/dev/null || echo "  Survival download failed"
fi

if [ ! -f "tcga_coad_phenotype.tsv.gz" ]; then
    echo "  Downloading phenotype data..."
    wget -q --show-progress -O tcga_coad_phenotype.tsv.gz \
        "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-COAD.GDC_phenotype.tsv.gz" \
        2>/dev/null || echo "  Phenotype download failed"
fi

echo ""
echo "=== Download status ==="
echo "Marteau Xenium:"
ls -lh "$BASE/crca_xenium.h5ad" 2>/dev/null || echo "  NOT FOUND"
echo "Marteau scRNA:"
ls -lh "$BASE/MUI_Innsbruck-adata.h5ad" 2>/dev/null || echo "  NOT FOUND"
echo "Pelka:"
ls -lh "$BASE/pelka/"* 2>/dev/null || echo "  NOT FOUND"
echo "Schürch CODEX:"
ls -lh "$BASE/schurch_codex/"* 2>/dev/null || echo "  NOT FOUND"
echo "TCGA:"
ls -lh "$BASE/tcga/"* 2>/dev/null || echo "  NOT FOUND"

echo ""
echo "Done: $(date)"
