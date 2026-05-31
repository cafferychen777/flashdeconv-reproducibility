# FlashDeconv Reproducibility

This repository contains code and data links for reproducing the results in the FlashDeconv paper.

> **FlashDeconv enables atlas-scale, multi-resolution spatial deconvolution via structure-preserving sketching**
>
> Chen Yang, Jun Chen, Xianyang Zhang. *bioRxiv* (2025). DOI: [10.64898/2025.12.22.696108](https://doi.org/10.64898/2025.12.22.696108)

## Quick Start

```bash
# 1. Install FlashDeconv
pip install flashdeconv

# 2. Download benchmark data (requires ~13 GB)
bash scripts/download_spotless_data.sh ./data/spotless

# 3. Validate the download (checks directory structure and file counts)
bash scripts/validate_data.sh ./data/spotless

# 4. Convert data to Python-readable format
Rscript scripts/convert_spotless_data.R ./data/spotless

# 5. Validate again (now checks converted files too)
bash scripts/validate_data.sh ./data/spotless

# 6. Run benchmarks
python benchmarks/benchmark_silver_standards.py --data_dir ./data/spotless/converted
python benchmarks/benchmark_gold_standard.py --data_dir ./data/spotless/converted
python benchmarks/benchmark_liver.py --data_dir ./data/spotless/converted
python benchmarks/benchmark_melanoma.py --data_dir ./data/spotless/converted
python benchmarks/benchmark_scalability.py --output_dir ./results
```

## Notebook Reproducibility Layer

The command-line scripts remain the canonical full reproduction path. For
interactive inspection of the main narrative analyses, this repository also
provides Jupyter notebooks under `notebooks/`.

```bash
pip install -r requirements-notebooks.txt

# Fast synthetic/data-check path
export FD_REPRO_MODE=smoke
jupyter notebook notebooks/

# Full-data path after completing the data preparation steps below
export FD_REPRO_MODE=full
export FD_DATA_DIR=./data
export FD_RESULTS_DIR=./results
```

The first notebook release covers Visium HD/tuft discovery, cortex lamination,
and the leverage-score mechanism. Benchmark notebooks are intentionally deferred:
benchmarks are better run as CLI jobs and inspected from their CSV outputs.

## Repository Structure

```
flashdeconv-reproducibility/
├── README.md                           # This file
├── benchmarks/                         # Benchmark scripts
│   ├── benchmark_silver_standards.py   # Silver Standard (56 synthetic datasets)
│   ├── benchmark_gold_standard.py      # Gold Standard (STARMap + seqFISH+)
│   ├── benchmark_liver.py              # Liver case study
│   ├── benchmark_melanoma.py           # Melanoma case study
│   └── benchmark_scalability.py        # Scalability benchmark (1K-100K spots)
├── analysis/                           # Analysis pipelines
│   ├── leverage_deep_dive.py           # Leverage mechanism analysis (Figure 2)
│   ├── resolution_horizon_analysis.py  # Visium HD multi-scale analysis (Figure 6)
│   └── hidden_cell_analysis.py         # Hidden cell type discovery
├── figures/                            # Figure generation scripts
│   ├── main/                           # Main paper figures
│   │   ├── figure2_leverage_mechanism.py
│   │   ├── figure5_cortex_lamination.py
│   │   └── figure7_tuft_discovery.py
│   └── supplementary/                  # Supplementary figures
├── notebooks/                          # Narrative smoke/full reproduction notebooks
├── repro/                              # Shared helpers used by scripts and notebooks
├── scripts/                            # Data preparation scripts
│   ├── download_spotless_data.sh       # Download Spotless data from Zenodo
│   ├── convert_spotless_data.R         # Convert RDS to MTX format
│   ├── validate_data.sh               # Validate downloaded/converted data
│   ├── download_visium_hd_data.sh      # Download Visium HD data
│   └── prepare_haber_reference.py      # Prepare Haber et al. reference
└── results/                            # Output directory for results
```

## Software

The FlashDeconv software package is available at: https://github.com/cafferychen777/flashdeconv

```bash
pip install flashdeconv
```

## Data Sources

All datasets used in this study are publicly available.

### 1. Spotless Benchmark Datasets

The primary benchmark datasets are from the Spotless benchmark study ([Sang-aram et al., 2024](https://elifesciences.org/articles/88431)).

| Dataset | Description | Source |
|---------|-------------|--------|
| **Silver Standard** | 56 selected synthetic datasets (6 tissues; replicate 1 from each abundance-pattern directory) | [Zenodo](https://zenodo.org/records/10277187) |
| **Gold Standard** | STARMap + seqFISH+ (real spatial transcriptomics) | [Zenodo](https://zenodo.org/records/10277187) |
| **Liver Case Study** | Mouse liver Visium sections (4 samples) | [Zenodo](https://zenodo.org/records/10277187) |
| **Melanoma Case Study** | Mouse melanoma tumor sections (3 samples) | [Zenodo](https://zenodo.org/records/10277187) |

**Download**: https://zenodo.org/records/10277187

Files:
- `standards.tar.gz` (6.9 GB) - Silver and Gold standards
- `liver_datasets.tar.gz` (4.5 GB) - Liver case study
- `melanoma_datasets.tar.gz` (1.3 GB) - Melanoma case study

After downloading and extracting (via `download_spotless_data.sh`), the expected directory structure is:

```
data/spotless/
├── reference/                          # Silver Standard scRNA-seq references (6 RDS files)
│   ├── silver_standard_1_brain_cortex.rds
│   ├── silver_standard_2_cerebellum_cell.rds
│   ├── silver_standard_3_cerebellum_nucleus.rds
│   ├── silver_standard_4_hippocampus.rds
│   ├── silver_standard_5_kidney.rds
│   └── silver_standard_6_scc_p5.rds
├── silver_standard_1-1/                # Silver Standard pseudo-spots; each folder has replicate RDS files
│   ├── brain_cortex_artificial_uniform_distinct_rep1.rds
│   ├── ...
│   └── brain_cortex_artificial_uniform_distinct_rep10.rds
├── silver_standard_1-2/
├── ...                                 # brain_cortex has patterns 1-11; others have 1-9
├── silver_standard_6-9/
├── gold_standard_1/                    # seqFISH+ cortex (Eng et al. 2019)
├── gold_standard_2/                    # seqFISH+ olfactory bulb (Eng et al. 2019)
├── gold_standard_3/                    # STARMap (Wang et al. 2018)
├── liver/                              # Liver case study (5 RDS files)
│   ├── liver_mouseStSt_9celltypes.rds  # snRNA-seq reference
│   ├── liver_mouseVisium_JB01.rds      # Visium sample 1
│   ├── liver_mouseVisium_JB02.rds
│   ├── liver_mouseVisium_JB03.rds
│   └── liver_mouseVisium_JB04.rds
└── melanoma/                           # Melanoma case study (4 RDS files)
    ├── melanoma_scrna_ref.rds          # scRNA-seq reference
    ├── melanoma_visium_sample02.rds    # Visium sample 2
    ├── melanoma_visium_sample03.rds
    └── melanoma_visium_sample04.rds
```

The converter accepts this current Spotless layout and older flat `test_silver_standard/*.rds` layouts. For the 56-dataset benchmark scripts, `scripts/convert_spotless_data.R` uses the `rep1` file from each `silver_standard_<dataset>-<pattern>/` directory and writes it as `converted/silver_<dataset>_<pattern>_*`.

Use `bash scripts/validate_data.sh ./data/spotless` to verify this structure after extraction.

---

### 2. 10x Genomics Datasets

#### Visium HD Mouse Small Intestine

Used for resolution horizon analysis and rare cell type discovery.

| Resource | Link |
|----------|------|
| **Dataset Page** | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-intestine) |
| **License** | CC BY 4.0 |

```bash
curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz
```

#### Xenium Fresh Frozen Mouse Colon

Ground truth validation dataset with single-cell resolution.

| Resource | Link |
|----------|------|
| **Dataset Page** | [10x Genomics](https://www.10xgenomics.com/datasets/fresh-frozen-mouse-colon-with-xenium-multimodal-cell-segmentation-1-standard) |
| **Cells Detected** | 219,797 |
| **License** | CC BY 4.0 |

```bash
curl -O https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_mouse_Colon_FF/Xenium_V1_mouse_Colon_FF_outs.zip
```

---

### 3. Cell2location Mouse Brain Dataset

Used for cortical lamination validation.

| Resource | Link |
|----------|------|
| **Data Portal** | [cell2location.cog.sanger.ac.uk](https://cell2location.cog.sanger.ac.uk/tutorial/) |
| **ArrayExpress** | [E-MTAB-11114](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11114) (Visium), [E-MTAB-11115](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11115) (snRNA-seq) |

```bash
# Download pre-processed tutorial data
curl -O https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_visium_wo_cloupe_data.zip
curl -O https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_snrna/regression_model/RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes/sc.h5ad
```

---

### 4. Intestinal scRNA-seq Reference (Haber et al., 2017)

Single-cell reference for intestinal deconvolution.

| Resource | Link |
|----------|------|
| **Pre-processed (Zenodo)** | [zenodo.org/records/4447233](https://zenodo.org/records/4447233) |
| **Original GEO** | [GSE92332](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92332) |
| **Publication** | [Haber et al., Nature 2017](https://www.nature.com/articles/nature24489) |

```bash
curl -L -o haber_processed.h5ad "https://zenodo.org/records/4447233/files/haber_processed.h5ad?download=1"
```

---

## Reproducing Results

### Prerequisites

```bash
# Python dependencies
pip install flashdeconv scanpy anndata pandas matplotlib seaborn psutil

# R dependencies (for data conversion)
R -e "install.packages(c('Seurat', 'Matrix'))"
```

### Part 1: Spotless Benchmark

#### Step 1: Download Data

```bash
# Download Spotless benchmark data (~13 GB total)
bash scripts/download_spotless_data.sh ./data/spotless

# Validate directory structure
bash scripts/validate_data.sh ./data/spotless
```

#### Step 2: Convert Data

```bash
# Convert RDS files to MTX format
Rscript scripts/convert_spotless_data.R ./data/spotless

# Validate converted files
bash scripts/validate_data.sh ./data/spotless
```

The conversion script prints a summary showing how many datasets were converted vs. skipped. If many are skipped, see the [Troubleshooting](#troubleshooting) section.

#### Step 3: Run Benchmarks

```bash
# Silver Standard benchmark (56 synthetic datasets)
python benchmarks/benchmark_silver_standards.py \
    --data_dir ./data/spotless/converted \
    --output_dir ./results

# Gold Standard benchmark (STARMap + seqFISH+)
python benchmarks/benchmark_gold_standard.py \
    --data_dir ./data/spotless/converted \
    --output_dir ./results

# Case studies
python benchmarks/benchmark_liver.py \
    --data_dir ./data/spotless/converted \
    --output_dir ./results

python benchmarks/benchmark_melanoma.py \
    --data_dir ./data/spotless/converted \
    --output_dir ./results

# Scalability benchmark (no external data needed)
python benchmarks/benchmark_scalability.py \
    --output_dir ./results
```

### Part 2: Visium HD Analysis (Resolution Horizon)

This analysis demonstrates FlashDeconv's scale-space capability, showing how
different cell types have characteristic spatial scales and identifying the
"resolution horizon" (8-16μm threshold).

#### Step 1: Download Data

```bash
# Download Visium HD and reference data (~6.2 GB)
bash scripts/download_visium_hd_data.sh ./data
```

#### Step 2: Prepare Reference

```bash
# Prepare Haber et al. reference with cell type annotations
python scripts/prepare_haber_reference.py --data-dir ./data --output ./data/haber_intestine_reference.h5ad
```

#### Step 3: Run Analysis

```bash
# Resolution horizon analysis (multi-scale deconvolution)
python analysis/resolution_horizon_analysis.py \
    --data_dir ./data \
    --output_dir ./results \
    --bins 16,32,64,128

# Generate Figure 7: Tuft-Stem discovery
python figures/main/figure7_tuft_discovery.py \
    --results_dir ./results \
    --output_dir ./figures
```

### Part 3: Cortex Lamination (Figure 5)

This analysis validates FlashDeconv using Cell2location's paired mouse brain dataset.

#### Step 1: Download Data

```bash
# Download Cell2location mouse brain data
mkdir -p ./data/mouse_brain/C2L/ST/48/spatial
cd ./data/mouse_brain

# scRNA-seq reference
curl -o scrna_reference.h5ad https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_snrna/regression_model/RegressionGeneBackgroundCoverageTorch_65covariates_40532cells_12819genes/sc.h5ad

# Cell annotation
curl -o cell_annotation.csv https://cell2location.cog.sanger.ac.uk/tutorial/mouse_brain_snrna_sc_adata_annotation.csv

# Visium data (ST8059048)
cd C2L/ST/48
curl -o ST8059048_filtered_feature_bc_matrix.h5 "https://cell2location.cog.sanger.ac.uk/tutorial/ST/ST8059048/filtered_feature_bc_matrix.h5"
cd spatial
curl -o tissue_positions_list.csv "https://cell2location.cog.sanger.ac.uk/tutorial/ST/ST8059048/spatial/tissue_positions_list.csv"
cd ../../../../..
```

#### Step 2: Run Analysis

```bash
# Run deconvolution (generates level2_v3_data.npz)
python analysis/cortex_deconvolution.py \
    --data_dir ./data \
    --output_dir ./results

# Generate Figure 5
python figures/main/figure5_cortex_lamination.py \
    --results_dir ./results \
    --output_dir ./figures
```

### Part 4: Leverage Mechanism (Figure 2)

This analysis demonstrates how leverage scores decouple biological identity from population abundance.

```bash
# Run leverage analysis
python analysis/leverage_deep_dive.py \
    --data_dir ./data \
    --output_dir ./results

# Generate Figure 2
python figures/main/figure2_leverage_mechanism.py \
    --results_dir ./results \
    --output_dir ./figures
```

---

## Troubleshooting

### Conversion script skips most files

If `convert_spotless_data.R` reports many `[SKIP]` messages, the tarballs likely extracted with a top-level directory (e.g., `standards/reference/` instead of `reference/`). Check:

```bash
ls ./data/spotless/
```

If you see a `standards/` subdirectory instead of `reference/` and `test_silver_standard/`, move its contents up:

```bash
mv ./data/spotless/standards/* ./data/spotless/
rmdir ./data/spotless/standards
```

Then re-run the conversion script.

### Download fails or produces small files

The Spotless data is hosted on Zenodo ([record 10277187](https://zenodo.org/records/10277187)). If downloads fail:

1. Check that the Zenodo record is accessible in your browser
2. Check your internet connection and any institutional proxy settings
3. For large files, `curl` may time out — retry with a longer timeout:
   ```bash
   curl -fL --max-time 3600 -o file.tar.gz "URL"
   ```
4. As a fallback, download the tarballs manually from the Zenodo page and place them in `./data/spotless/`

### Validation script reports missing files

Run `bash scripts/validate_data.sh ./data/spotless` to see exactly which directories or files are missing. The script checks both raw RDS files and converted MTX files.

---

## Citation

If you use FlashDeconv or this reproducibility code, please cite:

> Yang, C., Chen, J. & Zhang, X. FlashDeconv enables atlas-scale, multi-resolution spatial deconvolution via structure-preserving sketching. *bioRxiv* (2025). https://doi.org/10.64898/2025.12.22.696108

```bibtex
@article{yang2025flashdeconv,
  title={FlashDeconv enables atlas-scale, multi-resolution spatial deconvolution via structure-preserving sketching},
  author={Yang, Chen and Chen, Jun and Zhang, Xianyang},
  journal={bioRxiv},
  year={2025},
  doi={10.64898/2025.12.22.696108},
  url={https://doi.org/10.64898/2025.12.22.696108}
}
```

## License

- **Code**: GPL-3.0
- **Data**: See individual dataset licenses above

## References

1. Sang-aram, C., Browaeys, R., Seurinck, R., & Saeys, Y. (2024). Spotless, a reproducible pipeline for benchmarking cell type deconvolution in spatial transcriptomics. *eLife*, 12, RP88431.

2. Kleshchevnikov, V., Shmatko, A., Dann, E., et al. (2022). Cell2location maps fine-grained cell types in spatial transcriptomics. *Nature Biotechnology*, 40, 661-671.

3. Haber, A. L., Biton, M., Rogel, N., et al. (2017). A single-cell survey of the small intestinal epithelium. *Nature*, 551, 333-339.
