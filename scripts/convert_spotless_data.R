#!/usr/bin/env Rscript
#
# Convert Spotless Benchmark RDS files to MTX format for FlashDeconv
#
# This script converts the Seurat objects from the Spotless benchmark
# into a format that can be read by Python scripts.
#
# Usage:
#     Rscript convert_spotless_data.R [DATA_DIR]
#
# Example:
#     Rscript convert_spotless_data.R ./data/spotless
#
# Requirements:
#     - R packages: Seurat, Matrix

library(Seurat)
library(Matrix)

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else "./data/spotless"
output_dir <- file.path(data_dir, "converted")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat(strrep("=", 60), "\n")
cat("Converting Spotless Benchmark Data\n")
cat("Input directory:", data_dir, "\n")
cat("Output directory:", output_dir, "\n")
cat(strrep("=", 60), "\n\n")

# ============================================================
# Helper function to save Seurat object
# ============================================================
save_seurat <- function(obj, prefix, is_spatial = FALSE) {
    # Get counts
    counts <- tryCatch({
        GetAssayData(obj, layer = "counts")
    }, error = function(e) {
        GetAssayData(obj, slot = "counts")
    })

    genes <- rownames(obj)

    if (is_spatial) {
        # For spatial data: save as spots x genes
        writeMM(t(counts), paste0(prefix, "_counts.mtx"))

        # Get coordinates
        coords <- tryCatch({
            GetTissueCoordinates(obj)
        }, error = function(e) {
            if (length(obj@images) > 0) {
                img_name <- names(obj@images)[1]
                coords_data <- obj@images[[img_name]]@coordinates
                data.frame(
                    x = coords_data$imagerow,
                    y = coords_data$imagecol
                )
            } else {
                NULL
            }
        })

        if (!is.null(coords)) {
            write.csv(coords, paste0(prefix, "_coords.csv"))
        }

        # Save metadata
        write.csv(obj@meta.data, paste0(prefix, "_metadata.csv"))

    } else {
        # For reference data: save as genes x cells
        writeMM(counts, paste0(prefix, "_counts.mtx"))

        # Get cell types
        celltypes <- NULL
        for (col in c("celltype", "cell_type", "annot_cd45", "cluster", "label")) {
            if (col %in% colnames(obj@meta.data)) {
                celltypes <- obj@meta.data[[col]]
                break
            }
        }

        if (is.null(celltypes)) {
            ct_cols <- grep("cell.*type|cluster|label", colnames(obj@meta.data),
                            ignore.case = TRUE, value = TRUE)
            if (length(ct_cols) > 0) {
                celltypes <- obj@meta.data[[ct_cols[1]]]
            }
        }

        if (!is.null(celltypes)) {
            writeLines(as.character(celltypes), paste0(prefix, "_celltypes.txt"))
        }

        cells <- colnames(obj)
        writeLines(cells, paste0(prefix, "_cells.txt"))
    }

    writeLines(genes, paste0(prefix, "_genes.txt"))
}

# ============================================================
# 1. Convert Silver Standard References
# ============================================================
cat("\n=== Converting Silver Standard References ===\n")

silver_refs <- list(
    "1" = "silver_standard_1_brain_cortex",
    "2" = "silver_standard_2_cerebellum_cell",
    "3" = "silver_standard_3_cerebellum_nucleus",
    "4" = "silver_standard_4_hippocampus",
    "5" = "silver_standard_5_kidney",
    "6" = "silver_standard_6_scc_p5"
)

ref_dir <- file.path(data_dir, "reference")

for (id in names(silver_refs)) {
    name <- silver_refs[[id]]
    rds_file <- file.path(ref_dir, paste0(name, ".rds"))

    if (!file.exists(rds_file)) {
        cat("  [SKIP] Reference", id, "not found:", rds_file, "\n")
        next
    }

    cat("  Converting reference", id, ":", name, "...\n")
    ref <- readRDS(rds_file)
    prefix <- file.path(output_dir, paste0("reference_", id))
    save_seurat(ref, prefix, is_spatial = FALSE)
    cat("    Cells:", ncol(ref), " Genes:", nrow(ref), "\n")
}

# ============================================================
# 2. Convert Silver Standard Test Datasets
# ============================================================
cat("\n=== Converting Silver Standard Test Datasets ===\n")

test_dir <- file.path(data_dir, "test_silver_standard")

silver_tests <- list(
    "1" = list(name = "brain_cortex", patterns = 11),
    "2" = list(name = "cerebellum_cell", patterns = 9),
    "3" = list(name = "cerebellum_nucleus", patterns = 9),
    "4" = list(name = "hippocampus", patterns = 9),
    "5" = list(name = "kidney", patterns = 9),
    "6" = list(name = "scc_p5", patterns = 9)
)

for (ds_id in names(silver_tests)) {
    info <- silver_tests[[ds_id]]
    cat("  Dataset", ds_id, ":", info$name, "\n")

    for (pattern_id in 1:info$patterns) {
        rds_file <- file.path(test_dir,
            paste0("silver_standard_", ds_id, "_", info$name, "_", pattern_id, ".rds"))

        if (!file.exists(rds_file)) {
            next
        }

        test <- readRDS(rds_file)

        # Get counts and proportions
        counts <- tryCatch({
            GetAssayData(test, layer = "counts")
        }, error = function(e) {
            GetAssayData(test, slot = "counts")
        })

        genes <- rownames(test)

        # Get proportions from metadata
        prop_cols <- grep("^prob_", colnames(test@meta.data), value = TRUE)
        if (length(prop_cols) == 0) {
            prop_cols <- setdiff(colnames(test@meta.data),
                c("nCount_RNA", "nFeature_RNA", "orig.ident"))
        }

        proportions <- test@meta.data[, prop_cols, drop = FALSE]
        colnames(proportions) <- gsub("^prob_", "", colnames(proportions))

        # Save
        prefix <- file.path(output_dir, paste0("silver_", ds_id, "_", pattern_id))
        writeMM(t(counts), paste0(prefix, "_counts.mtx"))  # spots x genes
        writeLines(genes, paste0(prefix, "_genes.txt"))
        write.csv(proportions, paste0(prefix, "_proportions.csv"))

        cat("    Pattern", pattern_id, ": spots=", ncol(test), "\n")
    }
}

# ============================================================
# 3. Convert Liver Case Study
# ============================================================
cat("\n=== Converting Liver Case Study ===\n")

liver_dir <- file.path(data_dir, "liver")

# Reference
liver_ref_file <- file.path(liver_dir, "liver_mouseStSt_9celltypes.rds")
if (file.exists(liver_ref_file)) {
    cat("  Loading liver reference...\n")
    ref <- readRDS(liver_ref_file)
    prefix <- file.path(output_dir, "liver_ref_9ct")
    save_seurat(ref, prefix, is_spatial = FALSE)
    cat("    Cells:", ncol(ref), " Genes:", nrow(ref), "\n")
}

# Visium samples
for (sample_id in c("JB01", "JB02", "JB03", "JB04")) {
    sample_file <- file.path(liver_dir, paste0("liver_mouseVisium_", sample_id, ".rds"))
    if (!file.exists(sample_file)) {
        cat("  [SKIP] Sample", sample_id, "not found\n")
        next
    }

    cat("  Converting Visium sample:", sample_id, "...\n")
    sp <- readRDS(sample_file)
    prefix <- file.path(output_dir, paste0("liver_mouseVisium_", sample_id))
    save_seurat(sp, prefix, is_spatial = TRUE)
    cat("    Spots:", ncol(sp), " Genes:", nrow(sp), "\n")
}

# ============================================================
# 4. Convert Melanoma Case Study
# ============================================================
cat("\n=== Converting Melanoma Case Study ===\n")

melanoma_dir <- file.path(data_dir, "melanoma")

# Reference
melanoma_ref_file <- file.path(melanoma_dir, "melanoma_scrna_ref.rds")
if (file.exists(melanoma_ref_file)) {
    cat("  Loading melanoma reference...\n")
    ref <- readRDS(melanoma_ref_file)
    prefix <- file.path(output_dir, "melanoma_ref")
    save_seurat(ref, prefix, is_spatial = FALSE)
    cat("    Cells:", ncol(ref), " Genes:", nrow(ref), "\n")
}

# Visium samples
for (sample_id in c("02", "03", "04")) {
    sample_file <- file.path(melanoma_dir, paste0("melanoma_visium_sample", sample_id, ".rds"))
    if (!file.exists(sample_file)) {
        cat("  [SKIP] Sample", sample_id, "not found\n")
        next
    }

    cat("  Converting Visium sample:", sample_id, "...\n")
    sp <- readRDS(sample_file)
    prefix <- file.path(output_dir, paste0("melanoma_visium_sample", sample_id))
    save_seurat(sp, prefix, is_spatial = TRUE)
    cat("    Spots:", ncol(sp), " Genes:", nrow(sp), "\n")
}

# ============================================================
# 5. Convert Gold Standard Datasets
# ============================================================
cat("\n=== Converting Gold Standard Datasets ===\n")

convert_gold_standard <- function(rds_file, output_dir) {
    cat("  Converting:", basename(rds_file), "\n")

    data <- readRDS(rds_file)
    base_name <- tools::file_path_sans_ext(basename(rds_file))

    # Extract counts
    if (is.list(data) && "counts" %in% names(data)) {
        counts <- data$counts
    } else if (is.list(data) && "sc.counts" %in% names(data)) {
        counts <- data$sc.counts
    } else {
        cat("    Unknown structure, skipping\n")
        return(NULL)
    }

    # Save counts as MTX
    writeMM(as(counts, "dgCMatrix"), file.path(output_dir, paste0(base_name, "_counts.mtx")))

    # Save gene names
    writeLines(rownames(counts), file.path(output_dir, paste0(base_name, "_genes.txt")))

    # Save spot/cell names
    writeLines(colnames(counts), file.path(output_dir, paste0(base_name, "_spots.txt")))

    # Extract proportions if available
    if ("relative_spot_composition" %in% names(data)) {
        write.csv(data$relative_spot_composition,
                  file.path(output_dir, paste0(base_name, "_proportions.csv")))
    }

    # Extract coordinates if available
    if ("coordinates" %in% names(data)) {
        write.csv(data$coordinates,
                  file.path(output_dir, paste0(base_name, "_coords.csv")))
    }

    # Extract cell types if available
    if ("cell.types" %in% names(data)) {
        writeLines(as.character(data$cell.types),
                   file.path(output_dir, paste0(base_name, "_celltypes.txt")))
    }

    cat("    Spots/cells:", ncol(counts), " Genes:", nrow(counts), "\n")
}

# Gold Standard directories (seqFISH+ and STARMap)
gold_dirs <- c("gold_standard_1", "gold_standard_2", "gold_standard_3")

for (gold_dir in gold_dirs) {
    gold_path <- file.path(data_dir, gold_dir)
    if (dir.exists(gold_path)) {
        rds_files <- list.files(gold_path, pattern = "\\.rds$", full.names = TRUE)
        cat("  Found", length(rds_files), "files in", gold_dir, "\n")

        for (f in rds_files) {
            tryCatch({
                convert_gold_standard(f, output_dir)
            }, error = function(e) {
                cat("    Error:", e$message, "\n")
            })
        }
    }
}

cat("\n")
cat(strrep("=", 60), "\n")
cat("Conversion complete!\n")
cat("Output directory:", output_dir, "\n")
cat(strrep("=", 60), "\n")
