#!/usr/bin/env Rscript
#
# Convert Spotless Benchmark RDS files to MTX format for FlashDeconv
#
# This script converts Spotless benchmark RDS files into a format that
# can be read by the Python benchmark scripts.
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
input_dir <- data_dir

if (!dir.exists(file.path(input_dir, "reference")) &&
        dir.exists(file.path(input_dir, "standards", "reference"))) {
    input_dir <- file.path(input_dir, "standards")
}

output_dir <- file.path(data_dir, "converted")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat(strrep("=", 60), "\n")
cat("Converting Spotless Benchmark Data\n")
cat("Input directory:", input_dir, "\n")
cat("Output directory:", output_dir, "\n")
cat(strrep("=", 60), "\n\n")

# Track conversion results for final summary
converted <- character(0)
skipped <- character(0)

# ============================================================
# Helper function to save Seurat object
# ============================================================
get_counts <- function(obj) {
    tryCatch({
        GetAssayData(obj, layer = "counts")
    }, error = function(e) {
        GetAssayData(obj, slot = "counts")
    })
}

as_sparse_matrix <- function(counts) {
    if (inherits(counts, "Matrix")) {
        return(counts)
    }
    Matrix(counts, sparse = TRUE)
}

save_seurat <- function(obj, prefix, is_spatial = FALSE) {
    # Get counts
    counts <- get_counts(obj)

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

extract_rep_id <- function(path) {
    match <- regexec("_rep([0-9]+)\\.rds$", basename(path))
    parts <- regmatches(basename(path), match)[[1]]
    if (length(parts) >= 2) {
        return(as.integer(parts[2]))
    }
    NA_integer_
}

sort_rds_files <- function(files) {
    reps <- vapply(files, extract_rep_id, integer(1))
    files[order(ifelse(is.na(reps), .Machine$integer.max, reps), basename(files))]
}

select_representative_rds <- function(files) {
    files <- sort_rds_files(files)
    reps <- vapply(files, extract_rep_id, integer(1))
    rep1_idx <- which(reps == 1)
    if (length(rep1_idx) > 0) {
        return(files[rep1_idx[1]])
    }
    files[1]
}

find_silver_test_files <- function(ds_id, dataset_name, pattern_id) {
    legacy_file <- file.path(
        input_dir,
        "test_silver_standard",
        paste0("silver_standard_", ds_id, "_", dataset_name, "_", pattern_id, ".rds")
    )

    if (file.exists(legacy_file)) {
        return(legacy_file)
    }

    pattern_dir <- paste0("silver_standard_", ds_id, "-", pattern_id)
    search_dirs <- c(
        file.path(input_dir, pattern_dir),
        file.path(input_dir, "test_silver_standard", pattern_dir)
    )

    files <- character(0)
    for (dir in search_dirs) {
        if (dir.exists(dir)) {
            files <- c(files, list.files(dir, pattern = "\\.rds$", full.names = TRUE))
        }
    }

    sort_rds_files(unique(files))
}

convert_silver_test <- function(rds_file, prefix) {
    test <- readRDS(rds_file)

    if (is.list(test) &&
            all(c("counts", "relative_spot_composition") %in% names(test))) {
        counts <- as_sparse_matrix(test$counts)
        genes <- rownames(counts)
        spots <- colnames(counts)
        proportions <- as.data.frame(test$relative_spot_composition)
        prop_cols <- setdiff(colnames(proportions), c("name", "region", "spot_no"))
        proportions <- proportions[, prop_cols, drop = FALSE]

        # Synthspot counts are genes x spots. Keep that orientation because
        # the Python benchmark loaders transpose MTX files to spots x genes.
        writeMM(counts, paste0(prefix, "_counts.mtx"))
        writeLines(genes, paste0(prefix, "_genes.txt"))
        if (!is.null(spots)) {
            writeLines(spots, paste0(prefix, "_spots.txt"))
        }
        write.csv(proportions, paste0(prefix, "_proportions.csv"))

        return(list(spots = ncol(counts), genes = nrow(counts)))
    }

    counts <- get_counts(test)
    genes <- rownames(test)

    prop_cols <- grep("^prob_", colnames(test@meta.data), value = TRUE)
    if (length(prop_cols) == 0) {
        prop_cols <- setdiff(colnames(test@meta.data),
            c("nCount_RNA", "nFeature_RNA", "orig.ident"))
    }

    proportions <- test@meta.data[, prop_cols, drop = FALSE]
    colnames(proportions) <- gsub("^prob_", "", colnames(proportions))

    # Seurat counts are also genes x spots.
    writeMM(counts, paste0(prefix, "_counts.mtx"))
    writeLines(genes, paste0(prefix, "_genes.txt"))
    writeLines(colnames(test), paste0(prefix, "_spots.txt"))
    write.csv(proportions, paste0(prefix, "_proportions.csv"))

    list(spots = ncol(test), genes = nrow(test))
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

ref_dir <- file.path(input_dir, "reference")

for (id in names(silver_refs)) {
    name <- silver_refs[[id]]
    rds_file <- file.path(ref_dir, paste0(name, ".rds"))

    if (!file.exists(rds_file)) {
        cat("  [SKIP] Reference", id, "not found:", rds_file, "\n")
        skipped <<- c(skipped, paste0("reference_", id))
        next
    }

    cat("  Converting reference", id, ":", name, "...\n")
    ref <- readRDS(rds_file)
    prefix <- file.path(output_dir, paste0("reference_", id))
    save_seurat(ref, prefix, is_spatial = FALSE)
    cat("    Cells:", ncol(ref), " Genes:", nrow(ref), "\n")
    converted <<- c(converted, paste0("reference_", id))
}

# ============================================================
# 2. Convert Silver Standard Test Datasets
# ============================================================
cat("\n=== Converting Silver Standard Test Datasets ===\n")

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
        rds_files <- find_silver_test_files(ds_id, info$name, pattern_id)

        if (length(rds_files) == 0) {
            skipped <<- c(skipped, paste0("silver_", ds_id, "_", pattern_id))
            next
        }

        rds_file <- select_representative_rds(rds_files)
        prefix <- file.path(output_dir, paste0("silver_", ds_id, "_", pattern_id))
        stats <- convert_silver_test(rds_file, prefix)

        cat("    Pattern", pattern_id, ": using", basename(rds_file),
            "spots=", stats$spots, "genes=", stats$genes, "\n")
        converted <<- c(converted, paste0("silver_", ds_id, "_", pattern_id))
    }
}

# ============================================================
# 3. Convert Liver Case Study
# ============================================================
cat("\n=== Converting Liver Case Study ===\n")

liver_dir <- file.path(input_dir, "liver")

# Reference
liver_ref_file <- file.path(liver_dir, "liver_mouseStSt_9celltypes.rds")
if (file.exists(liver_ref_file)) {
    cat("  Loading liver reference...\n")
    ref <- readRDS(liver_ref_file)
    prefix <- file.path(output_dir, "liver_ref_9ct")
    save_seurat(ref, prefix, is_spatial = FALSE)
    cat("    Cells:", ncol(ref), " Genes:", nrow(ref), "\n")
    converted <- c(converted, "liver_ref")
} else {
    cat("  [SKIP] Liver reference not found:", liver_ref_file, "\n")
    skipped <- c(skipped, "liver_ref")
}

# Visium samples
for (sample_id in c("JB01", "JB02", "JB03", "JB04")) {
    sample_file <- file.path(liver_dir, paste0("liver_mouseVisium_", sample_id, ".rds"))
    if (!file.exists(sample_file)) {
        cat("  [SKIP] Sample", sample_id, "not found\n")
        skipped <- c(skipped, paste0("liver_", sample_id))
        next
    }

    cat("  Converting Visium sample:", sample_id, "...\n")
    sp <- readRDS(sample_file)
    prefix <- file.path(output_dir, paste0("liver_mouseVisium_", sample_id))
    save_seurat(sp, prefix, is_spatial = TRUE)
    cat("    Spots:", ncol(sp), " Genes:", nrow(sp), "\n")
    converted <- c(converted, paste0("liver_", sample_id))
}

# ============================================================
# 4. Convert Melanoma Case Study
# ============================================================
cat("\n=== Converting Melanoma Case Study ===\n")

melanoma_dir <- file.path(input_dir, "melanoma")

# Reference
melanoma_ref_file <- file.path(melanoma_dir, "melanoma_scrna_ref.rds")
if (file.exists(melanoma_ref_file)) {
    cat("  Loading melanoma reference...\n")
    ref <- readRDS(melanoma_ref_file)
    prefix <- file.path(output_dir, "melanoma_ref")
    save_seurat(ref, prefix, is_spatial = FALSE)
    cat("    Cells:", ncol(ref), " Genes:", nrow(ref), "\n")
    converted <- c(converted, "melanoma_ref")
} else {
    cat("  [SKIP] Melanoma reference not found:", melanoma_ref_file, "\n")
    skipped <- c(skipped, "melanoma_ref")
}

# Visium samples
for (sample_id in c("02", "03", "04")) {
    sample_file <- file.path(melanoma_dir, paste0("melanoma_visium_sample", sample_id, ".rds"))
    if (!file.exists(sample_file)) {
        cat("  [SKIP] Sample", sample_id, "not found\n")
        skipped <- c(skipped, paste0("melanoma_", sample_id))
        next
    }

    cat("  Converting Visium sample:", sample_id, "...\n")
    sp <- readRDS(sample_file)
    prefix <- file.path(output_dir, paste0("melanoma_visium_sample", sample_id))
    save_seurat(sp, prefix, is_spatial = TRUE)
    cat("    Spots:", ncol(sp), " Genes:", nrow(sp), "\n")
    converted <- c(converted, paste0("melanoma_", sample_id))
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
    gold_path <- file.path(input_dir, gold_dir)
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
cat("Conversion Summary\n")
cat(strrep("=", 60), "\n")
cat("  Converted:", length(converted), "datasets\n")
cat("  Skipped:  ", length(skipped), "datasets\n")
cat("  Output:   ", output_dir, "\n")

if (length(skipped) > 0) {
    cat("\nSkipped datasets:\n")
    for (s in skipped) {
        cat("  -", s, "\n")
    }
    cat("\nIf many datasets are skipped, the tarball may have extracted")
    cat("\nwith a top-level directory. Check:\n")
    cat("  ls", input_dir, "\n")
    cat("The converter also supports archives extracted under a top-level\n")
    cat("'standards/' directory, so moving files is usually not required.\n")
}

# Expected counts for validation
expected_silver_refs <- 6
expected_silver_tests <- 56  # 11 + 9*5
expected_liver <- 5          # 1 ref + 4 samples
expected_melanoma <- 4       # 1 ref + 3 samples

n_silver_refs <- sum(grepl("^reference_", converted))
n_silver_tests <- sum(grepl("^silver_", converted))
n_liver <- sum(grepl("^liver_", converted))
n_melanoma <- sum(grepl("^melanoma_", converted))

cat("\nExpected vs actual:\n")
cat(sprintf("  Silver references : %d / %d\n", n_silver_refs, expected_silver_refs))
cat(sprintf("  Silver test sets  : %d / %d\n", n_silver_tests, expected_silver_tests))
cat(sprintf("  Liver datasets    : %d / %d\n", n_liver, expected_liver))
cat(sprintf("  Melanoma datasets : %d / %d\n", n_melanoma, expected_melanoma))

if (length(skipped) == 0) {
    cat("\nAll datasets converted successfully.\n")
} else {
    cat("\nWARNING: Some datasets were not found. Benchmarks may be incomplete.\n")
    quit(status = 1)
}
cat(strrep("=", 60), "\n")
