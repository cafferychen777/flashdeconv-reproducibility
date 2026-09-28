# Convert all Silver Standard test datasets to CSV/MTX format
library(Matrix)

base_dir <- "validation/benchmark_data"
out_dir <- "validation/benchmark_data/converted"

# Dataset mapping
datasets <- list(
    "1" = "brain_cortex",
    "2" = "cerebellum_cell",
    "3" = "cerebellum_nucleus",
    "4" = "hippocampus",
    "5" = "kidney",
    "6" = "scc_p5"
)

for (ds_id in names(datasets)) {
    ds_name <- datasets[[ds_id]]

    # Find all pattern directories for this dataset
    pattern <- paste0("silver_standard_", ds_id, "-*")
    dirs <- list.dirs(base_dir, recursive = FALSE)
    ds_dirs <- dirs[grepl(paste0("silver_standard_", ds_id, "-"), dirs)]

    cat("\n=== Dataset", ds_id, "(", ds_name, ") ===\n")
    cat("Found", length(ds_dirs), "patterns\n")

    for (ds_dir in ds_dirs) {
        # Get pattern ID
        dir_name <- basename(ds_dir)
        pattern_id <- sub(paste0("silver_standard_", ds_id, "-"), "", dir_name)

        # Find RDS files (replicates)
        rds_files <- list.files(ds_dir, pattern = "\\.rds$", full.names = TRUE)

        if (length(rds_files) == 0) {
            cat("  No RDS files in", dir_name, "\n")
            next
        }

        # Use first replicate for testing (could aggregate later)
        rds_file <- rds_files[1]
        cat("  Converting", dir_name, "rep1...\n")

        tryCatch({
            ss <- readRDS(rds_file)

            # Get counts (genes x spots)
            counts <- ss$counts
            if (!inherits(counts, "dgCMatrix")) {
                counts <- as(counts, "dgCMatrix")
            }

            # Get proportions
            props <- ss$relative_spot_composition

            # Remove non-numeric columns
            numeric_cols <- sapply(props, is.numeric)
            props_numeric <- props[, numeric_cols, drop = FALSE]

            # Output prefix
            prefix <- file.path(out_dir, paste0("silver_", ds_id, "_", pattern_id))

            # Write counts (genes x spots -> need to transpose for our format)
            writeMM(counts, paste0(prefix, "_counts.mtx"))

            # Write genes
            writeLines(rownames(counts), paste0(prefix, "_genes.txt"))

            # Write spots
            writeLines(as.character(1:ncol(counts)), paste0(prefix, "_spots.txt"))

            # Write proportions
            write.csv(props_numeric, paste0(prefix, "_proportions.csv"), row.names = TRUE)

            cat("    Spots:", ncol(counts), "Genes:", nrow(counts), "Cell types:", ncol(props_numeric), "\n")

        }, error = function(e) {
            cat("    Error:", conditionMessage(e), "\n")
        })
    }
}

cat("\n=== Done ===\n")
