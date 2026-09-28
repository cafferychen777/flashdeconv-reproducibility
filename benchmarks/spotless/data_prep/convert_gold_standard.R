# Convert Spotless Gold Standard RDS files to CSV/MTX format
library(Matrix)

convert_gold_standard <- function(rds_file, output_dir) {
  cat("Converting:", rds_file, "\n")

  # Load the RDS file
  data <- readRDS(rds_file)

  # Get base name
  base_name <- tools::file_path_sans_ext(basename(rds_file))

  # Check structure
  cat("  Class:", class(data), "\n")
  if (is.list(data)) {
    cat("  Names:", names(data), "\n")
  }

  # Extract counts
  if (is.list(data) && "counts" %in% names(data)) {
    counts <- data$counts
  } else if (is.list(data) && "sc.counts" %in% names(data)) {
    counts <- data$sc.counts
  } else {
    cat("  Unknown structure, skipping\n")
    return(NULL)
  }

  cat("  Counts dim:", dim(counts), "\n")

  # Save counts as MTX
  counts_file <- file.path(output_dir, paste0(base_name, "_counts.mtx"))
  writeMM(as(counts, "dgCMatrix"), counts_file)

  # Save gene names
  genes_file <- file.path(output_dir, paste0(base_name, "_genes.txt"))
  write.table(rownames(counts), genes_file, row.names=FALSE, col.names=FALSE, quote=FALSE)

  # Save spot/cell names
  spots_file <- file.path(output_dir, paste0(base_name, "_spots.txt"))
  write.table(colnames(counts), spots_file, row.names=FALSE, col.names=FALSE, quote=FALSE)

  # Extract proportions if available
  if ("relative_spot_composition" %in% names(data)) {
    props <- data$relative_spot_composition
    props_file <- file.path(output_dir, paste0(base_name, "_proportions.csv"))
    write.csv(props, props_file)
    cat("  Proportions dim:", dim(props), "\n")
  }

  # Extract coordinates if available
  if ("coordinates" %in% names(data)) {
    coords <- data$coordinates
    coords_file <- file.path(output_dir, paste0(base_name, "_coords.csv"))
    write.csv(coords, coords_file)
    cat("  Coords dim:", dim(coords), "\n")
  }

  # Extract cell types if available
  if ("cell.types" %in% names(data)) {
    ct <- data$cell.types
    ct_file <- file.path(output_dir, paste0(base_name, "_celltypes.txt"))
    write.table(ct, ct_file, row.names=FALSE, col.names=FALSE, quote=FALSE)
    cat("  Cell types:", length(unique(ct)), "\n")
  }

  cat("  Saved to:", output_dir, "\n")
}

# Main
base_dir <- "/Users/apple/Research/FlashDeconv/validation/benchmark_data"
output_dir <- file.path(base_dir, "converted")
dir.create(output_dir, showWarnings = FALSE)

# Convert gold standard files
gold_dirs <- c("gold_standard_1", "gold_standard_2", "gold_standard_3")

for (gold_dir in gold_dirs) {
  gold_path <- file.path(base_dir, gold_dir)
  if (dir.exists(gold_path)) {
    rds_files <- list.files(gold_path, pattern = "\\.rds$", full.names = TRUE)
    cat("\n=== Processing", gold_dir, "===\n")
    cat("Found", length(rds_files), "RDS files\n")

    for (f in rds_files) {
      tryCatch({
        convert_gold_standard(f, output_dir)
      }, error = function(e) {
        cat("Error converting", f, ":", e$message, "\n")
      })
    }
  }
}

cat("\nDone!\n")
