#!/usr/bin/env Rscript
# Convert Melanoma Seurat objects to MTX format for FlashDeconv

library(Seurat)
library(Matrix)

data_dir <- "/Users/apple/Research/FlashDeconv/validation/melanoma_data"
output_dir <- "/Users/apple/Research/FlashDeconv/validation/benchmark_data/converted"

cat(strrep("=", 60), "\n")
cat("Converting Melanoma data for FlashDeconv\n")
cat(strrep("=", 60), "\n\n")

# Convert reference scRNA-seq
cat("Loading reference scRNA-seq data...\n")
ref <- readRDS(file.path(data_dir, "melanoma_seurat_obj_filtered.rds"))

cat("Reference object info:\n")
cat("  Cells:", ncol(ref), "\n")
cat("  Genes:", nrow(ref), "\n")
cat("  Cell types:", length(unique(ref$celltype)), "\n")
cat("  Types:", paste(unique(ref$celltype), collapse=", "), "\n\n")

# Extract reference counts and metadata
counts <- GetAssayData(ref, layer = "counts")
if (is.null(counts)) {
    counts <- GetAssayData(ref, slot = "counts")
}
genes <- rownames(ref)
celltypes <- ref$celltype

# Save reference
prefix <- file.path(output_dir, "melanoma_ref")
cat("Saving reference to:", prefix, "\n")
writeMM(counts, paste0(prefix, "_counts.mtx"))
writeLines(genes, paste0(prefix, "_genes.txt"))
writeLines(celltypes, paste0(prefix, "_celltypes.txt"))

cat("\nReference saved:\n")
cat("  Cells:", ncol(counts), "\n")
cat("  Genes:", nrow(counts), "\n\n")

# Convert Visium samples
visium_files <- c(
    "melanoma_visium_sample02.rds",
    "melanoma_visium_sample03.rds",
    "melanoma_visium_sample04.rds"
)

for (f in visium_files) {
    cat("Loading:", f, "\n")
    sp <- readRDS(file.path(data_dir, f))

    sample_name <- gsub(".rds", "", f)

    # Get counts
    sp_counts <- GetAssayData(sp, layer = "counts")
    if (is.null(sp_counts)) {
        sp_counts <- GetAssayData(sp, slot = "counts")
    }
    sp_genes <- rownames(sp)

    # Get coordinates
    coords <- GetTissueCoordinates(sp)
    if (is.null(coords)) {
        # Try alternative method
        coords <- sp@images[[1]]@coordinates
        coords <- data.frame(
            x = coords$imagerow,
            y = coords$imagecol
        )
    }

    # Save spatial data
    prefix <- file.path(output_dir, sample_name)
    writeMM(t(sp_counts), paste0(prefix, "_counts.mtx"))
    writeLines(sp_genes, paste0(prefix, "_genes.txt"))
    writeLines(colnames(sp_counts), paste0(prefix, "_barcodes.txt"))
    coords <- coords[colnames(sp_counts), , drop = FALSE]
    write.csv(coords, paste0(prefix, "_coords.csv"))

    cat("  Spots:", ncol(sp_counts), "\n")
    cat("  Genes:", nrow(sp_counts), "\n\n")
}

cat(strrep("=", 60), "\n")
cat("Conversion complete!\n")
cat(strrep("=", 60), "\n")
