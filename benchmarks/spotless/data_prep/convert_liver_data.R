#!/usr/bin/env Rscript
# Convert Liver Seurat objects to MTX format for FlashDeconv

library(Seurat)
library(Matrix)

data_dir <- "/Users/apple/Research/FlashDeconv/validation/liver_data"
output_dir <- "/Users/apple/Research/FlashDeconv/validation/benchmark_data/converted"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat(strrep("=", 60), "\n")
cat("Converting Liver data for FlashDeconv\n")
cat(strrep("=", 60), "\n\n")

# Convert reference scRNA-seq (9 cell types, combined protocols)
cat("Loading reference scRNA-seq data (9 cell types)...\n")
ref <- readRDS(file.path(data_dir, "liver_mouseStSt_9celltypes.rds"))

cat("Reference object info:\n")
cat("  Cells:", ncol(ref), "\n")
cat("  Genes:", nrow(ref), "\n")
cat("  Cell types:", length(unique(ref$annot_cd45)), "\n")
cat("  Types:", paste(unique(ref$annot_cd45), collapse=", "), "\n\n")

# Extract reference counts and metadata
counts <- GetAssayData(ref, layer = "counts")
if (is.null(counts) || length(counts) == 0) {
    counts <- GetAssayData(ref, slot = "counts")
}
genes <- rownames(ref)
celltypes <- ref$annot_cd45

# Save reference
prefix <- file.path(output_dir, "liver_ref_9ct")
cat("Saving reference to:", prefix, "\n")
writeMM(counts, paste0(prefix, "_counts.mtx"))
writeLines(genes, paste0(prefix, "_genes.txt"))
writeLines(as.character(celltypes), paste0(prefix, "_celltypes.txt"))

cat("\nReference saved:\n")
cat("  Cells:", ncol(counts), "\n")
cat("  Genes:", nrow(counts), "\n\n")

# Convert individual protocol references
protocols <- c(
    "exVivo" = "liver_mouseStSt_exVivo_9celltypes_annot_cd45.rds",
    "inVivo" = "liver_mouseStSt_inVivo_9celltypes_annot_cd45.rds",
    "nuclei" = "liver_mouseStSt_nuclei_9celltypes_annot_cd45.rds"
)

for (proto_name in names(protocols)) {
    cat("Loading reference:", proto_name, "...\n")
    ref_proto <- readRDS(file.path(data_dir, protocols[proto_name]))

    counts <- GetAssayData(ref_proto, layer = "counts")
    if (is.null(counts) || length(counts) == 0) {
        counts <- GetAssayData(ref_proto, slot = "counts")
    }
    genes <- rownames(ref_proto)
    celltypes <- ref_proto$annot_cd45

    prefix <- file.path(output_dir, paste0("liver_ref_", proto_name))
    writeMM(counts, paste0(prefix, "_counts.mtx"))
    writeLines(genes, paste0(prefix, "_genes.txt"))
    writeLines(as.character(celltypes), paste0(prefix, "_celltypes.txt"))

    cat("  Cells:", ncol(counts), "  Genes:", nrow(counts), "\n")
}

cat("\n")

# Convert Visium samples
visium_files <- c(
    "liver_mouseVisium_JB01.rds",
    "liver_mouseVisium_JB02.rds",
    "liver_mouseVisium_JB03.rds",
    "liver_mouseVisium_JB04.rds"
)

for (f in visium_files) {
    cat("Loading:", f, "\n")
    sp <- readRDS(file.path(data_dir, f))

    sample_name <- gsub(".rds", "", f)

    # Get counts
    sp_counts <- GetAssayData(sp, layer = "counts")
    if (is.null(sp_counts) || length(sp_counts) == 0) {
        sp_counts <- GetAssayData(sp, slot = "counts")
    }
    sp_genes <- rownames(sp)

    # Get coordinates
    coords <- GetTissueCoordinates(sp)
    if (is.null(coords)) {
        # Try alternative method for older Seurat
        if (length(sp@images) > 0) {
            img_name <- names(sp@images)[1]
            coords_data <- sp@images[[img_name]]@coordinates
            coords <- data.frame(
                x = coords_data$imagerow,
                y = coords_data$imagecol
            )
        }
    }

    # Get zonation annotations if available
    zonation <- NULL
    if ("zonation" %in% colnames(sp@meta.data)) {
        zonation <- sp$zonation
    } else if ("zone" %in% colnames(sp@meta.data)) {
        zonation <- sp$zone
    }

    # Save spatial data
    prefix <- file.path(output_dir, sample_name)
    writeMM(t(sp_counts), paste0(prefix, "_counts.mtx"))
    writeLines(sp_genes, paste0(prefix, "_genes.txt"))
    writeLines(colnames(sp_counts), paste0(prefix, "_barcodes.txt"))
    coords <- coords[colnames(sp_counts), , drop = FALSE]
    write.csv(coords, paste0(prefix, "_coords.csv"))

    # Save zonation if available
    if (!is.null(zonation)) {
        writeLines(as.character(zonation), paste0(prefix, "_zonation.txt"))
        cat("  Zonation annotations saved\n")
    }

    # Save all metadata for AUPR calculation
    meta <- sp@meta.data[colnames(sp_counts), , drop = FALSE]
    write.csv(meta, paste0(prefix, "_metadata.csv"))

    cat("  Spots:", ncol(sp_counts), "\n")
    cat("  Genes:", nrow(sp_counts), "\n\n")
}

cat(strrep("=", 60), "\n")
cat("Conversion complete!\n")
cat(strrep("=", 60), "\n")
