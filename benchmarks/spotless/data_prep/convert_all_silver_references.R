# Convert all Silver Standard reference datasets to CSV/MTX format
library(Seurat)
library(Matrix)

ref_dir <- "validation/benchmark_data/reference"
out_dir <- "validation/benchmark_data/converted"

# Silver standard references
refs <- list(
    "1" = "silver_standard_1_brain_cortex",
    "2" = "silver_standard_2_cerebellum_cell",
    "3" = "silver_standard_3_cerebellum_nucleus",
    "4" = "silver_standard_4_hippocampus",
    "5" = "silver_standard_5_kidney",
    "6" = "silver_standard_6_scc_p5"
)

for (id in names(refs)) {
    name <- refs[[id]]
    rds_file <- file.path(ref_dir, paste0(name, ".rds"))

    if (!file.exists(rds_file)) {
        cat("Skipping", name, "- file not found\n")
        next
    }

    cat("\n=== Converting", name, "===\n")

    # Load reference
    ref <- readRDS(rds_file)

    cat("Type:", class(ref), "\n")

    if (inherits(ref, "Seurat")) {
        # Get counts
        counts <- GetAssayData(ref, slot = "counts")
        if (inherits(counts, "dgCMatrix")) {
            counts_t <- t(counts)  # cells x genes -> genes x cells for writeMM
        } else {
            counts_t <- as(t(counts), "dgCMatrix")
        }

        genes <- rownames(ref)
        cells <- colnames(ref)

        # Get cell types
        if ("celltype" %in% colnames(ref@meta.data)) {
            celltypes <- ref@meta.data$celltype
        } else if ("cell_type" %in% colnames(ref@meta.data)) {
            celltypes <- ref@meta.data$cell_type
        } else {
            # Try to find cell type column
            ct_cols <- grep("cell.*type|cluster|label", colnames(ref@meta.data), ignore.case = TRUE, value = TRUE)
            if (length(ct_cols) > 0) {
                celltypes <- ref@meta.data[[ct_cols[1]]]
                cat("Using column:", ct_cols[1], "\n")
            } else {
                cat("Warning: No cell type column found. Available:", paste(colnames(ref@meta.data), collapse=", "), "\n")
                celltypes <- rep("Unknown", ncol(ref))
            }
        }

        cat("Cells:", length(cells), "\n")
        cat("Genes:", length(genes), "\n")
        cat("Cell types:", length(unique(celltypes)), "\n")

        # Write output
        prefix <- file.path(out_dir, paste0("reference_", id))

        writeMM(t(counts_t), paste0(prefix, "_counts.mtx"))  # genes x cells
        writeLines(genes, paste0(prefix, "_genes.txt"))
        writeLines(cells, paste0(prefix, "_cells.txt"))
        writeLines(as.character(celltypes), paste0(prefix, "_celltypes.txt"))

        cat("Saved to:", prefix, "\n")
    } else {
        cat("Unknown type, skipping\n")
    }
}

cat("\n=== Done ===\n")
