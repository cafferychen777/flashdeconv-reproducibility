# Convert Gold Standard reference datasets to CSV/MTX format
library(Matrix)

out_dir <- "validation/benchmark_data/converted"
ref_dir <- "validation/benchmark_data/reference"

# Gold Standard 1: seqFISH+ cortex reference
cat("=== Converting Gold Standard 1 (seqFISH+ cortex) ===\n")
gs1 <- readRDS(file.path(ref_dir, "gold_standard_1.rds"))
counts1 <- gs1@assays$RNA@counts
meta1 <- gs1@meta.data

cat("Cells:", ncol(counts1), "\n")
cat("Genes:", nrow(counts1), "\n")
cat("Cell types:", length(unique(meta1$celltype)), "\n")

# Write gold_ref_1
prefix1 <- file.path(out_dir, "gold_ref_1")
writeMM(counts1, paste0(prefix1, "_counts.mtx"))
writeLines(rownames(counts1), paste0(prefix1, "_genes.txt"))
writeLines(colnames(counts1), paste0(prefix1, "_cells.txt"))
writeLines(as.character(meta1$celltype), paste0(prefix1, "_celltypes.txt"))
cat("Saved to:", prefix1, "\n")

# Gold Standard 2: seqFISH+ olfactory bulb reference
cat("\n=== Converting Gold Standard 2 (seqFISH+ ob) ===\n")
gs2 <- readRDS(file.path(ref_dir, "gold_standard_2.rds"))
counts2 <- gs2@assays$RNA@counts
meta2 <- gs2@meta.data

cat("Cells:", ncol(counts2), "\n")
cat("Genes:", nrow(counts2), "\n")
cat("Cell types:", length(unique(meta2$celltype)), "\n")

# Write gold_ref_2
prefix2 <- file.path(out_dir, "gold_ref_2")
writeMM(counts2, paste0(prefix2, "_counts.mtx"))
writeLines(rownames(counts2), paste0(prefix2, "_genes.txt"))
writeLines(colnames(counts2), paste0(prefix2, "_cells.txt"))
writeLines(as.character(meta2$celltype), paste0(prefix2, "_celltypes.txt"))
cat("Saved to:", prefix2, "\n")

# Gold Standard 3: STARMap reference (Allen Brain Atlas)
cat("\n=== Converting Gold Standard 3 (STARMap) ===\n")
gs3 <- readRDS(file.path(ref_dir, "gold_standard_3_12celltypes.rds"))
counts3 <- gs3@assays$RNA@counts
meta3 <- gs3@meta.data

cat("Cells:", ncol(counts3), "\n")
cat("Genes:", nrow(counts3), "\n")
cat("Cell types:", length(unique(meta3$celltype)), "\n")

# Write gold_ref_3
prefix3 <- file.path(out_dir, "gold_ref_3")
writeMM(counts3, paste0(prefix3, "_counts.mtx"))
writeLines(rownames(counts3), paste0(prefix3, "_genes.txt"))
writeLines(colnames(counts3), paste0(prefix3, "_cells.txt"))
writeLines(as.character(meta3$celltype), paste0(prefix3, "_celltypes.txt"))
cat("Saved to:", prefix3, "\n")

cat("\n=== Done ===\n")
