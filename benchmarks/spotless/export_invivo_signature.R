# Export mean raw counts per annotated cell type of the Spotless liver inVivo
# reference (the per-cell MTX used by benchmark_liver_stability.py is no longer on disk).
suppressMessages(library(Matrix))
obj <- readRDS("/Users/apple/Research/FlashDeconv/validation/liver_data/liver_mouseStSt_inVivo_9celltypes_annot_cd45.rds")
cls <- class(obj)
cat("class:", cls, "\n")
if ("Seurat" %in% cls) {
  suppressMessages(library(SeuratObject))
  m <- tryCatch(GetAssayData(obj, layer = "counts"), error = function(e) GetAssayData(obj, slot = "counts"))
  meta <- obj@meta.data
} else if ("SingleCellExperiment" %in% cls) {
  m <- SummarizedExperiment::assay(obj, "counts"); meta <- as.data.frame(SummarizedExperiment::colData(obj))
}
print(colnames(meta))
ct_col <- "annot_cd45"; cat("ncells", ncol(m), "\n")
ct <- as.character(meta[[ct_col]])
print(table(ct))
types <- sort(unique(ct))
sig <- sapply(types, function(t) Matrix::rowMeans(m[, ct == t, drop = FALSE]))
write.csv(sig, gzfile("liver_inVivo_signature.csv.gz"))
