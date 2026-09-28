#!/usr/bin/env Rscript
# Official spacexr RCTD on C2 pseudo-Visium HD bins.
# Usage: Rscript run_rctd.R <res_um> <doublet|full> <umi_min> <cores> <out_dir>
# Writes <out_dir>/preds/<res>um/RCTD__<mode>__umi<umi_min>.csv.gz (+ .meta.txt).
#
# Reference: Xenium self-reference (all 38 annotated types, 422 genes), each type
# capped at 10,000 cells (= spacexr default n_max_cells), prepared in prepare_bins.py.
# Reference min_UMI lowered to 10 because Xenium cells have low counts (median ~57
# UMI); spacexr's default 100 would discard most reference cells.
#
# Spatial UMI filter: UMI_min = <umi_min> (100 = spacexr default; 20 = lowered),
# counts_MIN = 10 (default). UMI_min_sigma = 300 (default) unless fewer than 100
# bins reach 300 UMI; then it is set to the UMI of the 1000th-highest bin (but not
# below UMI_min) so sigma estimation has pixels to fit. Recorded in meta.
#
# Doublet-mode -> proportions: singlet -> first_type = 1; doublet_certain and
# doublet_uncertain -> weights_doublet (first_type, second_type, renormalized);
# reject -> no prediction (treated as uncovered).
# Full mode -> normalize_weights(results$weights).

suppressPackageStartupMessages({library(spacexr); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
res <- as.integer(args[1]); mode <- args[2]; umi_min <- as.integer(args[3])
cores <- as.integer(args[4]); out_dir <- args[5]
bdir <- file.path(out_dir, "bins", "rctd")
pdir <- file.path(out_dir, "preds", paste0(res, "um"))
n_test <- if (length(args) >= 6) as.integer(args[6]) else 0  # >0: quick test on a bin subset
if (n_test > 0) pdir <- file.path(pdir, "_test")
dir.create(pdir, recursive = TRUE, showWarnings = FALSE)
out_csv <- file.path(pdir, sprintf("RCTD__%s__umi%d.csv.gz", mode, umi_min))
if (file.exists(out_csv)) { cat("exists, skip:", out_csv, "\n"); quit(save = "no") }

genes <- readLines(file.path(bdir, "genes.txt"))
ref_counts <- as(readMM(file.path(bdir, "ref_counts.mtx")), "CsparseMatrix")
ref_meta <- read.csv(file.path(bdir, "ref_meta.csv"), stringsAsFactors = FALSE)
rownames(ref_counts) <- genes; colnames(ref_counts) <- ref_meta$barcode
ct <- factor(ref_meta$cell_type); names(ct) <- ref_meta$barcode
nUMI_ref <- ref_meta$nUMI; names(nUMI_ref) <- ref_meta$barcode
reference <- Reference(ref_counts, ct, nUMI_ref, min_UMI = 10)
cat("Reference built:", ncol(reference@counts), "cells,", length(levels(reference@cell_types)), "types\n")

counts <- as(readMM(file.path(bdir, sprintf("bins_%dum.mtx", res))), "CsparseMatrix")
coords <- read.csv(file.path(bdir, sprintf("bins_%dum_coords.csv", res)), row.names = 1)
rownames(counts) <- genes; colnames(counts) <- rownames(coords)
n_bins <- ncol(counts)
nUMI <- colSums(counts)
keep <- nUMI > 0
if (n_test > 0) keep <- keep & (cumsum(keep) <= n_test)
counts <- counts[, keep]; coords <- coords[keep, ]; nUMI <- nUMI[keep]
n_pass <- sum(nUMI >= umi_min)
cat(sprintf("%dum: %d bins, %d nonzero, %d (%.1f%%) with UMI >= %d\n",
            res, n_bins, sum(keep), n_pass, 100 * n_pass / n_bins, umi_min))

sigma_thr <- 300
if (sum(nUMI >= 300) < 100) {
  sigma_thr <- max(umi_min, sort(nUMI, decreasing = TRUE)[min(1000, length(nUMI))])
}
meta_file <- sub("\\.csv\\.gz$", ".meta.txt", out_csv)
write_meta <- function(fit_s, n_cov, note) {
  writeLines(c(sprintf("fit_seconds=%.2f", fit_s), sprintf("n_bins=%d", n_bins),
               sprintf("n_covered=%d", n_cov), sprintf("umi_min=%d", umi_min),
               sprintf("umi_min_sigma=%s", sigma_thr), sprintf("mode=%s", mode),
               sprintf("version=spacexr %s", as.character(packageVersion("spacexr"))),
               sprintf("notes=%s", note)), meta_file)
}
if (n_pass < 10) {
  cat("Too few bins pass UMI_min; recording zero coverage\n")
  write_meta(0, 0, sprintf("only %d bins with UMI>=%d; RCTD not run", n_pass, umi_min))
  write.csv(data.frame(barcode = character(0)), gzfile(out_csv), row.names = FALSE)
  quit(save = "no")
}
cat("UMI_min_sigma =", sigma_thr, "\n")

rds_file <- sub("\\.csv\\.gz$", ".results.rds", out_csv)
if (file.exists(rds_file)) {
  # Resume: fit already done, only redo conversion.
  cached <- readRDS(rds_file); r <- cached$results; fit_s <- cached$fit_s
  cat("Loaded cached RCTD results:", rds_file, "\n")
} else {
  puck <- SpatialRNA(coords, counts, nUMI)
  t0 <- Sys.time()
  myRCTD <- create.RCTD(puck, reference, max_cores = cores, UMI_min = umi_min,
                        counts_MIN = 10, UMI_min_sigma = sigma_thr)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = mode)
  fit_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  r <- myRCTD@results
  saveRDS(list(results = r, fit_s = fit_s), rds_file)
}
cat(sprintf("RCTD %s fit: %.1f s\n", mode, fit_s))

types <- levels(reference@cell_types)
if (mode == "full") {
  W <- as.matrix(normalize_weights(r$weights))
  spot_class <- rep("full", nrow(W))
} else {
  df <- r$results_df
  wd <- r$weights_doublet
  W <- matrix(0, nrow(df), length(types), dimnames = list(rownames(df), types))
  sc <- as.character(df$spot_class)
  ft <- as.character(df$first_type); st <- as.character(df$second_type)
  for (i in seq_len(nrow(df))) {
    if (sc[i] == "singlet") {
      W[i, ft[i]] <- 1
    } else if (sc[i] %in% c("doublet_certain", "doublet_uncertain")) {
      w <- if (!is.null(rownames(wd))) wd[rownames(df)[i], ] else wd[i, ]
      w <- w / sum(w)
      W[i, ft[i]] <- W[i, ft[i]] + w[1]
      W[i, st[i]] <- W[i, st[i]] + w[2]
    }
  }
  spot_class <- sc
  cat("spot_class counts:\n"); print(table(sc))
  keep_rows <- sc != "reject"
  W <- W[keep_rows, , drop = FALSE]; spot_class <- sc[keep_rows]
}
out <- data.frame(barcode = rownames(W), spot_class = spot_class, W, check.names = FALSE)
tmp_csv <- paste0(out_csv, ".tmp.gz")
write.csv(out, gzfile(tmp_csv), row.names = FALSE)
write_meta(fit_s, nrow(W), sprintf("covered=%d/%d (%.1f%%)", nrow(W), n_bins, 100 * nrow(W) / n_bins))
file.rename(tmp_csv, out_csv)
cat("RCTD_DONE", out_csv, "\n")
