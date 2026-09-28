#!/usr/bin/env Rscript
# C1 runner: RCTD via spacexr (10x-modified spacexr, dmcable/spacexr PR #206, as used for
# the Visium HD CRC study). Follows 10x Methods/Deconvolution.R: Reference(counts,
# cell_types, colSums(counts)) with defaults (n_max_cells = 10000, min_UMI = 100),
# SpatialRNA(coords, counts, colSums(counts)), create.RCTD(puck, reference,
# max_cores = <allocated cores>) with default UMI_min = 100, run.RCTD(doublet_mode = mode).
# Usage: run_rctd.R <data_dir> <scale> <mode: doublet|full> <props_dir> <n_cores>
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]; scale <- as.integer(args[2]); mode <- args[3]
props_dir <- args[4]; n_cores <- as.integer(args[5])
src_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(src_dir, "c1_common.R"))
suppressPackageStartupMessages(library(spacexr))
desc <- packageDescription("spacexr")
ver <- sprintf("spacexr %s (%s@%s); R %s", desc$Version,
               ifelse(is.null(desc$RemoteRepo), "unknown", paste0(desc$RemoteUsername, "/", desc$RemoteRepo)),
               substr(ifelse(is.null(desc$RemoteSha), "", desc$RemoteSha), 1, 8),
               paste(R.version$major, R.version$minor, sep = "."))

load_s <- NA; fit_s <- NA; n_bins <- NA; n_genes <- NA; n_types <- NA
res <- tryCatch({
  t0 <- proc.time()
  st <- read_h5ad_counts(file.path(data_dir, sprintf("st_%d.h5ad", scale)))
  xy <- read_h5ad_spatial(file.path(data_dir, sprintf("st_%d.h5ad", scale)), length(st$obs))
  ref <- read_h5ad_counts(file.path(data_dir, "ref.h5ad"))
  meta <- read.csv(file.path(data_dir, "ref_meta.csv"), stringsAsFactors = FALSE)
  load_s <- elapsed(t0)
  n_bins <- ncol(st$counts); n_genes <- nrow(st$counts)
  stopifnot(identical(meta$barcode, colnames(ref$counts)))

  t1 <- proc.time()
  ct <- as.factor(gsub("/", "_", meta$cell_type)); names(ct) <- meta$barcode
  n_types <- nlevels(ct)
  reference <- Reference(ref$counts, ct, colSums(ref$counts))
  coords <- data.frame(x = xy[, 1], y = xy[, 2], row.names = colnames(st$counts))
  puck <- SpatialRNA(coords, st$counts, colSums(st$counts))
  rm(ref, st); invisible(gc())
  myRCTD <- create.RCTD(puck, reference, max_cores = n_cores)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = mode)
  fit_s <- elapsed(t1)

  r <- myRCTD@results
  w <- as.matrix(r$weights)
  n_fit <- nrow(w)
  extra <- ""
  if (mode == "doublet") {
    tab <- table(r$results_df$spot_class)
    extra <- paste(names(tab), as.integer(tab), sep = "=", collapse = ",")
  }
  if (scale <= 10000) {
    wn <- sweep(w, 1, pmax(rowSums(w), 1e-12), "/")
    out <- file.path(props_dir, sprintf("rctd_%s_%d.csv.gz", mode, scale))
    df <- data.frame(bin = rownames(wn), wn, check.names = FALSE)
    if (mode == "doublet") {
      rd <- r$results_df[rownames(wn), ]
      df$spot_class <- as.character(rd$spot_class)
      df$first_type <- as.character(rd$first_type)
      # the 10x fork's gather_results names this column "scond_type" (typo)
      df$second_type <- as.character(if (!is.null(rd$second_type)) rd$second_type else rd$scond_type)
    }
    gz <- gzfile(out, "w"); write.csv(df, gz, row.names = FALSE); close(gz)
  }
  write_result(status = "OK", fit_seconds = fit_s, load_seconds = load_s, n_bins = n_bins,
               n_bins_fit = n_fit, n_genes = n_genes, n_types = n_types, version = ver,
               notes = sprintf("doublet_mode=%s; max_cores=%d; UMI_min=100 (default) so bins <100 UMI are excluded; spot_class: %s",
                               mode, n_cores, extra))
  0
}, error = function(e) {
  msg <- conditionMessage(e)
  message("ERROR: ", msg)
  write_result(status = status_from_error(msg), fit_seconds = ifelse(is.na(fit_s), "", fit_s),
               load_seconds = ifelse(is.na(load_s), "", load_s), n_bins = ifelse(is.na(n_bins), "", n_bins),
               n_genes = ifelse(is.na(n_genes), "", n_genes), n_types = ifelse(is.na(n_types), "", n_types),
               version = ver, notes = substr(msg, 1, 300))
  1
})
quit(status = res, save = "no")
