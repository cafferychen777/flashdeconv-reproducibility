#!/usr/bin/env Rscript
# S1 runner: official RCTD (spacexr, 10x-modified fork used in C1), settings of the C1 runtime
# benchmark except Reference(min_UMI = 10): reference cells carry only panel-gene counts (1.8k genes;
# median 25-400 per cell), so the default min_UMI = 100 would discard most epithelial reference cells
# (and leave <25 cells for several types). Otherwise Reference defaults (n_max_cells=10000), SpatialRNA(coords, counts, colSums), create.RCTD(max_cores=<n_cores>) with default
# UMI_min=100 (bins below 100 UMI are not fitted -> coverage < 100%), run.RCTD(doublet_mode=mode).
# Full-fit weights (results$weights, row-normalised) are written for every fitted bin; in doublet
# mode the doublet-mode call (spot_class, first/second type, weights_doublet) is written too.
# Usage: run_rctd.R <st.h5ad> <ref_dir> <mode: doublet|full> <props.csv.gz> <n_cores> [UMI_min, default 100]
# (UMI_min is lowered only in an explicitly labelled sensitivity arm.)
args <- commandArgs(trailingOnly = TRUE)
st_path <- args[1]; ref_dir <- args[2]; mode <- args[3]; props_out <- args[4]; n_cores <- as.integer(args[5])
umi_min <- if (length(args) >= 6) as.integer(args[6]) else 100L
src_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(src_dir, "s1_common.R"))
suppressPackageStartupMessages(library(spacexr))
desc <- packageDescription("spacexr")
ver <- sprintf("spacexr %s (%s@%s); R %s", desc$Version,
               ifelse(is.null(desc$RemoteRepo), "unknown", paste0(desc$RemoteUsername, "/", desc$RemoteRepo)),
               substr(ifelse(is.null(desc$RemoteSha), "", desc$RemoteSha), 1, 8),
               paste(R.version$major, R.version$minor, sep = "."))

load_s <- NA; fit_s <- NA; n_bins <- NA; n_genes <- NA; n_types <- NA
res <- tryCatch({
  t0 <- proc.time()
  st <- read_h5ad_counts(st_path)
  xy <- read_h5ad_spatial(st_path, length(st$obs))
  ref <- read_h5ad_counts(file.path(ref_dir, "ref.h5ad"))
  meta <- read.csv(file.path(ref_dir, "ref_meta.csv"), stringsAsFactors = FALSE)
  load_s <- elapsed(t0)
  n_bins <- ncol(st$counts); n_genes <- nrow(st$counts)
  stopifnot(identical(meta$barcode, colnames(ref$counts)))

  t1 <- proc.time()
  ct <- as.factor(gsub("/", "_", meta$cell_type)); names(ct) <- meta$barcode
  n_types <- nlevels(ct)
  reference <- Reference(ref$counts, ct, colSums(ref$counts), min_UMI = 10)
  coords <- data.frame(x = xy[, 1], y = xy[, 2], row.names = colnames(st$counts))
  puck <- SpatialRNA(coords, st$counts, colSums(st$counts))
  rm(ref, st); invisible(gc())
  myRCTD <- create.RCTD(puck, reference, max_cores = n_cores, UMI_min = umi_min)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = mode)
  fit_s <- elapsed(t1)

  r <- myRCTD@results
  w <- as.matrix(r$weights)
  n_fit <- nrow(w)
  extra <- ""
  wn <- sweep(w, 1, pmax(rowSums(w), 1e-12), "/")
  df <- data.frame(bin = rownames(wn), wn, check.names = FALSE)
  if (mode == "doublet") {
    tab <- table(r$results_df$spot_class)
    extra <- paste(names(tab), as.integer(tab), sep = "=", collapse = ",")
    rd <- r$results_df[rownames(wn), ]
    wd <- as.matrix(r$weights_doublet)[rownames(wn), , drop = FALSE]
    df$spot_class <- as.character(rd$spot_class)
    df$first_type <- as.character(rd$first_type)
    # the 10x fork's gather_results names this column "scond_type" (typo)
    df$second_type <- as.character(if (!is.null(rd$second_type)) rd$second_type else rd$scond_type)
    df$w_first <- wd[, 1]
    df$w_second <- wd[, 2]
  }
  gz <- gzfile(props_out, "w"); write.csv(df, gz, row.names = FALSE); close(gz)
  write_result(status = "OK", fit_seconds = fit_s, load_seconds = load_s, n_bins = n_bins,
               n_bins_fit = n_fit, n_genes = n_genes, n_types = n_types, version = ver,
               notes = sprintf("doublet_mode=%s; max_cores=%d; UMI_min=%d; spot_class: %s",
                               mode, n_cores, umi_min, extra))
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
