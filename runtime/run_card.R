#!/usr/bin/env Rscript
# C1 runner: CARD (CPU) with the tutorial defaults: createCARDObject(minCountGene = 100,
# minCountSpot = 5, all reference cell types, sample.varname = patient) + CARD_deconvolution.
# Usage: run_card.R <data_dir> <scale> <props_dir>
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]; scale <- as.integer(args[2]); props_dir <- args[3]
src_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(src_dir, "c1_common.R"))
suppressPackageStartupMessages(library(CARD))
desc <- packageDescription("CARD")
ver <- sprintf("CARD %s (%s); R %s", desc$Version,
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

  t1 <- proc.time()
  sc_meta <- data.frame(cellID = meta$barcode, cellType = meta$cell_type, sampleInfo = meta$patient,
                        row.names = meta$barcode)
  n_types <- length(unique(sc_meta$cellType))
  loc <- data.frame(x = xy[, 1], y = xy[, 2], row.names = colnames(st$counts))
  obj <- createCARDObject(sc_count = ref$counts, sc_meta = sc_meta, spatial_count = st$counts,
                          spatial_location = loc, ct.varname = "cellType",
                          ct.select = unique(sc_meta$cellType), sample.varname = "sampleInfo",
                          minCountGene = 100, minCountSpot = 5)
  rm(ref, st); invisible(gc())
  obj <- CARD_deconvolution(CARD_object = obj)
  fit_s <- elapsed(t1)
  p <- obj@Proportion_CARD
  if (scale <= 10000) {
    gz <- gzfile(file.path(props_dir, sprintf("card_default_%d.csv.gz", scale)), "w")
    write.csv(data.frame(bin = rownames(p), p, check.names = FALSE), gz, row.names = FALSE); close(gz)
  }
  write_result(status = "OK", fit_seconds = fit_s, load_seconds = load_s, n_bins = n_bins,
               n_bins_fit = nrow(p), n_genes = n_genes, n_types = n_types, version = ver,
               notes = "createCARDObject(minCountGene=100, minCountSpot=5) + CARD_deconvolution defaults")
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
