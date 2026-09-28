# Export the Li et al. 2023 (Nat Commun 14:1548) simulated MERFISH datasets
# (Zenodo 10.5281/zenodo.10184476, MERFISH.zip) from RDS to plain CSV so that
# they can be consumed from Python without rpy2.
#
# Each simMERFISH_<res>.RDS holds (built with STdeconvolve::buildBregmaCorpus on
# Moffitt et al. 2018, animal 1, 12 Bregma sections):
#   sim          genes x spots simple_triplet_matrix (spot = sum of cells in a patch)
#   gtSpotTopics spots x 6 ground-truth cell-type proportions
#   annotDf      per-cell table with Bregma and patch_id
#   st_location  spot coordinates
#
# Usage: Rscript li2023_export_merfish.R <data_dir>

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else
  "/Users/apple/Research/FlashDeconv/data/external_benchmarks/li2023_merfish"
raw_dir <- file.path(data_dir, "raw")
out_dir <- file.path(data_dir, "csv")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (res in c(100, 50, 20)) {
  d <- readRDS(file.path(raw_dir, sprintf("simMERFISH_%d.RDS", res)))
  s <- d$sim
  counts <- matrix(0, nrow = s$nrow, ncol = s$ncol,
                   dimnames = s$dimnames)
  counts[cbind(s$i, s$j)] <- s$v
  # sim is already spots x genes

  gt <- as.matrix(d$gtSpotTopics)
  stopifnot(all(rownames(gt) %in% rownames(counts)))
  counts <- counts[rownames(gt), , drop = FALSE]
  loc <- d$st_location[rownames(gt), c("X", "Y")]

  # Spot -> Bregma section (one section per patch)
  pb <- unique(d$annotDf[, c("patch_id", "Bregma")])
  pb <- pb[pb$patch_id %in% rownames(gt), ]
  stopifnot(!any(duplicated(pb$patch_id)))
  bregma <- pb$Bregma[match(rownames(gt), pb$patch_id)]
  n_na <- sum(is.na(bregma))

  meta <- data.frame(spot = rownames(gt), x = loc$X, y = loc$Y,
                     bregma = bregma, n_cells = d$cellCounts[rownames(gt), "counts"])

  write.csv(counts, file.path(out_dir, sprintf("merfish_%d_counts.csv", res)))
  write.csv(gt, file.path(out_dir, sprintf("merfish_%d_gt.csv", res)))
  write.csv(meta, file.path(out_dir, sprintf("merfish_%d_meta.csv", res)),
            row.names = FALSE)
  cat(sprintf("res=%d: %d spots x %d genes, %d cell types, %d spots without Bregma\n",
              res, nrow(counts), ncol(counts), ncol(gt), n_na))
}
