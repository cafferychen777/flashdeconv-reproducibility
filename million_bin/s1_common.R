# Shared helpers for S1 R runners (copied from runtime_benchmark_c1; read AnnData .h5ad CSR counts with rhdf5).
suppressPackageStartupMessages({
  library(rhdf5)
  library(Matrix)
})

read_h5ad_counts <- function(path) {
  # CSR (obs x var) in the file == CSC (var x obs) in R: genes x bins dgCMatrix.
  shape <- as.integer(h5readAttributes(path, "X")$shape)
  obs <- as.character(h5read(path, "obs/_index"))
  var <- as.character(h5read(path, "var/_index"))
  x <- as.numeric(h5read(path, "X/data"))
  i <- as.integer(h5read(path, "X/indices"))
  p <- as.integer(h5read(path, "X/indptr"))
  m <- sparseMatrix(i = i + 1L, p = p, x = x, dims = c(shape[2], shape[1]),
                    dimnames = list(var, obs), repr = "C")
  rm(x, i, p)
  list(counts = m, obs = obs, var = var)
}

read_h5ad_spatial <- function(path, n_obs) {
  xy <- h5read(path, "obsm/spatial")
  if (nrow(xy) != n_obs) xy <- t(xy)
  xy
}

json_escape <- function(s) gsub('"', "'", gsub("\\\\", "/", gsub("[\r\n\t]", " ", s)))

write_result <- function(...) {
  kv <- list(...)
  parts <- vapply(names(kv), function(k) {
    v <- kv[[k]]
    if (is.numeric(v)) sprintf('"%s": %s', k, format(v, digits = 10))
    else sprintf('"%s": "%s"', k, json_escape(as.character(v)))
  }, character(1))
  txt <- paste0("{", paste(parts, collapse = ", "), "}")
  path <- Sys.getenv("C1_RESULT_JSON")
  if (nzchar(path)) writeLines(txt, path)
  cat("[runner] result:", txt, "\n")
}

status_from_error <- function(msg) {
  if (grepl("cannot allocate|memory exhausted|std::bad_alloc|vector memory limit|Cannot allocate", msg,
            ignore.case = TRUE)) "OOM" else "ERROR"
}

elapsed <- function(t0) as.numeric((proc.time() - t0)["elapsed"])
