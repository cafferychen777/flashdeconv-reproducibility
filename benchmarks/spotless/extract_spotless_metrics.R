#!/usr/bin/env Rscript
# Extract Spotless benchmark metrics from RDS files
# Output: CSV files for supplementary tables

library(tidyverse)

# Output directory
out_dir <- "validation"

# =============================================================================
# 1. Liver Case Study Metrics
# =============================================================================
cat(strrep("=", 70), "\n")
cat("1. Loading Liver Case Study Metrics\n")
cat(strrep("=", 70), "\n")

liver_file <- "validation/spotless/raw_results/metrics/liver_all_metrics.rds"
if (file.exists(liver_file)) {
  liver_metrics <- readRDS(liver_file)
  cat("Liver metrics loaded successfully\n")
  cat("Structure:\n")
  print(str(liver_metrics))
  cat("\nHead:\n")
  print(head(liver_metrics, 20))

  # Save to CSV
  write.csv(liver_metrics, file.path(out_dir, "spotless_liver_all_methods.csv"), row.names = FALSE)
  cat("\nSaved to: spotless_liver_all_methods.csv\n")
} else {
  cat("Liver file not found!\n")
}

# =============================================================================
# 2. seqFISH Gold Standard Metrics
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("2. Loading seqFISH Gold Standard Metrics\n")
cat(strrep("=", 70), "\n")

seqfish_file <- "validation/spotless/raw_results/metrics/ref_all_metrics_seqfish.rds"
if (file.exists(seqfish_file)) {
  seqfish_metrics <- readRDS(seqfish_file)
  cat("seqFISH metrics loaded successfully\n")
  cat("Structure:\n")
  print(str(seqfish_metrics))
  cat("\nHead:\n")
  print(head(seqfish_metrics, 20))

  # Save to CSV
  write.csv(seqfish_metrics, file.path(out_dir, "spotless_seqfish_all_methods.csv"), row.names = FALSE)
  cat("\nSaved to: spotless_seqfish_all_methods.csv\n")
} else {
  cat("seqFISH file not found!\n")
}

# =============================================================================
# 3. Reference Sensitivity (Liver)
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("3. Loading Reference Sensitivity Metrics\n")
cat(strrep("=", 70), "\n")

ref_sens_file <- "validation/spotless/raw_results/metrics/liver_metrics_ref_sensitivity.rds"
if (file.exists(ref_sens_file)) {
  ref_sens <- readRDS(ref_sens_file)
  cat("Reference sensitivity metrics loaded successfully\n")
  cat("Structure:\n")
  print(str(ref_sens))
  cat("\nHead:\n")
  print(head(ref_sens, 20))

  # Save to CSV
  write.csv(ref_sens, file.path(out_dir, "spotless_liver_ref_sensitivity.csv"), row.names = FALSE)
  cat("\nSaved to: spotless_liver_ref_sensitivity.csv\n")
} else {
  cat("Reference sensitivity file not found!\n")
}

# =============================================================================
# 4. Silver Standard Sensitivity
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("4. Loading Silver Standard Sensitivity Metrics\n")
cat(strrep("=", 70), "\n")

ss_sens_file <- "validation/spotless/raw_results/metrics/ssmetrics_ref_sensitivity.rds"
if (file.exists(ss_sens_file)) {
  ss_sens <- readRDS(ss_sens_file)
  cat("Silver Standard sensitivity metrics loaded successfully\n")
  cat("Structure:\n")
  print(str(ss_sens))
  cat("\nHead:\n")
  print(head(ss_sens, 20))

  # Save to CSV
  write.csv(ss_sens, file.path(out_dir, "spotless_silver_ref_sensitivity.csv"), row.names = FALSE)
  cat("\nSaved to: spotless_silver_ref_sensitivity.csv\n")
} else {
  cat("Silver Standard sensitivity file not found!\n")
}

# =============================================================================
# 5. Scalability
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("5. Loading Scalability Metrics\n")
cat(strrep("=", 70), "\n")

scale_file <- "validation/spotless/raw_results/metrics/scalability.rds"
if (file.exists(scale_file)) {
  scalability <- readRDS(scale_file)
  cat("Scalability metrics loaded successfully\n")
  cat("Structure:\n")
  print(str(scalability))
  cat("\nHead:\n")
  print(head(scalability, 20))

  # Save to CSV
  write.csv(scalability, file.path(out_dir, "spotless_scalability.csv"), row.names = FALSE)
  cat("\nSaved to: spotless_scalability.csv\n")
} else {
  cat("Scalability file not found!\n")
}

cat("\n", strrep("=", 70), "\n")
cat("Done! All available metrics extracted.\n")
cat(strrep("=", 70), "\n")
