#!/usr/bin/env Rscript
# Extract ALL Spotless benchmark metrics from RDS files

library(tidyverse)
out_dir <- "validation"
metrics_dir <- "validation/spotless/raw_results/metrics"

cat("Extracting all Spotless metrics...\n\n")

# =============================================================================
# 1. Melanoma Metrics
# =============================================================================
cat("1. Melanoma Metrics\n")
melanoma_file <- file.path(metrics_dir, "melanoma_metrics.rds")
if (file.exists(melanoma_file)) {
  melanoma <- readRDS(melanoma_file)

  # Extract JSD (main comparison metric)
  melanoma_jsd <- melanoma$jsd
  melanoma_jsd <- melanoma_jsd %>% arrange(jsd)
  melanoma_jsd$rank <- 1:nrow(melanoma_jsd)

  cat("\nMelanoma JSD Rankings:\n")
  print(melanoma_jsd)

  write.csv(melanoma_jsd, file.path(out_dir, "spotless_melanoma_jsd.csv"), row.names = FALSE)
  cat("\nSaved: spotless_melanoma_jsd.csv\n\n")
}

# =============================================================================
# 2. seqFISH Rankings
# =============================================================================
cat("2. seqFISH Rankings\n")
seqfish_file <- file.path(metrics_dir, "seqfish_rankings.rds")
if (file.exists(seqfish_file)) {
  seqfish <- readRDS(seqfish_file)
  cat("\nStructure:\n")
  print(str(seqfish))
  cat("\nData:\n")
  print(seqfish)
  write.csv(seqfish, file.path(out_dir, "spotless_seqfish_rankings.csv"), row.names = FALSE)
  cat("\nSaved: spotless_seqfish_rankings.csv\n\n")
}

# =============================================================================
# 3. STARMap Rankings
# =============================================================================
cat("3. STARMap Rankings\n")
starmap_file <- file.path(metrics_dir, "starmap_rankings.rds")
if (file.exists(starmap_file)) {
  starmap <- readRDS(starmap_file)
  cat("\nStructure:\n")
  print(str(starmap))
  cat("\nData:\n")
  print(starmap)
  write.csv(starmap, file.path(out_dir, "spotless_starmap_rankings.csv"), row.names = FALSE)
  cat("\nSaved: spotless_starmap_rankings.csv\n\n")
}

# =============================================================================
# 4. Silver Standard All Methods
# =============================================================================
cat("4. Silver Standard All Methods\n")
silver_file <- file.path(metrics_dir, "ref_all_metrics_silver.rds")
if (file.exists(silver_file)) {
  silver <- readRDS(silver_file)
  cat("\nStructure:\n")
  print(str(silver))
  cat("\nFirst 20 rows:\n")
  print(head(silver, 20))

  # Summary by method
  if ("method" %in% names(silver) && "corr" %in% names(silver)) {
    summary_df <- silver %>%
      group_by(method) %>%
      summarise(
        n = n(),
        corr_mean = mean(corr, na.rm = TRUE),
        corr_std = sd(corr, na.rm = TRUE),
        rmse_mean = mean(RMSE, na.rm = TRUE),
        jsd_mean = mean(jsd, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(desc(corr_mean))

    cat("\nSummary by method:\n")
    print(summary_df)
    write.csv(summary_df, file.path(out_dir, "spotless_silver_summary_by_method.csv"), row.names = FALSE)
  }

  write.csv(silver, file.path(out_dir, "spotless_silver_all_methods.csv"), row.names = FALSE)
  cat("\nSaved: spotless_silver_all_methods.csv\n\n")
}

# =============================================================================
# 5. Liver Rankings
# =============================================================================
cat("5. Liver Rankings\n")
liver_rank_file <- file.path(metrics_dir, "liver_all_rankings.rds")
if (file.exists(liver_rank_file)) {
  liver_rank <- readRDS(liver_rank_file)
  cat("\nStructure:\n")
  print(str(liver_rank))
  cat("\nData:\n")
  print(liver_rank)
  write.csv(liver_rank, file.path(out_dir, "spotless_liver_rankings.csv"), row.names = FALSE)
  cat("\nSaved: spotless_liver_rankings.csv\n\n")
}

# =============================================================================
# 6. Runtime
# =============================================================================
cat("6. Runtime\n")
runtime_file <- file.path(metrics_dir, "runtime.rds")
if (file.exists(runtime_file)) {
  runtime <- readRDS(runtime_file)
  cat("\nStructure:\n")
  print(str(runtime))
  cat("\nFirst 30 rows:\n")
  print(head(runtime, 30))

  # Summary by method
  if ("method" %in% names(runtime) && "time" %in% names(runtime)) {
    runtime_summary <- runtime %>%
      group_by(method) %>%
      summarise(
        mean_time_sec = mean(time, na.rm = TRUE),
        median_time_sec = median(time, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(mean_time_sec)
    cat("\nRuntime summary:\n")
    print(runtime_summary)
  }

  write.csv(runtime, file.path(out_dir, "spotless_runtime.csv"), row.names = FALSE)
  cat("\nSaved: spotless_runtime.csv\n\n")
}

cat("\n=== Done! ===\n")
