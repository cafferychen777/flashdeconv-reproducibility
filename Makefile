# FlashDeconv Reproducibility
# One-command reproduction of all benchmark and analysis results.
#
# Quick start:
#   make benchmark          # Spotless benchmark (requires ~13 GB download)
#   make all                # Full reproduction (all analyses + figures)
#
# Individual targets:
#   make download           # Download Spotless data from Zenodo
#   make convert            # Convert RDS to MTX (requires R + Seurat)
#   make validate           # Verify data integrity
#   make figures            # Generate all main figures
#   make clean              # Remove results and figures (keeps data)
#
# Environment variables:
#   DATA_DIR     (default: ./data)
#   RESULTS_DIR  (default: ./results)
#   FIGURES_DIR  (default: ./figures)

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

DATA_DIR      ?= ./data
SPOTLESS_DIR  := $(DATA_DIR)/spotless
CONVERTED_DIR := $(SPOTLESS_DIR)/converted
BRAIN_DIR     := $(DATA_DIR)/mouse_brain
RESULTS_DIR   ?= ./results
FIGURES_DIR   ?= ./figures

.PHONY: all benchmark download convert validate verify-checksums \
        benchmark-silver benchmark-gold benchmark-liver benchmark-melanoma \
        benchmark-scalability \
        visium-hd cortex-lamination leverage-mechanism figures \
        clean help

# ── Top-level targets ─────────────────────────────────────────────

help:
	@echo "Usage:  make <target>"
	@echo ""
	@echo "  benchmark     Run full Spotless benchmark suite (download + convert + run)"
	@echo "  all           Full reproduction: benchmark + analyses + figures"
	@echo "  download      Download Spotless data from Zenodo (~13 GB)"
	@echo "  convert       Convert RDS files to Python-readable MTX"
	@echo "  validate      Verify data directory structure and file counts"
	@echo "  clean         Remove results and figures (keeps downloaded data)"
	@echo "  help          Show this message"

all: benchmark visium-hd cortex-lamination leverage-mechanism figures

# ── Spotless Benchmark ────────────────────────────────────────────

download: verify-checksums

$(SPOTLESS_DIR)/reference:
	bash scripts/download_spotless_data.sh $(SPOTLESS_DIR)

verify-checksums: $(SPOTLESS_DIR)/reference
	bash scripts/verify_checksums.sh $(SPOTLESS_DIR)

convert: download
	Rscript scripts/convert_spotless_data.R $(SPOTLESS_DIR)

validate: convert
	bash scripts/validate_data.sh $(SPOTLESS_DIR)

benchmark: validate benchmark-silver benchmark-gold benchmark-liver benchmark-melanoma benchmark-scalability
	@echo ""
	@echo "============================================================"
	@echo "Benchmark complete. Results in $(RESULTS_DIR)/"
	@echo "============================================================"

benchmark-silver: validate
	python benchmarks/benchmark_silver_standards.py \
		--data_dir $(CONVERTED_DIR) --output_dir $(RESULTS_DIR)

benchmark-gold: validate
	python benchmarks/benchmark_gold_standard.py \
		--data_dir $(CONVERTED_DIR) --output_dir $(RESULTS_DIR)

benchmark-liver: validate
	python benchmarks/benchmark_liver.py \
		--data_dir $(CONVERTED_DIR) --output_dir $(RESULTS_DIR)

benchmark-melanoma: validate
	python benchmarks/benchmark_melanoma.py \
		--data_dir $(CONVERTED_DIR) --output_dir $(RESULTS_DIR)

benchmark-scalability:
	python benchmarks/benchmark_scalability.py --output_dir $(RESULTS_DIR)

# ── Analyses ──────────────────────────────────────────────────────

visium-hd:
	bash scripts/download_visium_hd_data.sh $(DATA_DIR)
	python analysis/resolution_horizon_analysis.py \
		--data_dir $(DATA_DIR) --output_dir $(RESULTS_DIR) --bins 16,32,64,128
	python analysis/tuft_stem_discovery.py \
		--data_dir $(DATA_DIR) --output_dir $(RESULTS_DIR)

cortex-lamination:
	bash scripts/download_cell2location_data.sh $(BRAIN_DIR)
	python analysis/cortex_deconvolution.py \
		--data_dir $(DATA_DIR) --output_dir $(RESULTS_DIR)

leverage-mechanism:
	bash scripts/download_cell2location_data.sh $(BRAIN_DIR)
	python analysis/leverage_deep_dive.py \
		--data_dir $(DATA_DIR) --output_dir $(RESULTS_DIR)

# ── Figures ───────────────────────────────────────────────────────

figures:
	python figures/main/figure2_leverage_mechanism.py \
		--results_dir $(RESULTS_DIR) --output_dir $(FIGURES_DIR)
	python figures/main/figure5_cortex_lamination.py \
		--results_dir $(RESULTS_DIR) --output_dir $(FIGURES_DIR)
	python figures/main/figure7_tuft_discovery.py \
		--results_dir $(RESULTS_DIR) --output_dir $(FIGURES_DIR)

# ── Cleanup ───────────────────────────────────────────────────────

clean:
	rm -rf $(RESULTS_DIR) $(FIGURES_DIR)/*.png $(FIGURES_DIR)/*.pdf
	@echo "Cleaned results and figures. Downloaded data preserved in $(DATA_DIR)/."
