#!/bin/bash
#
# Validate that Spotless benchmark data has been correctly downloaded and converted.
#
# Usage:
#     bash scripts/validate_data.sh [DATA_DIR]
#
# Example:
#     bash scripts/validate_data.sh ./data/spotless

set -euo pipefail

DATA_DIR="${1:-./data/spotless}"
INPUT_DIR="$DATA_DIR"
CONVERTED_DIR="$DATA_DIR/converted"
ERRORS=0
WARNINGS=0

if [ ! -d "$INPUT_DIR/reference" ] && [ -d "$INPUT_DIR/standards/reference" ]; then
    INPUT_DIR="$INPUT_DIR/standards"
fi

echo "============================================================"
echo "Validating Spotless Benchmark Data"
echo "Data directory: $DATA_DIR"
echo "Raw data directory: $INPUT_DIR"
echo "============================================================"

# --- Phase 1: Check raw RDS directories ---

echo ""
echo "Phase 1: Checking extracted RDS directories"
echo "------------------------------------------------------------"

check_rds_dir() {
    local dir="$1"
    local expected_min="$2"
    local label="$3"

    if [ ! -d "$INPUT_DIR/$dir" ]; then
        echo "  FAIL  $label: directory $dir/ not found"
        ERRORS=$((ERRORS + 1))
        return
    fi

    local count
    count=$(find "$INPUT_DIR/$dir" -name "*.rds" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -lt "$expected_min" ]; then
        echo "  WARN  $label: found $count RDS files (expected >= $expected_min)"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  OK    $label: $count RDS files"
    fi
}

check_silver_tests() {
    local expected_min="$1"
    local label="$2"
    local count=0
    local current_count=0
    local legacy_count=0

    if [ -d "$INPUT_DIR/test_silver_standard" ]; then
        legacy_count=$(find "$INPUT_DIR/test_silver_standard" -maxdepth 1 -name "*.rds" 2>/dev/null | wc -l | tr -d ' ')
    fi

    for root in "$INPUT_DIR" "$INPUT_DIR/test_silver_standard"; do
        if [ -d "$root" ]; then
            current_count=$(find "$root" -mindepth 2 -maxdepth 2 -type f -path "*/silver_standard_*-*/*.rds" 2>/dev/null | wc -l | tr -d ' ')
            count=$((count + current_count))
        fi
    done

    count=$((count + legacy_count))

    if [ "$count" -lt "$expected_min" ]; then
        echo "  WARN  $label: found $count RDS files (expected >= $expected_min)"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  OK    $label: $count RDS files"
    fi
}

check_rds_dir "reference"            6  "Silver Standard references"
check_silver_tests                  56 "Silver Standard test datasets"
check_rds_dir "gold_standard_1"      1  "Gold Standard 1 (seqFISH+ cortex)"
check_rds_dir "gold_standard_2"      1  "Gold Standard 2 (seqFISH+ OB)"
check_rds_dir "gold_standard_3"      1  "Gold Standard 3 (STARMap)"
check_rds_dir "liver"                5  "Liver case study"
check_rds_dir "melanoma"             4  "Melanoma case study"

# --- Phase 2: Check converted MTX files ---

echo ""
echo "Phase 2: Checking converted MTX files"
echo "------------------------------------------------------------"

if [ ! -d "$CONVERTED_DIR" ]; then
    echo "  SKIP  Converted directory not found: $CONVERTED_DIR"
    echo "        Run: Rscript scripts/convert_spotless_data.R $DATA_DIR"
else
    check_converted() {
        local pattern="$1"
        local expected="$2"
        local label="$3"

        local count
        count=$(find "$CONVERTED_DIR" -name "${pattern}_counts.mtx" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$count" -lt "$expected" ]; then
            echo "  WARN  $label: $count converted (expected $expected)"
            WARNINGS=$((WARNINGS + 1))
        else
            echo "  OK    $label: $count converted"
        fi
    }

    check_converted "reference_*"          6  "Silver references"
    check_converted "silver_*"            56  "Silver test datasets"
    check_converted "liver_*"              5  "Liver datasets"
    check_converted "melanoma_*"           4  "Melanoma datasets"

    # Gold standard files have varied names; just count total
    gold_count=$(find "$CONVERTED_DIR" -name "*gold*_counts.mtx" -o -name "*Eng2019*_counts.mtx" -o -name "*Wang2018*_counts.mtx" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$gold_count" -lt 1 ]; then
        echo "  WARN  Gold Standard: no converted files found"
        WARNINGS=$((WARNINGS + 1))
    else
        echo "  OK    Gold Standard: $gold_count converted"
    fi
fi

# --- Summary ---

echo ""
echo "============================================================"
echo "Validation Summary"
echo "============================================================"
echo "  Errors:   $ERRORS"
echo "  Warnings: $WARNINGS"

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "ERRORS found. Data download or extraction likely failed."
    echo "Re-run: bash scripts/download_spotless_data.sh $DATA_DIR"
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo ""
    echo "WARNINGS found. Some datasets may be missing."
    echo "Benchmarks can still run, but results may be incomplete."
    exit 0
else
    echo ""
    echo "All checks passed. Data is ready for benchmarking."
    exit 0
fi
