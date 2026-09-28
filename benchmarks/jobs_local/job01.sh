#!/bin/bash
cd /Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/spotless && FD_FITLOG=/Users/apple/Research/FlashDeconv/results/rerun_final/fitlogs/local_spotless_silver.csv FD_TAG=spotless_silver /Users/apple/Research/FlashDeconv/.venv/bin/python run_silver_gold.py silver > /Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless/log_silver.txt 2>&1
echo "job01 rc=$?" >> /Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/jobs_local.done
