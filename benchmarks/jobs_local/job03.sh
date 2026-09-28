#!/bin/bash
cd /Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/spotless && FD_FITLOG=/Users/apple/Research/FlashDeconv/results/rerun_final/fitlogs/local_liver.csv FD_TAG=liver /Users/apple/Research/FlashDeconv/.venv/bin/python run_liver.py all > /Users/apple/Research/FlashDeconv/results/rerun_final/benchmarks/spotless/log_liver.txt 2>&1
echo "job03 rc=$?" >> /Users/apple/Research/FlashDeconv/validation/rerun_final/benchmarks/jobs_local.done
