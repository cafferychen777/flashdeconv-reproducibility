#!/bin/bash
# Run locally once arseven job 2693897 (C2 standard rescoring queued with
# --dependency=afterany:2693139, output c2_standard_metrics_final_allrctd.csv) has
# finished. If 2693139 fails or is resubmitted, run on arseven instead:
#   sbatch --export=ALL,FDFIN_C2_METRICS=c2_standard_metrics_final_allrctd.csv \
#     /scratch/user/cafferychen777/fd_final/code/benchmarks/c2/rescore.sbatch
set -euo pipefail
P=/Users/apple/Research/FlashDeconv
rsync -a arseven:/scratch/user/cafferychen777/fd_final/results/c2/c2_standard_metrics_final_allrctd.csv \
  $P/results/rerun_final/benchmarks/c2/
$P/.venv/bin/python $P/validation/rerun_final/benchmarks/build_part_summary_arseven.py
