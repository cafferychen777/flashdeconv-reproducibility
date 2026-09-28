#!/bin/bash
#SBATCH -J fdfin_c2rs
#SBATCH -p medium
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/user/cafferychen777/fd_final/logs/%x_%j.out
# Standard-metric rescoring of all C2 configs (C2 read-only preds + final FlashDeconv).
# Output name via env FDFIN_C2_METRICS (default c2_standard_metrics_final.csv).
W=/scratch/user/cafferychen777/fd_final
export OMP_NUM_THREADS=4 PYTHONPATH=/scratch/user/cafferychen777/FlashDeconv/validation
$W/env/bin/python $W/code/benchmarks/c2/rescore_standard.py 2 4 8 16 32
$W/env/bin/python $W/code/benchmarks/determinism.py c2 $W/results/c2
