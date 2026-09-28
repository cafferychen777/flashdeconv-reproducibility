#!/bin/bash
#SBATCH --job-name=wt_xen
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/cafferychen777/fd_final/weighting/logs/xen_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/weighting/logs/xen_%A_%a.err
# Array index = bin size in um (2, 4, 8). Resumable.
set -eo pipefail
source /scratch/user/cafferychen777/fd_final/weighting/code/env.sh
$PY run_weighting_final.py xenium --res "$SLURM_ARRAY_TASK_ID"
