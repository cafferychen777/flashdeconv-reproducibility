#!/bin/bash
#SBATCH --job-name=wt_spot
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/user/cafferychen777/fd_final/weighting/logs/spot_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/weighting/logs/spot_%A_%a.err
# Array index -> line (index+1) of spotless_tasks.txt. Resumable.
set -eo pipefail
source /scratch/user/cafferychen777/fd_final/weighting/code/env.sh
NPZ=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" spotless_tasks.txt)
echo "host=$(hostname) $NPZ"
$PY run_weighting_final.py spotless --npz "$NPZ"
