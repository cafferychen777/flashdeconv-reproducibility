#!/bin/bash
#SBATCH --job-name=fdfin_c2
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/user/cafferychen777/fd_final/logs/%x_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/logs/%x_%A_%a.err
# Array index = resolution in um (2,4,8,16,32).
set -eo pipefail
W=/scratch/user/cafferychen777/fd_final
NC=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_fdfinc2 PYTHONUNBUFFERED=1
export PYTHONPATH=$W/code FD_FITLOG=$W/results/fitlogs/c2_fitlog_${SLURM_ARRAY_TASK_ID}um.csv
mkdir -p $W/results/fitlogs
MODES="final_default_auto final_default_l0"
[[ $SLURM_ARRAY_TASK_ID == 16 ]] && MODES="$MODES final_default_auto_seed0"
cd $W/code/benchmarks/c2
$W/env/bin/python run_fd_final.py --res $SLURM_ARRAY_TASK_ID --modes $MODES
