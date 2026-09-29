#!/bin/bash
#SBATCH --job-name=pen_sparse
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/user/cafferychen777/fd_final/logs/%x_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/logs/%x_%A_%a.err
# Array index = line of tasks.txt (0-based): "s1 <scale> <p>" or "c2 <res_um>".
set -eo pipefail
W=/scratch/user/cafferychen777/fd_final
C=$W/code/penalty_sparsity
OUT=$W/results/penalty_sparsity
NC=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_pensp PYTHONUNBUFFERED=1 PYTHONPATH=$C
read -r DS A1 A2 < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $C/tasks.txt)
cd $C
if [[ $DS == s1 ]]; then
  $W/env/bin/python s1_run.py --scale $A1 --p $A2 --out $OUT/s1
else
  $W/env/bin/python c2_run.py --res $A1 --out $OUT/c2
fi
