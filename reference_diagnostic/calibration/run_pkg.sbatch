#!/bin/bash
#SBATCH --job-name=rd2_pkg
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --array=0-3
#SBATCH --output=/scratch/user/cafferychen777/fd_rd2/logs/pkg_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/fd_rd2/logs/pkg_%A_%a.err
set -eo pipefail
ARGS=("crc P1_CRC" "crc P2_CRC" "crc P5_CRC" "runtime")
NC=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_rd2 MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_rd2 PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/user/cafferychen777/fd_rd2/pkg
export RD2_OUT=/scratch/user/cafferychen777/fd_rd2/results
cd /scratch/user/cafferychen777/fd_rd2/code
/scratch/user/cafferychen777/fd_final/env/bin/python pkg_crc_runtime.py ${ARGS[$SLURM_ARRAY_TASK_ID]}
