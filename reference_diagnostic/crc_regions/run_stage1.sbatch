#!/bin/bash
#SBATCH --job-name=b2_s1
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --array=0-2
#SBATCH --output=/scratch/user/cafferychen777/fd_b2/logs/s1_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/fd_b2/logs/s1_%A_%a.err
set -eo pipefail
SAMPLES=(P1_CRC P2_CRC P5_CRC)
NC=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_b2 MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_b2 PYTHONUNBUFFERED=1
export PYTHONPATH=/scratch/user/cafferychen777/fd_b2/pkg
cd /scratch/user/cafferychen777/fd_b2/code
/scratch/user/cafferychen777/fd_final/env/bin/python b2_stage1.py --sample "${SAMPLES[$SLURM_ARRAY_TASK_ID]}"
