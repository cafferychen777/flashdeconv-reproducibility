#!/bin/bash
#SBATCH --job-name=ctl_crc0
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --array=0-2
#SBATCH --output=/scratch/user/cafferychen777/controls_editor/logs/crc_%A_%a.out
set -eo pipefail
SAMPLES=(P1_CRC P2_CRC P5_CRC)
S=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
NC=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_ctl MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_ctl PYTHONUNBUFFERED=1
cd /scratch/user/cafferychen777/controls_editor/code
/scratch/user/cafferychen777/fd_final/env/bin/python c2_crc_lam0.py --sample "$S"
