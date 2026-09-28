#!/bin/bash
#SBATCH --job-name=crc_final
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/user/cafferychen777/fd_final/crc/logs/%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/crc/logs/%A_%a.err
# One array task per patient (P1, P2, P5): final FlashDeconv defaults + all CRC analyses.
set -eo pipefail
SAMPLES=(P1_CRC P2_CRC P5_CRC)
S=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
NC=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_crcfinal MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_crcfinal
export PYTHONUNBUFFERED=1
export FD_FITLOG=/scratch/user/cafferychen777/fd_final/crc/fitlogs/crc_analysis_${S}.csv FD_TAG=crc_analysis_${S}
lscpu | grep -E "Model name|^CPU\(s\)" || true
cd /scratch/user/cafferychen777/fd_final/crc/code
/scratch/user/cafferychen777/fd_final/env/bin/python run_final.py --sample "$S"
