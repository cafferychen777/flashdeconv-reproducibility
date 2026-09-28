#!/bin/bash
#SBATCH --job-name=b2_s2
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/scratch/user/cafferychen777/fd_b2/logs/s2_%j.out
#SBATCH --error=/scratch/user/cafferychen777/fd_b2/logs/s2_%j.err
set -eo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_b2 MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_b2
export PYTHONPATH=/scratch/user/cafferychen777/fd_b2/pkg
cd /scratch/user/cafferychen777/fd_b2/code
/scratch/user/cafferychen777/fd_final/env/bin/python b2_stage2.py
