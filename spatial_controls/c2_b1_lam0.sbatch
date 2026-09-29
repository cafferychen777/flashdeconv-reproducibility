#!/bin/bash
#SBATCH -J ctl_b1
#SBATCH -p medium
#SBATCH -t 04:00:00
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --array=0-8
#SBATCH -o /scratch/user/cafferychen777/controls_editor/logs/b1_%A_%a.out
SECS=(CRC_P1 CRC_P2 CRC_P5 SPATCH_COAD SPATCH_OV SPATCH_HCC LUNG_X1 LUNG_X5K OV10X)
cd /scratch/user/cafferychen777/controls_editor/code
export PYTHONPATH=/scratch/user/cafferychen777/b1_pilot/pylib OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_ctlb1
/scratch/user/cafferychen777/fd_final/env/bin/python c2_b1_lam0.py fit ${SECS[$SLURM_ARRAY_TASK_ID]}
