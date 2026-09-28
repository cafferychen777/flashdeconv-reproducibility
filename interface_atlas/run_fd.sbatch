#!/bin/bash
#SBATCH -J b1_fd
#SBATCH -p medium
#SBATCH -t 04:00:00
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --array=0-5
#SBATCH -o /scratch/user/cafferychen777/b1_pilot/logs/fd_%A_%a.out
SECS=(CRC_P1 CRC_P2 CRC_P5 SPATCH_COAD SPATCH_OV SPATCH_HCC)
cd /scratch/user/cafferychen777/b1_pilot/code
export PYTHONPATH=/scratch/user/cafferychen777/b1_pilot/pylib OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "node $(hostname) start $(date)"
/scratch/user/cafferychen777/fd_final/env/bin/python run_fd.py ${SECS[$SLURM_ARRAY_TASK_ID]}
