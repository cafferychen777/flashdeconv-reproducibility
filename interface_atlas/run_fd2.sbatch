#!/bin/bash
#SBATCH -J b1_fd2
#SBATCH -p medium
#SBATCH -t 04:00:00
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH -o /scratch/user/cafferychen777/b1_pilot/logs/fd2_%A_%a.out
SECS=(LUNG_X1 LUNG_X5K OV10X)
cd /scratch/user/cafferychen777/b1_pilot/code
export PYTHONPATH=/scratch/user/cafferychen777/b1_pilot/pylib OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "node $(hostname) start $(date)"
/scratch/user/cafferychen777/fd_final/env/bin/python run_fd.py ${SECS[$SLURM_ARRAY_TASK_ID]}
