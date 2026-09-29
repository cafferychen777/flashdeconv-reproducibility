#!/bin/bash
#SBATCH -J ctl_b1g
#SBATCH -p short
#SBATCH -t 01:30:00
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -o /scratch/user/cafferychen777/controls_editor/logs/b1grad_%j.out
cd /scratch/user/cafferychen777/controls_editor/code
export PYTHONPATH=/scratch/user/cafferychen777/b1_pilot/pylib OMP_NUM_THREADS=4
/scratch/user/cafferychen777/fd_final/env/bin/python c2_b1_lam0.py grad
