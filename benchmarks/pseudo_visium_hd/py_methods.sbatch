#!/bin/bash
#SBATCH --job-name=c2_py
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/py_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/py_%A_%a.err
# Array index = resolution in um (2,4,8,16,32). Extra args = method keys.
set -eo pipefail
source /scratch/user/cafferychen777/FlashDeconv/validation/pseudo_vhd_c2/env.sh; c2_py
RES=${RES:-$SLURM_ARRAY_TASK_ID}
python run_py_methods.py --res $RES --methods "$@"
