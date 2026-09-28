#!/bin/bash
#SBATCH --job-name=c2_eval
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/eval_%j.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/eval_%j.err
set -eo pipefail
source /scratch/user/cafferychen777/FlashDeconv/validation/pseudo_vhd_c2/env.sh; c2_py
python evaluate.py "$@"
