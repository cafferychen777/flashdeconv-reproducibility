#!/bin/bash
#SBATCH --job-name=c2_prep
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/prep_%j.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/prep_%j.err
set -eo pipefail
source /scratch/user/cafferychen777/FlashDeconv/validation/pseudo_vhd_c2/env.sh; c2_py
python prepare_bins.py "$@"
