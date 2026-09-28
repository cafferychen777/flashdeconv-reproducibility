#!/bin/bash
#SBATCH --job-name=c1_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/logs/c1/prep_%j.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/logs/c1/prep_%j.err
set -euo pipefail
BASE=/scratch/user/cafferychen777/FlashDeconv/validation/runtime_benchmark_c1
hostname
/scratch/user/cafferychen777/venvs/c1_fd/bin/python "$BASE/prep_data.py"
du -sh /scratch/user/cafferychen777/FlashDeconv/data/runtime_benchmark_c1
