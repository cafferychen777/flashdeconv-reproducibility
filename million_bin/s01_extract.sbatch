#!/bin/bash
#SBATCH --job-name=s1_extract
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/logs/s1/%x_%j.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/logs/s1/%x_%j.err
set -euo pipefail
B=/scratch/user/cafferychen777/FlashDeconv/validation/s1_merfish_benchmark
D=/scratch/user/cafferychen777/FlashDeconv/data/s1_merfish
hostname; df -h ${TMPDIR:-/tmp} | tail -1
/scratch/user/cafferychen777/fd_final/env/bin/python $B/s01_extract_merfish.py $D/.dryad_url $D/merfish_all_rawcounts.h5ad
