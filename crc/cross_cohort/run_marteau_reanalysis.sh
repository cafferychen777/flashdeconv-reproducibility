#!/bin/bash
#SBATCH --job-name=marteau_annonly
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/analysis/cross_cohort_evidence/marteau_xenium/logs/reanalysis_%j.out

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pnf

cd /scratch/user/cafferychen777/FlashDeconv/analysis/cross_cohort_evidence

echo "=== Starting Marteau annotation-only reanalysis ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

python 01_marteau_xenium_neutrophil_mregdc.py

echo "=== Done ==="
echo "Date: $(date)"
