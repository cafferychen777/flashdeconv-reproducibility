#!/bin/bash
#SBATCH --job-name=c2_rctd
#SBATCH --partition=medium
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/rctd_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/results/pseudo_vhd_c2/logs/rctd_%A_%a.err
# Array index -> (res, mode, umi_min) via rctd_configs.txt (line index+1).
set -eo pipefail
source /scratch/user/cafferychen777/FlashDeconv/validation/pseudo_vhd_c2/env.sh; c2_r
read RES MODE UMI < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" rctd_configs.txt)
echo "config: res=$RES mode=$MODE umi_min=$UMI cores=$SLURM_CPUS_PER_TASK"
Rscript run_rctd.R $RES $MODE $UMI $SLURM_CPUS_PER_TASK $C2_OUT
