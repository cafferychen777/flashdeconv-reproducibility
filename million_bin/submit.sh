#!/bin/bash
# Submit S1 arrays (C1 resource limits: 32 CPUs, <=500 GB, 24 h cap; GPU: 1x A30 + 16 CPUs).
# Usage: bash submit.sh small|mid|big|gpu
set -euo pipefail
BASE=/scratch/user/cafferychen777/FlashDeconv/validation/s1_merfish_benchmark
T=$BASE/tasks
S="$BASE/run_task.sbatch"
n() { echo $(( $(grep -c . "$1") - 1 )); }
CPU="--partition=long --cpus-per-task=32 --time=1-00:30:00"
GPU="--partition=gpu --gres=gpu:a30:1 --cpus-per-task=16 --mem=128G --time=1-00:30:00"
case ${1:-} in
  small) sbatch --parsable -J s1_small $CPU --mem=96G --array=0-$(n $T/cpu_small.txt)%5 --export=ALL,TASKS=$T/cpu_small.txt "$S" ;;
  mid)   sbatch --parsable -J s1_mid $CPU --mem=160G --array=0-$(n $T/cpu_mid.txt)%4 --export=ALL,TASKS=$T/cpu_mid.txt "$S"
         sbatch --parsable -J s1_cardmid $CPU --mem=500G --array=0-$(n $T/card_mid.txt)%1 --export=ALL,TASKS=$T/card_mid.txt "$S" ;;
  big)   sbatch --parsable -J s1_big $CPU --mem=500G --array=0-$(n $T/cpu_big.txt)%2 --export=ALL,TASKS=$T/cpu_big.txt "$S" ;;
  gpu)   REF=$(sbatch --parsable -J s1_c2lref $GPU --array=0-1%1 --export=ALL,TASKS=$T/gpu_ref.txt "$S"); echo "$REF"
         sbatch --parsable -J s1_c2l $GPU --dependency=afterany:$REF --array=0-$(n $T/gpu_spatial.txt)%1 --export=ALL,TASKS=$T/gpu_spatial.txt "$S" ;;
  *) echo "usage: submit.sh small|mid|big|gpu"; exit 1 ;;
esac
