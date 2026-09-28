#!/bin/bash
# Submit C1 arrays. Usage: bash submit.sh <stage>
#   stage smoke : CPU 1e4 cells + cell2location reference training + c2l 1e4
#   stage prod  : all larger scales (CPU mid/big, CARD large, c2l spatial >= 1e5)
# CPU: 32 CPUs (SLURM cpus = hyperthreads; 16 physical cores on AMD EPYC 7763) per run.
# GPU: 1x A30 + 16 CPUs per run. Wall cap per run: 24 h (monitor.py), SLURM limit 24.5 h.
set -euo pipefail
BASE=/scratch/user/cafferychen777/FlashDeconv/validation/runtime_benchmark_c1
T=$BASE/tasks
S="$BASE/run_task.sbatch"
n() { echo $(( $(grep -c . "$1") - 1 )); }
CPU="--partition=long --cpus-per-task=32 --time=1-00:30:00"
GPU="--partition=gpu --gres=gpu:a30:1 --cpus-per-task=16 --mem=128G --time=1-00:30:00"
case ${1:-} in
  smoke)
    sbatch --parsable -J c1_cpu1e4 $CPU --mem=64G --array=0-$(n $T/cpu_1e4.txt)%4 --export=ALL,TASKS=$T/cpu_1e4.txt "$S"
    REF=$(sbatch --parsable -J c1_c2lref $GPU --array=0-0 --export=ALL,TASKS=$T/gpu_ref.txt "$S")
    echo "$REF"
    sbatch --parsable -J c1_c2l1e4 $GPU --dependency=afterok:$REF --array=0-0 --export=ALL,TASKS=$T/gpu_spatial.txt "$S"
    ;;
  prod)
    sbatch --parsable -J c1_cpumid $CPU --mem=384G --array=0-$(n $T/cpu_mid.txt)%3 --export=ALL,TASKS=$T/cpu_mid.txt "$S"
    sbatch --parsable -J c1_cpubig $CPU --mem=500G --array=0-$(n $T/cpu_big.txt)%2 --export=ALL,TASKS=$T/cpu_big.txt "$S"
    sbatch --parsable -J c1_card $CPU --mem=500G --array=0-$(n $T/card_large.txt)%1 --export=ALL,TASKS=$T/card_large.txt "$S"
    # REF_JOB = job id of the c1_c2lref job from the smoke stage
    sbatch --parsable -J c1_c2l $GPU --dependency=afterok:${REF_JOB:?set REF_JOB} --array=1-$(n $T/gpu_spatial.txt)%1 --export=ALL,TASKS=$T/gpu_spatial.txt "$S"
    ;;
  *) echo "usage: submit.sh smoke|prod"; exit 1 ;;
esac
