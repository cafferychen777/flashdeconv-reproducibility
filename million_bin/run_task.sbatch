#!/bin/bash
#SBATCH --job-name=s1
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/logs/s1/%x_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/logs/s1/%x_%A_%a.err
# Generic S1 array task. Resources are given at submission (C1 limits: 32 CPUs, <=500 GB, 24 h cap).
# TASKS (env): whitespace-separated file "method mode ref stfile" (line = array index + 1).
set -uo pipefail
BASE=/scratch/user/cafferychen777/FlashDeconv/validation/s1_merfish_benchmark
DATA=/scratch/user/cafferychen777/FlashDeconv/data/s1_merfish
RES=/scratch/user/cafferychen777/FlashDeconv/results/s1_merfish_benchmark
FD=/scratch/user/cafferychen777/fd_final/env/bin/python
V=/scratch/user/cafferychen777/venvs
RENV=/scratch/user/cafferychen777/conda_envs/c1_r
CSV=$RES/s1_runtime.csv
mkdir -p "$RES/props" "$RES/run_json"

read -r METHOD MODE REF STF < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
ST=$DATA/st/$STF
SCALE=$(echo "$STF" | sed -E 's/^st_([0-9]+)\.h5ad$/\1/')
REFDIR=$DATA/$REF
PROPS=$RES/props/${METHOD}_${MODE}_${REF}_${SCALE}.csv.gz
NC=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl
export PYTHONPATH=$BASE
echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD mode=$MODE ref=$REF st=$STF host=$(hostname) cpus=$NC"

DEVICE=cpu
case $METHOD in
  flashdeconv) CMD=("$FD" "$BASE/run_flashdeconv.py" --st "$ST" --ref-dir "$REFDIR" --props "$PROPS") ;;
  cell2location) DEVICE=gpu
               if [[ $MODE == ref_train ]]; then SCALE=0; CMD=("$V/c1_c2l/bin/python" "$BASE/run_c2l.py" --ref-dir "$REFDIR" --mode ref_train)
               else CMD=("$V/c1_c2l/bin/python" "$BASE/run_c2l.py" --ref-dir "$REFDIR" --st "$ST" --mode "$MODE" --props "$PROPS"); fi ;;
  rctd)        export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  # RCTD parallelizes over max_cores workers
               RMODE=${MODE%%_umi*}; UMIMIN=100; [[ $MODE == *_umi* ]] && UMIMIN=${MODE##*_umi}
               CMD=("$RENV/bin/Rscript" "$BASE/run_rctd.R" "$ST" "$REFDIR" "$RMODE" "$PROPS" "$NC" "$UMIMIN") ;;
  card)        MCG=100; [[ $MODE == *_min* ]] && MCG=${MODE##*_min}
               CMD=("$RENV/bin/Rscript" "$BASE/run_card.R" "$ST" "$REFDIR" "$PROPS" "$MCG") ;;
  *) echo "unknown method $METHOD"; exit 2 ;;
esac
if [[ $METHOD == rctd || $METHOD == card ]]; then
  export R_LIBS_USER="$RENV/lib/R/library" R_LIBS="$RENV/lib/R/library" R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
fi

"$FD" "$BASE/monitor.py" --method "$METHOD" --mode "${REF}-${MODE}" --scale "$SCALE" --rep 1 \
  --device "$DEVICE" --timeout-hours "${S1_TIMEOUT_H:-24}" --csv "$CSV" --workdir "$RES/run_json" -- "${CMD[@]}"
