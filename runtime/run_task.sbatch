#!/bin/bash
#SBATCH --job-name=c1
#SBATCH --output=/scratch/user/cafferychen777/FlashDeconv/logs/c1/%x_%A_%a.out
#SBATCH --error=/scratch/user/cafferychen777/FlashDeconv/logs/c1/%x_%A_%a.err
# Generic C1 array task. Resources (partition/cpus/mem/gres/time) are given at submission.
# TASKS (env) points to a whitespace-separated file: method mode scale rep  (line = array index + 1)
set -uo pipefail
BASE=/scratch/user/cafferychen777/FlashDeconv/validation/runtime_benchmark_c1
DATA=/scratch/user/cafferychen777/FlashDeconv/data/runtime_benchmark_c1
RES=/scratch/user/cafferychen777/FlashDeconv/results/runtime_benchmark_c1
V=/scratch/user/cafferychen777/venvs
RENV=/scratch/user/cafferychen777/conda_envs/c1_r
CSV=${C1_CSV:-$RES/c1_results.csv}
mkdir -p "$RES/props" "$RES/run_json"

read -r METHOD MODE SCALE REP < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
NC=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl
export PYTHONPATH=$BASE
echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD mode=$MODE scale=$SCALE rep=$REP host=$(hostname) cpus=$NC"

DEVICE=cpu
case $METHOD in
  flashdeconv) CMD=("$V/c1_fd/bin/python" "$BASE/run_flashdeconv.py" --data-dir "$DATA" --scale "$SCALE" --rep "$REP" --props-dir "$RES/props") ;;
  tacco)       CMD=("$V/c1_tacco/bin/python" "$BASE/run_tacco.py" --data-dir "$DATA" --scale "$SCALE" --props-dir "$RES/props") ;;
  cell2location) DEVICE=gpu
               CMD=("$V/c1_c2l/bin/python" "$BASE/run_c2l.py" --data-dir "$DATA" --scale "$SCALE" --mode "$MODE" --props-dir "$RES/props") ;;
  rctd)        export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1  # RCTD parallelizes over max_cores workers
               CMD=("$RENV/bin/Rscript" "$BASE/run_rctd.R" "$DATA" "$SCALE" "$MODE" "$RES/props" "$NC") ;;
  card)        CMD=("$RENV/bin/Rscript" "$BASE/run_card.R" "$DATA" "$SCALE" "$RES/props") ;;
  *) echo "unknown method $METHOD"; exit 2 ;;
esac
if [[ $METHOD == rctd || $METHOD == card ]]; then
  export R_LIBS_USER="$RENV/lib/R/library" R_LIBS="$RENV/lib/R/library" R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null
fi

"$V/c1_fd/bin/python" "$BASE/monitor.py" --method "$METHOD" --mode "$MODE" --scale "$SCALE" --rep "$REP" \
  --device "$DEVICE" --timeout-hours "${C1_TIMEOUT_H:-24}" --csv "$CSV" --workdir "$RES/run_json" -- "${CMD[@]}"
