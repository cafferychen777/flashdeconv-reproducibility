#!/bin/bash
#SBATCH --job-name=c1_c2l
#SBATCH --account=<ACES_ACCOUNT>  # allocation account removed for the public repository
#SBATCH --ntasks=1
#SBATCH --output=/scratch/user/u.cy359721/fd_c2l/logs/%x_%A_%a.out
# ACES port of FlashDeconv validation/runtime_benchmark_c1/run_task.sbatch (cell2location only).
# TASKS (env): "method mode scale rep" per line (line = array index + 1).
set -uo pipefail
B=/scratch/user/u.cy359721/fd_c2l
BASE=$B/code/c1
DATA=$B/data/runtime_benchmark_c1
RES=$B/results/c1
V=/scratch/group/<ACES_GROUP>/u.cy359721_fd_c2l/venvs/c2l
CSV=${C2L_CSV:-$RES/c1_c2l_aces.csv}
mkdir -p "$RES/props" "$RES/run_json"
read -r METHOD MODE SCALE REP < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
NC=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl XDG_CACHE_HOME=/scratch/group/<ACES_GROUP>/u.cy359721_fd_c2l/cache
export PYTHONPATH=$BASE
echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD mode=$MODE scale=$SCALE rep=$REP host=$(hostname) cpus=$NC"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$V/bin/python" "$BASE/monitor.py" --method "$METHOD" --mode "$MODE" --scale "$SCALE" --rep "$REP" \
  --device gpu --timeout-hours "${C1_TIMEOUT_H:-24}" --csv "$CSV" --workdir "$RES/run_json" -- \
  "$V/bin/python" "$BASE/run_c2l.py" --data-dir "$DATA" --scale "$SCALE" --mode "$MODE" --props-dir "$RES/props"
