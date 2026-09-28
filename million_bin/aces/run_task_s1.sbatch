#!/bin/bash
#SBATCH --job-name=s1_c2l
#SBATCH --account=<ACES_ACCOUNT>  # allocation account removed for the public repository
#SBATCH --ntasks=1
#SBATCH --output=/scratch/user/u.cy359721/fd_c2l/logs/%x_%A_%a.out
# ACES port of FlashDeconv validation/s1_merfish_benchmark/run_task.sbatch (cell2location only).
# TASKS (env): "method mode ref stfile" per line (line = array index + 1).
set -uo pipefail
B=/scratch/user/u.cy359721/fd_c2l
BASE=$B/code/s1
DATA=$B/data/s1_merfish
RES=$B/results/s1
V=/scratch/group/<ACES_GROUP>/u.cy359721_fd_c2l/venvs/c2l
CSV=${C2L_CSV:-$RES/s1_runtime_c2l_aces.csv}
mkdir -p "$RES/props" "$RES/run_json"
read -r METHOD MODE REF STF < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
ST=$DATA/st/$STF
SCALE=$(echo "$STF" | sed -E 's/^st_([0-9]+)\.h5ad$/\1/')
REFDIR=$DATA/$REF
PROPS=$RES/props/${METHOD}_${MODE}_${REF}_${SCALE}.csv.gz
NC=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl XDG_CACHE_HOME=/scratch/group/<ACES_GROUP>/u.cy359721_fd_c2l/cache
export PYTHONPATH=$BASE
echo "task=$SLURM_ARRAY_TASK_ID method=$METHOD mode=$MODE ref=$REF st=$STF host=$(hostname) cpus=$NC"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [[ $MODE == ref_train ]]; then SCALE=0; CMD=("$V/bin/python" "$BASE/run_c2l.py" --ref-dir "$REFDIR" --mode ref_train)
else CMD=("$V/bin/python" "$BASE/run_c2l.py" --ref-dir "$REFDIR" --st "$ST" --mode "$MODE" --props "$PROPS"); fi
"$V/bin/python" "$BASE/monitor.py" --method "$METHOD" --mode "${REF}-${MODE}" --scale "$SCALE" --rep 1 \
  --device gpu --timeout-hours "${S1_TIMEOUT_H:-24}" --csv "$CSV" --workdir "$RES/run_json" -- "${CMD[@]}"
