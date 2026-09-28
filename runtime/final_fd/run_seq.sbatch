#!/bin/bash
#SBATCH --job-name=fdfin_c1
#SBATCH --partition=long
#SBATCH --exclude=g01,g02,g03
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/user/cafferychen777/fd_final/logs/%x_%j.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/logs/%x_%j.err
# C1 runtime rows for the final package. All tasks of tasks_seq.txt run one after
# another in this single job (one node, 32 CPUs, same c-node type / partition as the
# C1 CPU competitor runs). Uses the C1 monitor/common code and C1 data read-only.
# COLD=1 (sbatch --export=ALL,COLD=1): fresh numba cache directory per task, so every
# task compiles from scratch as in the original C1 protocol (one SLURM job per task,
# job-scoped TMPDIR); outputs go to *_coldcache files. Default: one shared cache per job.
set -uo pipefail
C1=/scratch/user/cafferychen777/FlashDeconv/validation/runtime_benchmark_c1
DATA=/scratch/user/cafferychen777/FlashDeconv/data/runtime_benchmark_c1
W=/scratch/user/cafferychen777/fd_final
B=$W/code/benchmarks/c1
RES=$W/results/c1
SFX=""; [[ ${COLD:-0} == 1 ]] && SFX=_coldcache
mkdir -p "$RES/props$SFX" "$RES/run_json$SFX" "$W/results/fitlogs"
NC=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_fdfin MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_fdfin
export PYTHONPATH=$W/code:$C1
export FD_FITLOG=$W/results/fitlogs/c1_fitlog$SFX.csv
echo "host=$(hostname) cpus=$NC job=$SLURM_JOB_ID"
while read -r METHOD MODE SCALE REP; do
  [[ -z ${METHOD:-} ]] && continue
  SEED=42; [[ $MODE == seed0 ]] && SEED=0
  export FD_TAG="c1_${MODE}_${SCALE}_r${REP}"
  [[ -n $SFX ]] && export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_fdfin_$FD_TAG
  "$W/env/bin/python" "$C1/monitor.py" --method "$METHOD" --mode "$MODE" --scale "$SCALE" --rep "$REP" \
    --device cpu --timeout-hours 4 --csv "$RES/c1_flashdeconv_final$SFX.csv" --workdir "$RES/run_json$SFX" -- \
    "$W/env/bin/python" "$B/run_flashdeconv_final.py" --data-dir "$DATA" --scale "$SCALE" --rep "$REP" \
    --props-dir "$RES/props$SFX" --random-state $SEED
done < "$B/tasks_seq.txt"
echo DONE
