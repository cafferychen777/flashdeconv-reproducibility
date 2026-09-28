#!/bin/bash
#SBATCH --job-name=crc_final_time
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/scratch/user/cafferychen777/fd_final/crc/logs/timing_%j.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/crc/logs/timing_%j.err
# Sequential timing: P1, P2, P5 one after another in ONE job on ONE node (32 CPUs).
set -eo pipefail
NC=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_crcft MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_crcft PYTHONUNBUFFERED=1
hostname; lscpu | grep -E "Model name|^CPU\(s\)"
cd /scratch/user/cafferychen777/fd_final/crc/code
R=/scratch/user/cafferychen777/fd_final/crc/results/timing_final.csv
[ -f $R ] && mv $R ${R%.csv}_run_concurrent_${SLURM_JOB_ID}.csv
JOB_T0=$(date +%s)
for S in P1_CRC P2_CRC P5_CRC; do
  FD_FITLOG=/scratch/user/cafferychen777/fd_final/crc/fitlogs/crc_timing.csv FD_TAG=crc_timing_$S \
    /scratch/user/cafferychen777/fd_final/env/bin/python timing_final.py $S $R
done
echo "job wall seconds: $(( $(date +%s) - JOB_T0 ))"
echo "host=$(hostname) cpus=$NC model=$(lscpu | grep 'Model name' | sed 's/.*: *//')" > ${R%.csv}_host.txt
