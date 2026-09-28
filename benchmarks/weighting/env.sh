# Sourced inside weighting jobs only.
export FLASHDECONV_PROJECT_ROOT=/scratch/user/cafferychen777/FlashDeconv
export WT_WORK=/scratch/user/cafferychen777/fd_final/weighting
export FD_CODE=/scratch/user/cafferychen777/fd_final/code
export PY=/scratch/user/cafferychen777/fd_final/env/bin/python
NC=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_wt MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_wt
export PYTHONUNBUFFERED=1
cd $WT_WORK/code
