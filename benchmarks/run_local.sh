#!/bin/bash
# Run jobs_local/job*.sh with at most 5 concurrent single-threaded processes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
ls jobs_local/job*.sh | xargs -P 5 -n 1 bash
