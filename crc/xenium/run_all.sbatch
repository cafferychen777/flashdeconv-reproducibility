#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/cafferychen777/fd_final/crc/xenium/logs/%x_%j.out
#SBATCH --error=/scratch/user/cafferychen777/fd_final/crc/xenium/logs/%x_%j.err
# Final-package rerun of the CRC Xenium validations and the Visium HD demo ablations.
# Usage: sbatch --job-name=<xf_binning|xf_pvhd|xf_lambda|xf_lap|xf_zoom> run_all.sbatch
set -eo pipefail
NC=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NC MKL_NUM_THREADS=$NC OPENBLAS_NUM_THREADS=$NC NUMBA_NUM_THREADS=$NC
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_xfinal MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_xfinal
export PYTHONUNBUFFERED=1
lscpu | grep -E "Model name|^CPU\(s\)" || true
PY=/scratch/user/cafferychen777/fd_final/env/bin/python
W=/scratch/user/cafferychen777/fd_final/crc/xenium
DEMO=/scratch/user/cafferychen777/fd_final/crc/demo
V020=/scratch/user/cafferychen777/fd_v020_crc
export FLASHDECONV_PROJECT_ROOT=/scratch/user/cafferychen777/FlashDeconv
export FD_FITLOG=/scratch/user/cafferychen777/fd_final/crc/fitlogs/crc_xenium_${SLURM_JOB_NAME#xf_}.csv
export FD_TAG=$SLURM_JOB_NAME
mkdir -p /scratch/user/cafferychen777/fd_final/crc/fitlogs
case "$SLURM_JOB_NAME" in
  xf_binning)
    cd $W/code
    RERUN_OUT_DIR=$W/results/virtual_binning $PY run_xenium_crc_validation_final.py --step binning ;;
  xf_pvhd)
    cd $W/code
    $PY xenium_pseudo_visiumhd_benchmark.py --bin-sizes 4 8 16 32 --downsample-vhd \
        --ref-mode xenium --seeds 1 --output-dir $W/results/pseudo_vhd
    FD_TAG=xf_pvhd_determinism $PY check_determinism.py ;;
  xf_lambda)
    cd $W/code
    RERUN_OUT_DIR=$W/results/lambda_ablation XENIUM_IO_DATA_DIR=$FLASHDECONV_PROJECT_ROOT/analysis/xenium_crc_io_data \
    GSE132465_REF=$V020/xenium/data/GSE132465_CRC_reference.h5ad $PY xenium_crc_lambda_ablation.py ;;
  xf_lap)
    $PY $DEMO/code/laplacian_ablation_visiumhd_final.py $V020/demo/data $DEMO/results ;;
  xf_zoom)
    $PY $DEMO/code/zoom_predictions_final.py $V020/demo/data $DEMO/results ;;
esac
echo "end $(date)"
