# Sourced by C2 sbatch scripts (inside jobs only).
export PROJ=/scratch/user/cafferychen777/FlashDeconv
export C2_OUT=$PROJ/results/pseudo_vhd_c2
export FLASHDECONV_PROJECT_ROOT=$PROJ
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMBA_NUM_THREADS=$OMP_NUM_THREADS
c2_py() { module load Python/3.11.5-GCCcore-13.2.0; source /scratch/user/cafferychen777/envs/pvhd_c2_py/bin/activate; }
# R: avoid R 4.4 libs from ~/.Renviron; single-threaded BLAS per spacexr worker (workers = cores).
c2_r() { module load R/4.3.2-gfbf-2023a; export R_ENVIRON_USER=$PROJ/validation/pseudo_vhd_c2/Renviron_r43; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1; }
cd $PROJ/validation/pseudo_vhd_c2
