#!/bin/bash
# Global 38-type/lineage comparison + pathologist concordance for the FINAL P1 Visium HD
# proportions (no fitting; runs locally), then rebuild the Xenium claim tables.
set -eo pipefail
P=/Users/apple/Research/FlashDeconv
OUT=$P/results/rerun_final/crc/xenium/visium_hd_p1
mkdir -p $OUT
scp -q arseven:/scratch/user/cafferychen777/fd_final/crc/props/P1_CRC_final_8um.npz $OUT/_P1_CRC_final_8um.npz
cd $P/validation/rerun_final/crc/xenium
for step in global patho; do
  PYTHONPATH=$P/validation/rerun_final RERUN_OUT_DIR=$OUT $P/.venv/bin/python run_xenium_crc_validation_final.py \
      --step $step --props new --npz $OUT/_P1_CRC_final_8um.npz
done
rm -rf $OUT/props_final && mv $OUT/props_new $OUT/props_final
rm -f $OUT/_P1_CRC_final_8um.npz
$P/.venv/bin/python build_claims_final.py
