# FlashDeconv manuscript scripts

This repository contains the analysis and figure scripts used for the FlashDeconv manuscript. All results were produced with the FlashDeconv package version 0.2.0 ([github.com/cafferychen777/flashdeconv](https://github.com/cafferychen777/flashdeconv); Zenodo [10.5281/zenodo.23006113](https://doi.org/10.5281/zenodo.23006113)), and the scripts require FlashDeconv 0.2.0. The data sources are listed in the Data availability section of the manuscript. No data or results are included. The scripts are kept as they were run, so they still contain the absolute local and cluster (SLURM) paths of the original project. In those paths, `validation/<dir>` refers to the source folders listed under Layout below.

## Layout

| Folder | Contents | Original location |
|---|---|---|
| `common/` | `fdfinal.py`: fit-log hook imported by every final-run script | `validation/rerun_final/` |
| `benchmarks/spotless/` | Spotless silver and gold standards, liver and melanoma case studies, competitor rescoring, data conversion (`data_prep/`) | `validation/rerun_final/benchmarks/spotless/`, `validation/` |
| `benchmarks/li2023/` | Li et al. (2023) MERFISH and seqFISH+ benchmark | `validation/rerun_final/benchmarks/li2023/`, `validation/external_benchmarks/` |
| `benchmarks/pseudo_visium_hd/` | Xenium-derived pseudo-Visium HD bins, RCTD, NNLS and marker scoring (`final_fd/`: FlashDeconv 0.2.0 runs and rescoring) | `validation/pseudo_vhd_c2/`, `validation/rerun_final/benchmarks/c2/` |
| `benchmarks/weighting/` | Gene-weighting ablation (leverage, equal, variance) | `validation/rerun_final/weighting/` |
| `benchmarks/laplacian/`, `benchmarks/marker_scoring/` | Spatial-penalty ablation, marker-scoring comparison | `validation/rerun_final/benchmarks/` |
| `benchmarks/` (top level) | Local job list (`run_local.sh`, `jobs_local/`), seed checks (`seedpatch.py`, `determinism.py`), cluster summary tables | `validation/rerun_final/benchmarks/` |
| `runtime/` | Runtime and memory benchmark on pooled CRC Visium HD bins (`final_fd/`: FlashDeconv runs; `aces/`: Cell2location GPU runs) | `validation/runtime_benchmark_c1/`, `validation/rerun_final/benchmarks/c1/` |
| `million_bin/` | Million-bin MERFISH-derived mouse gut benchmark | `validation/s1_merfish_benchmark/` |
| `reference_diagnostic/` | `final/`: intestine and Spotless removal runs; `calibration/`: empirical-null calibration and CRC scoring; `crc_regions/`: CRC flagged regions and hallmark enrichment; `composite_reference/`: composite mouse small intestine reference | `validation/rerun_final/refdiag/`, `validation/reference_diagnostic_v2/`, `validation/b2_crc_refcheck/`, `validation/intestine_reference_v2/` |
| `crc/` | Visium HD CRC cohort analysis (`xenium/`: serial-section Xenium validation and pseudo-Visium HD bins; `demo/`: Visium HD spatial-penalty ablation; `cross_cohort/`: Marteau, Pelka, TCGA and Schürch cohorts) | `validation/rerun_final/crc/`, `analysis/cross_cohort_evidence/` |
| `interface_atlas/` | Nine-section tumour–stroma interface atlas and lung reference comparison | `validation/b1_pilot/` |
| `figures/`, `figures/supp/` | Main and Supplementary figures, Supplementary tables and Supplementary Data 1 | `validation/figures/` |

Figure 1 is a schematic that was drawn separately, so this repository has no script for it.

## Figures, tables and analyses

| Item | Analysis scripts | Figure/table script |
|---|---|---|
| **Fig. 2a**, Supp. Fig. weighting, Supp. Table weighting | `benchmarks/weighting/prep_spotless_local.py`, `run_weighting_final.py`, `analyze_weighting_final.py` | `figures/fig2_accuracy.py`, `figures/supp/supp_weighting.py`, `make_supp_tables.py` |
| **Fig. 2b** (Xenium bins, weighting) | `benchmarks/pseudo_visium_hd/prepare_bins.py`, `benchmarks/weighting/run_weighting_final.py xenium` | `figures/fig2_accuracy.py` |
| **Fig. 2c,d,h**, Supp. Fig. Spotless, Supp. Tables Spotless/gold/liver–melanoma/unified | `benchmarks/spotless/run_silver_gold.py`, `gold_competitors.py`, `run_liver.py`, `run_melanoma.py`, `analyze_silver.py`, `analyze_cases.py` (inputs: `data_prep/*.R`, `prep_gold.py`, `precompute_signature.py`, `export_invivo_signature.R`, `extract_*spotless_metrics.R`) | `figures/fig2_accuracy.py`, `figures/supp/supp_spotless.py`, `make_supp_tables.py` |
| Supp. Table liver collinearity | — | `figures/supp/liver_collinearity.py` |
| **Fig. 2e**, Supp. Note/Table Li et al. | `benchmarks/li2023/run_li2023.py` (+ `li2023_*.py/.R`) | `figures/fig2_accuracy.py`, `make_supp_tables.py` |
| **Fig. 2f,g**, Supp. Fig. and Tables pseudo-Visium HD | `benchmarks/pseudo_visium_hd/` (`prepare_bins.py`, `run_py_methods.py`, `run_rctd.R`, `evaluate.py`), `final_fd/run_fd_final.py`, `final_fd/rescore_standard.py`, `benchmarks/build_part_summary_arseven.py` | `figures/fig2_accuracy.py`, `figures/supp/supp_pseudo_vhd.py`, `make_supp_tables.py` |
| Supp. Fig. and Table pseudo-Visium HD (`c2_supp_table.csv`) | `benchmarks/build_part_summary_arseven.py` (writes `c2_final_summary_long.csv`) | `benchmarks/pseudo_visium_hd/make_c2_supp_table.py` |
| **Fig. 3a**, Supp. Note million-bin benchmark | `million_bin/s01`–`s04*.py`, `run_*.py/.R`, `run_task.sbatch`, `aces/` | `million_bin/s05_figure.py` (draft) |
| **Fig. 3b–d**, Supp. Note/Table runtime | `runtime/prep_data.py`, `run_*.py/.R`, `monitor.py`, `run_task.sbatch`, `final_fd/`, `aces/` | `make_supp_tables.py` |
| Supp. Fig. spatial regularization | `benchmarks/laplacian/ablation_laplacian_rerun.py`, `crc/xenium/xenium_crc_lambda_ablation.py`, `crc/demo/laplacian_ablation_visiumhd_final.py` | `figures/supp/supp_laplacian.py` |
| Supp. Note/Table marker scoring | `benchmarks/marker_scoring/compare_marker_scoring_rerun.py` | `make_supp_tables.py` |
| **Fig. 4a–f** (intestine, controlled removal) | `reference_diagnostic/composite_reference/a1_build_reference.py`, `reference_diagnostic/final/intestine.py`, `spotless.py`, `figure.py` | `figures/fig4_refcheck.py` |
| **Fig. 4g,h**, Supp. Fig. reference diagnostic a–e | `reference_diagnostic/calibration/` (`pkg_local.py`, `pkg_crc_runtime.py`, `diagnose.py`, `summarize.py`, `figure.py`), `reference_diagnostic/final/naming_nulls.py` | `figures/fig4_refcheck.py`, `figures/supp/supp_refdiag.py` |
| **Fig. 4i**, Supp. Fig. reference diagnostic g,h (lung) | `interface_atlas/run_lung_refs.py` | `figures/fig4_refcheck.py`, `figures/supp/supp_refdiag.py` |
| **Fig. 5a,c–g**, Supp. Note/Table CRC | `crc/run_final.py`, `crc_common.py`, `merge_final.py`, `annotation_correction.py`, `stage_figdata_final.py`, `timing_final.py`, `part_summary.py` | `figures/fig5_crc.py`, `make_supp_tables.py` |
| **Fig. 5b**, Supp. Fig. reference diagnostic i,j | `reference_diagnostic/calibration/diagnose.py crc`, `eval_crc.py`, `reference_diagnostic/crc_regions/b2_stage1.py`, `b2_stage2.py`, `b2_hallmark.py` | `figures/fig5_crc.py`, `figures/supp/supp_refdiag.py` |
| **Fig. 5h,i**, Supp. Fig. CRC validation a–e,h,i | `crc/xenium/run_xenium_crc_validation_final.py`, `xenium_crc_validation.py`, `xenium_pseudo_visiumhd_benchmark.py`, `build_claims_final.py`, `pathologist_p2.py` | `figures/fig5_crc.py`, `figures/supp/supp_crc_validation.py` |
| Supp. Fig. CRC validation f,g, Supp. Note CRC independent cohorts | `crc/cross_cohort/00_download_data.sh`, `01_marteau_xenium_neutrophil_mregdc.py`, `02_pelka_scrna_cooccurrence.py`, `04_tcga_survival.py`, `05_schurch_codex_neighborhood.py` | `figures/supp/supp_crc_validation.py` |
| **Fig. 6**, Supp. Fig. interface, Supp. Table interface | `interface_atlas/run_fd.py`, `prep_orthogonal.py`, `gradients.py`, `posthoc.py`, `posthoc_v2.py`, `lineages.py` | `figures/fig6_interface.py`, `figures/supp/supp_interface.py`, `make_supp_tables.py` |
| Determinism (identical output for any seed) | `benchmarks/jobs_local/job06,08,10.sh`, `benchmarks/determinism.py`, `crc/xenium/check_determinism.py`, `crc/merge_final.py` | — |
| Supplementary Data 1 | outputs of the benchmark scripts above | `figures/supp/make_supplementary_data_1.py` |

`figures/style.py` holds the shared plotting style. The main-text numbers are read from the result tables that the analysis scripts in the same row write.
