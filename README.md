# FlashDeconv manuscript scripts

This repository contains the analysis and figure scripts used for the FlashDeconv manuscript. All results were produced with the FlashDeconv package version 0.2.0 ([github.com/cafferychen777/flashdeconv](https://github.com/cafferychen777/flashdeconv); Zenodo [10.5281/zenodo.23006113](https://doi.org/10.5281/zenodo.23006113)), and the scripts require FlashDeconv 0.2.0. The data sources are listed in the Data availability section of the manuscript. No data or results are included. The scripts are kept as they were run; file paths refer to the authors' local and cluster (SLURM) directories.

## Layout

| Folder | Contents |
|---|---|
| `common/` | `fdfinal.py`: fit-log hook imported by every final-run script |
| `benchmarks/spotless/` | Spotless silver and gold standards, liver and melanoma case studies, competitor rescoring, data conversion (`data_prep/`) |
| `benchmarks/li2023/` | Li et al. (2023) MERFISH and seqFISH+ benchmark |
| `benchmarks/pseudo_visium_hd/` | Xenium-derived pseudo-Visium HD bins, RCTD, NNLS and marker scoring (`final_fd/`: FlashDeconv 0.2.0 runs and rescoring) |
| `benchmarks/weighting/` | Gene-weighting ablation (leverage, equal, variance) |
| `benchmarks/laplacian/`, `benchmarks/marker_scoring/` | Spatial-penalty ablation, marker-scoring comparison |
| `benchmarks/penalty_sparsity/` | Spatial penalty against bin depth (million-bin, pseudo-Visium HD and gold-standard data) |
| `benchmarks/` (top level) | Local job list (`run_local.sh`, `jobs_local/`), seed checks (`seedpatch.py`, `determinism.py`), cluster summary tables |
| `runtime/` | Runtime and memory benchmark on pooled CRC Visium HD bins (`final_fd/`: FlashDeconv runs; `aces/`: Cell2location GPU runs) |
| `million_bin/` | Million-bin MERFISH-derived mouse gut benchmark |
| `reference_diagnostic/` | `final/`: intestine and Spotless removal runs; `calibration/`: empirical-null calibration and CRC scoring; `crc_regions/`: CRC flagged regions and hallmark enrichment; `composite_reference/`: composite mouse small intestine reference |
| `crc/` | Visium HD CRC cohort analysis (`xenium/`: serial-section Xenium validation and pseudo-Visium HD bins; `demo/`: Visium HD spatial-penalty ablation; `cross_cohort/`: Marteau, Pelka, TCGA and Schürch cohorts) |
| `interface_atlas/` | Nine-section tumour–stroma interface atlas and lung reference comparison |
| `spatial_controls/` | Refits without the spatial penalty (λ = 0: Spotless, CRC cohort, interface atlas), Spotless layout control, gold standards on measured coordinates |
| `figures/`, `figures/supp/` | Main and Supplementary figures, Supplementary tables and Supplementary Data 1 |

Figure 1 is a schematic that was drawn separately, so this repository has no script for it.

## Figures, tables and analyses

| Item | Analysis scripts | Figure/table script |
|---|---|---|
| **Fig. 2a**, Supp. Fig. S1, Supp. Table S2 (gene weighting) | `benchmarks/weighting/prep_spotless_local.py`, `run_weighting_final.py`, `analyze_weighting_final.py`; Spotless at λ = 0: `spatial_controls/weighting_spotless_lam0.py` | `figures/fig2_accuracy.py`, `figures/supp/supp_weighting.py`, `make_supp_tables.py` |
| **Fig. 2b** (Xenium bins, weighting) | `benchmarks/pseudo_visium_hd/prepare_bins.py`, `benchmarks/weighting/run_weighting_final.py xenium` | `figures/fig2_accuracy.py` |
| **Fig. 2c,d,h**, Supp. Fig. S2, Supp. Tables S3–S6 (Spotless) | `benchmarks/spotless/run_silver_gold.py` (silver standards: config `final_default_lam0`), `gold_competitors.py`, `run_liver.py`, `run_melanoma.py`, `analyze_silver.py`, `analyze_cases.py` (inputs: `data_prep/*.R`, `prep_gold.py`, `precompute_signature.py`, `export_invivo_signature.R`, `extract_*spotless_metrics.R`); `spatial_controls/editor_lam0_summary.py`, `c1_spotless_layout.py`, `c1_summary.py` | `figures/fig2_accuracy.py`, `figures/supp/supp_spotless.py`, `make_supp_tables.py` |
| **Fig. 2e**, Supp. Table S7 (Li et al.) | `benchmarks/li2023/run_li2023.py` (+ `li2023_*.py/.R`) | `figures/fig2_accuracy.py`, `make_supp_tables.py` |
| **Fig. 2f,g**, Supp. Fig. S3, Supp. Tables S9–S11 (pseudo-Visium HD) | `benchmarks/pseudo_visium_hd/` (`prepare_bins.py`, `run_py_methods.py`, `run_rctd.R`, `evaluate.py`), `final_fd/run_fd_final.py`, `final_fd/rescore_standard.py`, `benchmarks/build_part_summary_arseven.py`, `make_c2_supp_table.py` | `figures/fig2_accuracy.py`, `figures/supp/supp_pseudo_vhd.py`, `make_supp_tables.py` |
| **Fig. 3a–f**, Supp. Fig. S4 (million-bin benchmark) | `million_bin/s01`–`s04*.py`, `s06_spatial_window.py`, `run_*.py/.R`, `run_task.sbatch`, `aces/` | `figures/fig3_scale.py`, `figures/supp/supp_s1_external_reference.py` |
| **Fig. 3g–i**, Supp. Table S12 (runtime and memory) | `runtime/prep_data.py`, `run_*.py/.R`, `monitor.py`, `run_task.sbatch`, `final_fd/`, `aces/` | `figures/fig3_scale.py`, `make_supp_tables.py` |
| Supp. Table S8 (marker scoring) | `benchmarks/marker_scoring/compare_marker_scoring_rerun.py`, `spatial_controls/marker_scoring_lam0.py` | `make_supp_tables.py` |
| **Fig. 4a–f** (intestine, controlled removal) | `reference_diagnostic/composite_reference/a1_build_reference.py`, `reference_diagnostic/final/intestine.py`, `spotless.py`, `figure.py`; Spotless at λ = 0: `spatial_controls/refdiag_spotless_lam0.py`, `refdiag_spotless_lam0_extra.py` | `figures/fig4_refcheck.py` |
| **Fig. 4g,h**, Supp. Fig. S5 (calibration) | `reference_diagnostic/calibration/` (`pkg_local.py`, `pkg_crc_runtime.py`, `diagnose.py`, `summarize.py`, `figure.py`), `reference_diagnostic/final/naming_nulls.py` | `figures/fig4_refcheck.py`, `figures/supp/supp_refdiag.py` |
| **Fig. 4i**, Supp. Fig. S5 (lung) | `interface_atlas/run_lung_refs.py` | `figures/fig4_refcheck.py`, `figures/supp/supp_refdiag.py` |
| **Fig. 5a,c–g**, Supp. Table S13 (CRC cohort) | `crc/run_final.py`, `crc_common.py`, `merge_final.py`, `rctd_label1_fix.py` (RCTD singlet label, hotspot RCTD classes, lineage agreement), `annotation_correction.py`, `stage_figdata_final.py`, `timing_final.py`, `part_summary.py` | `figures/fig5_crc.py`, `make_supp_tables.py` |
| **Fig. 5b**, Supp. Fig. S5 (CRC) | `reference_diagnostic/calibration/diagnose.py crc`, `eval_crc.py`, `reference_diagnostic/crc_regions/b2_stage1.py`, `b2_stage2.py`, `b2_hallmark.py` | `figures/fig5_crc.py`, `figures/supp/supp_refdiag.py` |
| **Fig. 5h,i**, Supp. Fig. S6 (Xenium validation) | `crc/xenium/run_xenium_crc_validation_final.py`, `xenium_crc_validation.py`, `xenium_pseudo_visiumhd_benchmark.py`, `build_claims_final.py`, `pathologist_p2.py`; per-type average precision: `benchmarks/pseudo_visium_hd/final_fd/per_type_ap.py` | `figures/fig5_crc.py`, `figures/supp/supp_crc_validation.py` |
| Supp. Fig. S6, Supp. Note CRC independent cohorts | `crc/cross_cohort/00_download_data.sh`, `01_marteau_xenium_neutrophil_mregdc.py`, `02_pelka_scrna_cooccurrence.py`, `04_tcga_survival.py`, `05_schurch_codex_neighborhood.py` | `figures/supp/supp_crc_validation.py` |
| **Fig. 6**, Supp. Fig. S7, Supp. Table S14 (interface atlas) | `interface_atlas/run_fd.py`, `prep_orthogonal.py`, `gradients.py`, `posthoc.py`, `posthoc_v2.py`, `lineages.py` | `figures/fig6_interface.py`, `figures/supp/supp_interface.py`, `make_supp_tables.py` |
| Supp. Table S15 (CRC and interface atlas at λ = 0) | `spatial_controls/c2_crc_lam0.py`, `c2_crc_merge.py`, `c2_b1_lam0.py`, `c2_b1_summary.py` | `make_supp_tables.py` |
| Supp. Fig. S8 (spatial regularization) | `spatial_controls/gold_realxy.py`, `crc/xenium/xenium_crc_lambda_ablation.py`, `crc/demo/laplacian_ablation_visiumhd_final.py` | `figures/supp/supp_laplacian.py` |
| Supp. Fig. S9 (spatial penalty and bin depth) | `benchmarks/penalty_sparsity/s1_run.py`, `c2_run.py`, `gold_run.py`, `ps_common.py`, `summarize.py` | `benchmarks/penalty_sparsity/make_figure.py` |
| Determinism (identical output for any seed) | `benchmarks/jobs_local/job06,08,10.sh`, `benchmarks/determinism.py`, `crc/xenium/check_determinism.py`, `crc/merge_final.py` | — |
| Supplementary Data 1 | outputs of the benchmark scripts above | `figures/supp/make_supplementary_data_1.py` |

Supplementary Table S1 (computational cost) has no script.

`figures/style.py` holds the shared plotting style. The main-text numbers are read from the result tables that the analysis scripts in the same row write.
