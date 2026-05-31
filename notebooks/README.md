# FlashDeconv Reproducibility Notebooks

These notebooks are narrative companions to the CLI scripts. The scripts remain
the canonical automation path; notebooks make intermediate checks, summary
tables, and figure previews easier to inspect.

## Runtime Modes

By default notebooks run in smoke mode:

```bash
pip install -r requirements-notebooks.txt
export FD_REPRO_MODE=smoke
python scripts/run_notebook_smoke.py
```

Smoke mode uses small synthetic data or already materialized summaries so the
notebook path can be validated quickly. Full mode uses the real downloaded data:

```bash
export FD_REPRO_MODE=full
export FD_DATA_DIR=./data
export FD_RESULTS_DIR=./results
```

Full mode expects the README data preparation steps to have completed. If input
files are missing, notebooks report the missing files and the command needed to
create them.

## Notebooks

- `01_visium_hd_resolution_and_tuft.ipynb`: Visium HD resolution horizon and
  tuft/stem niche checks.
- `02_cortex_lamination.ipynb`: Cell2location mouse brain deconvolution and
  cortical layer checks.
- `03_leverage_mechanism.ipynb`: leverage-score mechanism checks for Figure 2.

## Output Policy

Committed notebook outputs are intentionally small: configuration summaries,
compact tables, and lightweight plots. Generated datasets, CSV/NPZ results, and
publication figures should be written under `results/` or `figures/` and are not
tracked by git.
