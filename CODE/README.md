# TriGAC: paper-aligned implementation

This directory contains the current TriGAC method used for the manuscript's
**11 small multi-view datasets**. The main entry point is `run_trigac.py`.
The 11 dataset names and five independent pairwise-query caps are in
`paper_config.py`. The script does not run Animal, the four large real-world
datasets, or the separate million-object Synthetic experiment.

The current algorithm consists of the initialized object graphs, global
WCC-assignment object selection, direct pairwise annotations and object-graph
refinement, and view-graph-based marginal-contribution weighting. The default
setting is seed 42, `T=25`, `Omega=100`, PCA-50 and UMAP-20 where applicable.
Each cap starts an independent run; it is **not** a checkpoint from a longer
trajectory. `cons_cap` is the maximum number of pairwise queries, while
`actual_queries` records how many were used.

## Installation

Use Python 3.10 and install the dependencies in `requirements.txt`. For
example, from this directory:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
Place the 11 `.mat` files in `datasets/`, or pass `--data-dir PATH`.
The data loader accepts the same MATLAB multi-view format as the supplied
files. Do not present a one-seed result as a multi-seed average.
`Animal.mat` is not part of this 11-dataset release and is not distributed.

## Run

First check the dataset paths and budgets without running the algorithm:

```bash
python run_trigac.py --all --validate-only
```

Run one dataset or all 11:

```bash
python -u run_trigac.py --dataset Sources3
python -u run_trigac.py --all
```

Use `--budget 30` with `--dataset Sources3` for one cap. Options `--seed`,
`--t`, `--omega`, `--pca-dim`, `--umap-dim`, `--data-dir`, and `--output-root`
are available. Results are written to `results/` by default. Each raw CSV
contains the full round trajectory. `summary.csv` gives the final ARI, NMI,
cap and actual query count for every completed run. Restarting the same
command skips only trajectories whose input and code fingerprint matches.

Older research-workspace modules such as `ActiveSelecter.py`,
`HumanAnnotator.py`, `GeneralUtils.py`, and `NodeSkeletonRefiner.py` are
not required by this release or imported by the current entry point.

## Implementation files

- `run_trigac.py`: data preparation, five independent runs, validation,
  output and resume behavior.
- `paper_config.py`: the 11 datasets and their five query caps.
- `paper_engine.py` and `paper_core.py`: current selection, feedback,
  refinement and evaluation.
- `view_weighting.py`: current view grouping and marginal-contribution
  weighting.
- `main_realworld.py`, `NodeSkeletonBuilder.py`,
  `ViewSkeletonBuilder.py`, `WeightUpdator.py`: shared preprocessing and
  graph-construction components.

No paper table values are hard-coded as results; ARI and NMI are computed
from each run's predictions.
