# Graph Structural Encodings: Operations, Budgets, and Readouts

Experiment code and supplementary rebuttal experiments. BREC and mechanism experiments target Python 3.12 on Linux. TU experiments use Co-Sparsify's loader, Lightning training, and evaluation pipeline, with the backbone replaced by GNNPlus GINE from `gnnplus_official`. Run all commands from this directory and specify your own data paths and output directories.

## BREC v3 float32 experiments

```bash
source ~/miniforge3/bin/activate gnn270
python reproduce.py brec --brec-file /data/BREC/brec_v3.npy \
  --out outputs/brec_float32 --workers 14 --devices 0 1
```

The 2-, 3-, and 4-FWL target cohorts contain 270, 110, and 18 pairs respectively. All comparisons use `float32` at the fixed threshold of \(10^{-8}\). See the [BREC run guide](docs/brec_suite.md) for the pair IDs, reliability checks, and CPU options.

## Running the supplementary experiments

See the [supplementary experiment guide](docs/rebuttal_experiments.md) for environments, inputs, and complete commands. Each command runs only the specified task:

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec
bash tools/run_rebuttal.sh objects

CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-both
```

Set `BREC_FILE`, `TU_DATA_ROOT`, and other paths as described in the guide before running. The guide also records the status of each experiment.
`protein-both` uses Co-Sparsify's TU data, ten-fold tasks, and trainer, replacing the model with the official GNNPlus GINE layers, SAN graph head, and mean pooling. Encodings enter through node loop channels and directed-edge pair channels. PROTEINS_full and ENZYMES each use 9 encodings × 4 widths × 10 folds, for 720 training runs in total. The same held-out fold is used for checkpoint selection and scoring. Results are saved in `outputs/rebuttal/cosparsify_tu`.

Use `python tools/run_protein_grid.py` to run the full matrix as background shards; add `--dry-run` first to inspect the task plan. The runner supports scheduling across 4 GPUs, memory headroom, resuming, identity and checksum validation across shards, and a final merged report.

## Reading the code

| File | Responsibility |
|---|---|
| `rebuttal_experiments/__main__.py` | Experiment dispatch |
| `mechanisms.py` | Two exact mechanism controls |
| `encodings.py`, `intervals.py`, `readouts.py` | Encoding formulas, interval arithmetic, and discrimination rules |
| `protein.py` | Calls to the original Co-Sparsify entry point, reuse of completed runs, and read-only result capture |
| `protein_encodings.py` | Replacement of only the original poly transform; 9 encodings at matching widths |
| `protein_gnnplus.py` | Input adaptation and model construction using the official GNNPlus GINE layers and SAN head |
| `protein_report.py` | Summaries of original ten-fold scores and paired differences at matching widths |
| `common.py` | File I/O, run configuration, and resource limits |

All short filenames in the table are under `rebuttal_experiments/`. Start with the experiment entry point, then read the corresponding computation functions. The TU adapter reuses the original training and final scoring pipeline.

Other reusable code is retained in `brec/`, `discriminability/`, `downstream/`, and `gnnplus_official/`. See [additional experiment entry points](docs/experiments.md) for broader datasets, the [protocol guide](docs/protocols.md) for computational scope, and [THIRD_PARTY.md](THIRD_PARTY.md) for attribution.
