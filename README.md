# Graph Structural Encodings: Operations, Budgets, and Readouts

Experiment code and supplementary rebuttal experiments. BREC and mechanism experiments target Python 3.12 on Linux. TU experiments use Co-Sparsify's loader, Lightning training, and evaluation pipeline, with the backbone replaced by GNNPlus GINE from `gnnplus_official`. Run all commands from this directory and specify your own data paths and output directories.

## Complete BREC campaign through reproduce.py

```bash
python -m pip install -r requirements/brec_suite.txt
python -m pip install 'torch>=2.7,<3' --index-url https://download.pytorch.org/whl/cpu
python reproduce.py brec-run --brec-file /data/BREC/brec_v3.npy --root /results/brec_exact_v2 --workers 44 --memory-gib 100
```

This entry point uses exact structural aggregation, complete relabeling controls, memory-bounded workers, frozen provenance, and verified resume. The existing `brec-plan` / `brec-run` / `brec-report` commands now serve this protocol. See [BREC commands and guarantees](docs/brec_suite.md) for the full experiment list and resource options. Use a new output root; historical frozen plans remain separate.

## BREC pairs distinguishable by 2-FWL and 3-FWL

The standard hierarchy identifies `k`-FWL with `(k + 1)`-WL. We therefore use the following naming consistently:

- **2-FWL = 3-WL**
- **3-FWL = 4-WL**

The official BREC paper reports the Basic, Regular, Extension, and CFI construction levels in `k`-WL terminology. Combining those labels with the official category ranges gives:

| Pair IDs | Category | 2-FWL (= 3-WL) | 3-FWL (= 4-WL) |
|---|---|---:|---:|
| 0-59 | Basic | Distinguishes | Distinguishes |
| 60-109 | Simple Regular | Distinguishes | Distinguishes |
| 110-159 | Strongly Regular | Indistinguishable | Distinguishes |
| 160-259 | Extension | Distinguishes | Distinguishes |
| 260-319 | CFI, 3-WL level | Distinguishes | Distinguishes |
| 320-339 | CFI, 4-WL-only level | Indistinguishable | Distinguishes |
| 340-359 | CFI, 4-WL-indistinguishable | Indistinguishable | Indistinguishable |
| 360-379 | 4-Vertex Condition | Indistinguishable | Distinguishes |
| 380-399 | Distance Regular | Indistinguishable | Distinguishes |

Compact ID sets:

- **2-FWL distinguishes 270 pairs:** `0-109, 160-319`.
- **3-FWL distinguishes 380 pairs:** `0-339, 360-399`.
- **2-FWL cannot distinguish 130 pairs:** `110-159, 320-399`.
- **3-FWL cannot distinguish 20 pairs:** `340-359`.

The additional 110 pairs gained by moving from 2-FWL to 3-FWL are:

- Strongly Regular: `110-159` (50 pairs).
- CFI requiring 4-WL: `320-339` (20 pairs).
- 4-Vertex Condition: `360-379` (20 pairs).
- Distance Regular: `380-399` (20 pairs).

Sources: the official BREC category ranges and CFI construction levels in [An Empirical Study of Realized GNN Expressiveness](https://arxiv.org/abs/2304.07702); the official `brec_v3_3wl.npy` selection gives the 2-FWL/3-WL set, while the paper identifies the 20 4-WL-only and 20 4-WL-indistinguishable CFI pairs.

These labels concern 2-FWL and 3-FWL, not the finite `Q` polynomial budget in `rebuttal_experiments/encodings.py`. An `unresolved` interval result means that the current pair encoding did not certify a decision; it does not by itself prove that 2-FWL or 3-FWL is unable to distinguish the pair.

## Running the supplementary experiments

See the [supplementary experiment guide](docs/rebuttal_experiments.md) for environments, inputs, and complete commands. Each command runs only the specified task:

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec
bash tools/run_rebuttal.sh triple-encodings
bash tools/run_rebuttal.sh budget-extend
bash tools/run_rebuttal.sh triple-ppgn
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
| `brec.py` | BREC configuration loops, relabeling controls, and result summaries |
| `scheduling.py` | BREC multiprocessing, CPU affinity, and total memory budget monitoring |
| `encodings.py`, `intervals.py`, `readouts.py` | Encoding formulas, interval arithmetic, and discrimination rules |
| `protein.py` | Calls to the original Co-Sparsify entry point, reuse of completed runs, and read-only result capture |
| `protein_encodings.py` | Replacement of only the original poly transform; 9 encodings at matching widths |
| `protein_gnnplus.py` | Input adaptation and model construction using the official GNNPlus GINE layers and SAN head |
| `protein_report.py` | Summaries of original ten-fold scores and paired differences at matching widths |
| `common.py` | File I/O, run configuration, and resource limits |

All short filenames in the table are under `rebuttal_experiments/`. Start with the experiment entry point, then read the corresponding computation functions. The TU adapter reuses the original training and final scoring pipeline.

Other reusable code is retained in `brec/`, `discriminability/`, `downstream/`, and `gnnplus_official/`. See [additional experiment entry points](docs/experiments.md) for broader datasets and PPGN commands, the [protocol guide](docs/protocols.md) for computational scope, and [THIRD_PARTY.md](THIRD_PARTY.md) for attribution.
