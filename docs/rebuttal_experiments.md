# Supplementary rebuttal experiments: server run guide

BREC and mechanism experiments use protocol `rebuttal-2026-11-12-v2`; TU uses `gnnplus-gine-tu-structural-edge-v1`. The code is organized around experiment computations, execution workflows, and result summaries. Previous experiment results are not bundled. `mechanisms` has completed on the server: the exact base/joint multiset comparisons and the third-moment difference of 1/331776 under a change of basis agree with theory. Full BREC and protein prediction runs remain pending.

## Environments and inputs

Run BREC and exact mechanism experiments on Linux with Python 3.12, from this repository's root:

```bash
python -m venv .venv-rebuttal
source .venv-rebuttal/bin/activate
python -m pip install -r requirements/rebuttal.txt

export REBUTTAL_OUT=outputs/rebuttal
export BREC_FILE=/server/data/BREC/brec_v3.npy
```

Replace the input paths with their actual locations. BREC uses the official array of 51,200 graph6 entries, with its SHA-256 pinned in the configuration. TU experiments use the original Co-Sparsify environment and PyG loader described below. Install and run them in that environment, separately from BREC; missing data is handled by the original loader's download behavior.

Each experiment writes to a new directory by default and refuses to overwrite existing output. `--resume` checks the data, configuration, core computation code, and environment while preserving completed records. Removing unused supplementary modules or reorganizing entry points alone does not require recomputation. Original record identities are preserved; the current source inventory is recorded separately in the run manifest.

## Experiments and commands

| Experiment | Command | Question addressed |
|---|---|---|
| 1. joint and mean moments | `mechanisms` | Whether raw-feature discrimination and readout discrimination differ for the rook/Shrikhande graphs |
| 2. Change of basis and readouts | The same `mechanisms` command | Whether an information-preserving change of basis can improve mean-moment discrimination at Q=1, P=3 |
| 3. BREC | `brec` | Certifying pair-encoding differences with exact/interval computations while retaining unresolved cases |
| 4. Protein prediction | `protein-run/report` | Whether joint changes prediction performance at matching input widths and training budgets |

Run the experiments you need individually; these commands do not launch all tasks automatically:

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec
```

## BREC computations and results

`configs/rebuttal/brec.json` fixes two bases, polynomial/endpoint encodings, Q=2/4/8/16, and full-multiset or cumulative mean-moment readouts with P=1/2/4, for 64 pair-domain settings. Exact fractions are used when both graphs are regular; other cases use 128→256→512-bit intervals generated from integer adjacency matrices. The interval backend uses arbitrary-precision integer endpoints, with no Arb dependency and no wrapping of already-rounded floating-point features.

Mean-moment discrimination is reported only when a difference interval excludes 0. Full-multiset differences are certified by projection counts or a compatible-matching deficit. Overlapping intervals or a compatible perfect matching do not prove equality and are recorded as unresolved. The standalone default configuration uses 8 fixed relabelings per side, the first official reliability pair, and 2 further permutations of its first graph. Graph isomorphism aligns known isomorphic controls.

With `seconds_per_condition: null`, readouts for the same pair, basis, and variant share feature matrices up to the largest Q still pending at each precision. Lower Q settings select their columns by channel descriptor in the original order. The mean readouts take the requested channel subset and moment prefix from the shared accumulation. Conditions that stop at the same precision share the relabeling and reliability checks on their combined channels. A finite wall limit schedules each condition independently so each starts with its own time budget.

By default, workers run in parallel according to the available CPU count, with single-threaded numerical libraries in each process. CPU affinity reuses the physical-core-first ordering in `brec/schedule.py`. The `memory_gib: 52` setting is the total RSS budget for the scheduler and all workers. The scheduler admits tasks based on estimated memory, monitors actual RSS, and reserves 8 GiB of available system memory by default. Shared pages may be counted more than once in RSS, making this accounting conservative. Under memory pressure, a grouped task may split into independent conditions. When the scheduler stops an individual condition for exceeding its RAM policy, it records `resource_limited / scheduler_memory_limit`. Results are saved atomically per condition, interruptions clean up workers, and `--resume` skips existing records and continues unfinished conditions.

There is no per-condition time limit (`seconds_per_condition: null`); the compatible-matching edge limit remains 2 million. To impose a time limit, set `seconds_per_condition` to a positive number of seconds. Fix resource settings according to the server's capacity before producing formal results. Use a new output directory after changing the code, configuration, or concurrency settings. Processes already running continue to use the settings read at startup.

For example, limit execution to 28 workers while retaining the total memory budget of 52 GiB:

```bash
export REBUTTAL_OUT=outputs/rebuttal_parallel
bash tools/run_rebuttal.sh brec --workers 28
# Resume with the same code, configuration, and concurrency settings:
bash tools/run_rebuttal.sh brec --workers 28 --resume
```

Alternatively, set `BREC_WORKERS=28` for either `brec` or `brec-reproduce`. Use `--reserve-gib` to adjust the amount of available system memory reserved. Outputs are:

- `records/*.json`: settings, discrimination decisions, precision attempts, relabeling results, and numerical `evidence` summaries. Summaries contain distinguishing channels and differences, count bounds, or matching deficits, without repeatedly exporting full feature witnesses.
- `summary.json`: a denominator of 400 per setting, with separate counts for certified differences, certified equalities, unresolved cases, resource-limited cases, and unexecuted cases. A larger lower bound on detected differences does not by itself establish greater true expressive power.
- `manifest.json`: input, code, configuration, and environment identities, plus run status.

Evidence summaries explain the results; they are not complete certificates that can be verified independently of the inputs. The new program covers only pair-domain BREC, not full triple/joint encodings or neural outputs. The finite triple results from `mechanisms` do not substitute for experiments in those scopes.

All BREC entry points continue to fix random seeds, single-threaded numerical libraries, and strict PyTorch deterministic algorithms. The historical neural BREC workflow retains fixed physical batch sizes, dtype, and GPU assignments, with no automatic computation-path changes after out-of-memory errors. Repeated runs with the same inputs, configuration, software, and hardware are reported separately from relabeling checks.

To verify repeatability, the following command runs full BREC twice, then directly compares result records and evidence summaries:

```bash
bash tools/run_rebuttal.sh brec-reproduce
# Or compare two separately completed v2 BREC outputs:
python -m brec.reproducibility --first outputs/brec_run1 --second outputs/brec_run2 --out outputs/brec_repeat.json
```

The comparison excludes elapsed time. Incomplete or resource-limited runs cannot pass. `passed: true` means only that records agree for the requested pair-domain BREC conditions. It does not check neural weights, guarantee agreement across platforms, or provide a mathematical proof for arbitrary relabelings.

### CUDA execution

Install CuPy in the same Python environment as the BREC dependencies. Select the wheel for your NVIDIA driver and CUDA version. [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)

```bash
# Example for a server using CUDA 12
python -m pip install 'cupy-cuda12x[ctk]'
export BREC_FILE=/server/data/BREC/brec_v3.npy

# Smoke test on a few official pairs
REBUTTAL_OUT=outputs/brec_cuda_smoke bash tools/run_rebuttal.sh brec \
  --device cuda:0 --workers 2 --pair-ids 0 60 140 320

# Full run
REBUTTAL_OUT=outputs/brec_cuda bash tools/run_rebuttal.sh brec --device cuda:0 --workers 2
```

CUDA computes the sparse adjacency products in the polynomial recurrence with multiword integers. Regular graphs use common-denominator numerators. Irregular graphs round each endpoint product outward on the same dyadic grid as the CPU reference. Channel construction, moment accumulation, multiset decisions, and control comparisons run on the CPU. The startup probe compares sample CUDA products with the reference, and the manifest records the device and CUDA/CuPy versions. Resume with the same device and worker settings.

Use the same pair subset and worker count on CPU and CUDA to measure elapsed time. The speedup depends on graph sizes and the fraction of runtime spent in polynomial products. `--memory-gib` covers host RAM, and `--workers` limits concurrent CUDA contexts on the selected device.

### Exact 2-FWL pair-colour encoding

```bash
BREC_FILE=/data/BREC/brec_v3.npy REBUTTAL_OUT=outputs/pair_colours \
  bash tools/run_rebuttal.sh fwl2-encoding --pair-ids 260 261 262 --workers 3
```

The encoding refines ordered-pair colours to stability using the joint colours through a shared vertex. Each graph has its own canonical colour dictionaries. Together with the dictionaries, the colour counts are comparable across graphs. Local colour IDs index their graph's dictionary.

Each NPZ file stores the two stable pair-colour matrices, round dictionaries and exact multiplicities. The JSON record gives the first differing colour count and the checks for all 32 official BREC reliability pairs. The command uses CPU integer sorting and shows per-pair progress.

### Numerical 2-FWL pair encoding

Use `fwl2-encoding --numerical-rounds 5` for the fixed-width numerical variant. Each round applies two pointwise polynomials to the previous pair matrix and multiplies the transformed matrices across their shared pivot. The result is added to the previous state in each of two modular integer channels. Every graph uses the same constants and can be encoded independently.

The output contains the initial pair-type channel and the two channels from each round. Five rounds give 11 channels. Choose a common depth for the graphs being compared. The `extend_numeric_pairs` API appends further rounds to saved features using the last state. The readout compares exact multisets of numerical rows and records the earliest distinguishing prefix. The official BREC reliability checks compare the aligned numerical tensors.

At round `t`, each numerical state is a function of the exact round-`t` 2-FWL colour. The initial state records the pair type. The two input states at each pivot are determined by the corresponding colours, so their contracted sum is determined by the joint colour counts. A distinction in the numerical rows therefore implies a 2-FWL distinction. Modular features can merge distinct colours, and a finite depth bounds the refinement, so measured coverage refers to the evaluated cohort and depth.

### Progressive 2-FWL pair encoding

```bash
source ~/miniforge3/bin/activate gnn270
BREC_FILE=/data/BREC/brec_v3.npy REBUTTAL_OUT=outputs/brec_fwl2_progressive \
  bash tools/run_rebuttal.sh fwl2-encoding --full --workers 20
```

`--full` selects the 270 pairs that standard 2-FWL distinguishes (IDs `0–109, 160–319`). For a subset, use `--progressive --pair-ids ...`. The defaults are `--base-q 16 --basis monomial --variant endpoint --numerical-rounds 5`.

Layer 0 compares the original ordered-pair features with exact arithmetic. Subsequent layers retain those features and append two numerical channels to the same rows. Channels from the last layer are reused in the next contraction. A difference in the full row multiset persists at every higher layer because projecting the higher-layer rows recovers the earlier multiset. The experiment stops at the first difference or records `unresolved` at the depth limit.

Only the 32 official isomorphic BREC controls per pair are evaluated. The checks align the exact polynomial banks and the numerical features through the stopping depth for that main pair. The progress bar advances after each pair and its controls. The summary reports evaluated and newly distinguished pairs per layer, official controls passed, and the number of numerical rounds executed across the main pairs. The NPZ contains the input adjacencies and any computed numerical prefixes. The original features can be reconstructed from the adjacencies using the baseline settings in the manifest.

### Ordered 4-FWL encoding on small BREC graph pairs

```bash
source ~/miniforge3/bin/activate gnn270
export BREC_FILE=/data/BREC/brec_v3.npy
export REBUTTAL_OUT=outputs/brec_fwl4_ordered_small_reproduction

bash tools/run_rebuttal.sh fwl4-encoding --max-nodes 100 --workers 2 --preflight-only
bash tools/run_rebuttal.sh fwl4-encoding --max-nodes 100 --workers 2
```

The node cap selects IDs `343, 348, 355, 358` from the 20 BREC pairs whose exact 3-FWL reference stabilizes with equal histograms. The command reads `outputs/brec_fwl3_reference_340_359.json` by default and validates its dataset hash. Pass `--reference-3fwl PATH` if the audit is stored elsewhere. It runs exact ordered 4-FWL on the selected candidates and evaluates the encoding and official controls on the pairs the reference distinguishes. It starts from the exact `Q=1` normalized pair and shared-pivot joint channels lifted to four coordinates. The numerical extension retains the moduli and integer precision of the 3-FWL experiment, updates all four replacement coordinates through a shared pivot, and stops at the first distinguishing prefix.

The preflight checks host and cgroup memory, reserves 8 GiB, and allows 10 GiB of address space per worker. The implementation retains an `int32` orbit lookup for ordered quadruples and streams exact pivot profiles in blocks of 4,096 orbit rows. It compares weighted full-row multisets over all ordered quadruples. The 32 official controls per eligible pair check the prefix through that pair's stopping depth. The summary records reference membership, the layers actually used, official control counts, and peak worker RSS.

### Reuse across tuple orders

`bash tools/run_rebuttal.sh budget-extend` runs the pair and triple supplements with a shared cache at `$REBUTTAL_OUT/polynomial_banks`. Add `--device cuda:0` to use CUDA in both stages. When running `brec` and `triple-encodings` individually, pass the same `--bank-cache DIRECTORY` to both. Entries use the labeled adjacency and basis as their key. For a regular graph evaluated with exact rationals in the pair experiment, the triple experiment reads the same integer polynomial numerators and extends them if it needs a larger Q. The pair interval path retains its configured precisions, and the triple experiment generates its exact numerators for those graphs.

The triple readout also shares a pair channel's marginal moment sums across coordinate positions. Each ordered pair occurs exactly (n) times in any two distinct positions of the full ordered triple domain. Full multiset comparisons still use complete triple rows with their original multiplicities.

### Full pair and triple run

Pass `--full` to run the union of the existing grids in one invocation. Each row includes both bases, both variants for that tuple order, and the full multiset readout at every Q.

| Tuple order K | Q | Mean-moment P | Conditions per graph pair |
|---|---|---|---:|
| 2 | 2, 4, 8, 16 | 1, 2, 4, 8 | 80 |
| 3 | 1, 2, 4 | 1, 2, 4, 8, 16 | 72 |
| 3 | 8, 16 | 1, 2, 4, 8 | 40 |

Both full configurations use the complete official reliability block for each BREC graph pair. They check all 32 adjacent pairs of same-graph relabelings stored in the 64-entry block, identified by dataset indices in the result records. The pair stage checks all aligned entries using exact equality or interval overlap. The triple stage checks all aligned bank entries and the complete feature rows of up to 16 sampled triples per comparison. BREC's neural evaluation computes a Hotelling statistic from the same reliability block. [Official BREC implementation](https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC.py#L160-L219)

```bash
REBUTTAL_OUT=outputs/rebuttal_full \
  bash tools/run_rebuttal.sh budget-extend --full \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 --workers 4
```

Each stage spreads graph-pair jobs across all four devices. With four workers, each device handles at most one active job, subject to the host memory budget. The triple joint channels use CUDA for exact integer products and degree-group sums. Canonical radical conversion, multiset checks, and power readouts use the CPU. Run with a CUDA 12.8 or newer toolkit for GeForce RTX 5090 cards. [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)

All 400 pairs are selected by default, for 76,800 condition records. The run stores results in `brec/` and `triple_encodings/` under the chosen output root. Both stages use the shared polynomial bank cache. Repeat the same command with `--resume` to continue.

## Protein prediction

**Co-Sparsify data/training pipeline + GNNPlus GINE model.** Co-Sparsify is pinned to commit `505f6b50f019dc78edc88653d771667c0c561f5f`. Its TU loader, ten-fold splits, Lightning training, optimizer, scheduler, `val/acc` model selection, and evaluator are reused. `protein_gnnplus.py` replaces the original SPPGN1 with `GINEConvLayer` (hidden dimension 256, 3 layers) and `SANGraphHead` (3 layers, mean pooling, ReLU, dropout 0) from `gnnplus_official`. `pair_x` is mapped to the original directed edges by `(u,v)`, and `loop_x` is concatenated with the node inputs. The upstream files in both projects remain unchanged.

Sources: [entry point](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/run_tud.py), [TU configuration](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/configs/sppgn/tud.sppgn1.poly.yaml), [splits](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/datasets/kfold.py), and [trainer](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/lighting_interface/model.py).

| Item | Official setting reused |
|---|---|
| Data | `PROTEINS_full` and `ENZYMES`, using the non-cleaned versions. Continuous node/edge attributes are disabled by default in the original PyG loader; no extra constant columns or normalization are added |
| Splits | 10 folds; `StratifiedKFold(shuffle=True, random_state=42)` when the original `10fold` file is absent. Fold r uses the original code's `idx_list[r−1]`. Existing files are reused directly, and actual indices are recorded |
| Initialization | Fold r uses seed=(r+1)×1000; no additional repetitions with three seeds |
| Network | GNNPlus GINE, 3 layers, hidden dimension 256, residual connections, dropout 0, mean pooling, and a 3-layer SAN graph head |
| Encoding widths | 8, 12, 14, and 16 from the official script; all encoding models have the same parameter count at a given width |
| Training | Physical batch size explicitly overridden to 64; training uses `drop_last=True`, AdamW, lr=0.001, weight_decay=0, 100 epochs, 5 warmup epochs, the original cosine scheduler, and norm gradient clipping at 1.0 |
| Model selection | Maximum original `val/acc`; the evaluator's five-decimal rounding and original checkpoint tie handling are preserved, without switching to NLL |
| Final scoring | The original script uses the same held-out fold for validation and test, as explicitly requested by the user. This is not reported as an independent test estimate |
| Aggregation | Mean and population standard deviation (ddof=0) of the 10 fold scores for each encoding/width, retaining original fields such as `test_acc`; raw logits and indices are saved |
| Execution | Original Co-Sparsify GPU Trainer, 8 torch threads, and a 4-week Timer; the model calls native GNNPlus GINE layers and the SAN head, and Comet is replaced by `local_log` |

The original scheduler interprets min_lr=1e-6 according to its own code; no separate learning-rate implementation is introduced. The original DataLoader handles physical batches and `drop_last`, without simulating them through per-graph gradient accumulation. `configs/rebuttal/protein.json` explicitly sets `batch_size_override: 64`, which is written to every run manifest and record. The GNNPlus model does not require SPPGN1-specific `2fwl_connsp` triple indices, so that transform is the identity for this backend. All structural encodings are still generated in the poly transform's `pair_x`/`loop_x` fields.

### Nine encoding groups

For width K, let r=⌊K/2⌋, e=⌊(K−r)/2⌋, and j=K−r−e. For K=8/12/14/16, (r,e,j) is (4,2,2), (6,3,3), (7,3,4), and (8,4,4), respectively. M is the symmetrically normalized adjacency matrix, B_q=M^q, and h_q=B_q 1.

| Setting | K-dimensional encoding received by GNNPlus GINE |
|---|---|
| `rrwp` | Native Co-Sparsify RRWP, I,P,…,P^(K−1), where P is the row-normalized adjacency matrix; the baseline uses the original transform throughout |
| `mono` | Native I,M,…,M^(K−1) |
| `cheb` | Native full Chebyshev basis spanning the same polynomial degree space as mono |
| `none` | K zero slots; the model and native node/edge inputs are unchanged |
| `base` | First r monomial channels, with the remaining slots zero-padded |
| `endpoint` | base plus the first e entries of [h₁(u),h₁(v),h₂(u),h₂(v)], with the remaining j slots zero-padded |
| `powers` | endpoint plus the first j elementwise squares of retained B_q(u,v) channels; adds no raw information to this encoding |
| `joint` | endpoint plus J₁,…,Jⱼ, retaining all endpoint slots |
| `joint_mix` | Full K-dimensional orthogonal DCT change of basis of joint, without truncation or additional raw information |

The additional joint pair channels are obtained through a fixed triple readout:

\[
J_q(u,v)=A_{uv}\sum_w A_{uw}A_{vw}\sum_t B_q(u,t)B_q(v,t)B_q(w,t).
\]

The inner sum is a triple shared-pivot construction; the outer sum uses a triangle condition for a fixed readout through the original pair interface, with no additional trainable modules. On graphs without self-loops, J₁ involves four distinct, pairwise adjacent vertices and retains a signal related to 4-cliques. It is not the original full multiset of triple features, and no equivalence to full 3-FWL is claimed. The program holds only one local plane at a time, without storing a full n³ cache. All new encodings retain the original transform's float32 interface; no feature normalization is added.

The primary comparison is `joint − endpoint` at matching K, with additional prespecified comparisons against `powers`, native `mono`, and other settings. joint explicitly retains endpoint. At the same K, joint and full mono allocate their channel budgets differently, so general dominance is not assumed. `cheb/mono` and `joint_mix/joint` each control for different representations of the same information. All configurations are reported; no "independent test results after selecting the best encoding per fold" are generated.

### Running on the server

Use the original Co-Sparsify environment, following its README instructions for mamba, Sage, PyG, and CUDA extensions. The reference environment uses Python 3.11, NumPy 1.26, and PyTorch 2.2.2/CUDA 12.1. Keep the BREC NumPy 2.4 dependencies out of this environment. The adapter requires no packages beyond those in the original repository.

If the original repository is not yet on the server, prepare the pinned version first; reuse an existing checkout when available:

```bash
git clone https://github.com/RongqinChen/Co-Sparsify.git Co-Sparsify
git -C Co-Sparsify checkout 505f6b50f019dc78edc88653d771667c0c561f5f
```

Then run from this reproducibility repository's root:

```bash
mamba activate gnn222
export TU_DATA_ROOT=/server/data/TUD
CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-both 2>&1 | tee cosparsify_tu.log
```

The original loader downloads missing data through its usual workflow; no additional `--download` flag is needed. The fixed matrix is 2 datasets × 9 encodings × 4 widths × 10 folds: **720 native training runs** in total, 360 per dataset. The physical batch size is fixed at 64 in the current configuration. The runner does not automatically reduce the batch size, change precision, replace the model, or drop graphs to work around other failures.

You can also start with a small subset of formal conditions from the full matrix, then continue in the same directory without rerunning completed conditions:

```bash
PROTEIN_DATASET=ENZYMES CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-run --encodings rrwp --poly-dims 8 --folds 0
# Continue with the remaining conditions for the same dataset:
PROTEIN_DATASET=ENZYMES CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-run
PROTEIN_DATASET=ENZYMES bash tools/run_rebuttal.sh protein-report
```

To run the full matrix in the background, use the shard scheduler with separate encoding × width shards. It first warms up both TU datasets in the same native environment, then divides each dataset into 36 shards and assigns them to 4 GPUs within a fixed concurrency limit. Once all shards finish, it validates and merges the ten-fold reports:

```bash
python tools/run_protein_grid.py \
  --python "$(command -v python)" \
  --reference-root Co-Sparsify \
  --data-root "$TU_DATA_ROOT" \
  --output-root outputs/cosparsify_tu_grid \
  --gpus 0 1 2 3 \
  --workers-per-gpu 1
```

`--workers-per-gpu` defaults to 1; increase it only after measuring the original model's GPU and host memory requirements. Without `--resume`, existing output will not be overwritten; add the flag when continuing a run. Use `--dry-run` first to inspect the plan for 72 shards / 720 native training runs. Example background launch:

```bash
nohup setsid python tools/run_protein_grid.py \
  --reference-root Co-Sparsify --data-root "$TU_DATA_ROOT" \
  --output-root outputs/cosparsify_tu_grid \
  --gpus 0 1 2 3 > cosparsify_tu_grid.log 2>&1 < /dev/null &
```

The runner writes progress to `run_status.json` in the output root. Each shard still calls the existing `protein run` entry point, without copying or rewriting the Co-Sparsify training pipeline. The final `protein report --shards` rejects artifacts with mismatched identities, duplicate records, missing combinations, or checksum mismatches.

Use `--reference-root` to specify another identical, unmodified official checkout. Each new TU directory is `$REBUTTAL_OUT/cosparsify_tu/<dataset_name>/`, containing `records/`, `predictions/`, original checkpoints and local logs under `native/`, and the final `protein_report.json`. Logs retain the original preprocessing-time output; JSON retains original metrics such as `avg_train_time_epoch`. Earlier custom-model results with 27/64 slots are kept separately and excluded from this comparison.

Completed conditions are skipped only after their experiment identities and artifacts have been checked. Full aggregation then checks that actual data, indices, initialization, model parameter counts, and original settings agree across encoding records. Changing the requested subset of encodings, widths, or folds does not change the experiment identity, allowing the same matrix to be completed in batches. Removing the old custom loader/model/trainer does not generate an archive; BREC and mechanism code is unchanged and does not require rerunning.

## Code reading order

- BREC: `brec.py` orchestrates runs, `scheduling.py` schedules parallel tasks, `encodings.py` generates channels, and `readouts.py` performs discrimination.
- TU: `protein.py` calls the original entry point and saves provenance and results, `protein_encodings.py` replaces only the encodings, and `protein_report.py` summarizes original scores. The model, loader, and trainer are sourced from the external Co-Sparsify checkout.

Each run's `manifest.json` records configuration, input, code, and environment identities. The repository no longer retains old result tables, static source inventories, or archive export tools.

## BREC code validation

```bash
python -m unittest discover -s tests -p 'test_brec_*.py' -v
```

Tests cover rigorous interval enclosures, independent fraction formulas, both bases and all default Q budgets, cumulative mean moments, triple lifts, repeated-row matching, non-self-inverse relabelings, the official reliability mapping, parallel and repeated-run consistency, memory limits, interruption cleanup, atomic saving, and resume identities. Configuration errors are rejected before workers start. Reproducibility checks also verify condition coverage, record identities, and summary consistency. These tests use small graphs and do not replace running and reviewing all formal BREC conditions.
