# BREC through reproduce.py

Use the existing repository entry point. The implementation is entirely inside the existing `brec/`, `configs/`, and `requirements/` directories. It has no dependency on the separate draft directory or archive.

## Install

Python 3.12 on Linux is recommended. Install in the environment used for this campaign before creating results:

```bash
python -m pip install -r requirements/brec_suite.txt
python -m pip install 'torch>=2.7,<3' --index-url https://download.pytorch.org/whl/cpu
```

## Run all BREC experiments

```bash
python reproduce.py brec-run \
  --brec-file /data/BREC/brec_v3.npy \
  --root /results/brec_exact_v2 \
  --workers 44 --memory-gib 100
```

The default campaign includes input/reference validation, exact pair and triple dictionaries, Q=1 equal-width comparisons, exact mechanism controls, numerical sensitivity diagnostics, and 2,400 CPU neural runs. The main structural grid has 160 settings over all 400 pairs; the Q=1 supplement adds 32 settings. No reference label is used to skip a measured pair. All configured reliability controls are part of each relevant job.

The default limits reserve 8 GiB outside the campaign, cap the coordinator at 4 GiB, and admit at most 88 GiB of worker allocations. At most 44 single-thread workers are active; memory requirements and CPU/cgroup limits can reduce concurrency. Linux address-space limits enforce per-process allocations. Temporary pressure pauses admission. Native allocation failures remain failed/resource-limited results. The selected neural physical batch is independent of concurrency and retry memory limits.

Reissue the command to resume completed work. Retry unsuccessful jobs explicitly:

```bash
python reproduce.py brec-run --root /results/brec_exact_v2 \
  --brec-file /data/BREC/brec_v3.npy \
  --retry-incomplete --job-timeout 86400 --job-memory-scale 2
```

## Optional separate planning and stages

```bash
python reproduce.py brec-plan --brec-file /data/BREC/brec_v3.npy --root /results/brec_exact_v2
python reproduce.py brec-run --root /results/brec_exact_v2 --workers 44 --memory-gib 100
python reproduce.py brec-report --root /results/brec_exact_v2 --verify-artifacts --strict
```

Planning writes `plan.json` and does not execute experiments. `--dry-run` describes the selection without writing or executing. `--stage reference|mechanism|exact|diagnostic|neural` restricts execution; the default is `all`. `--pairs 0-399` selects pairs explicitly. Reports keep the 400-pair denominator and mark unrequested cases.

Use a new output root for the revised protocol. Existing legacy frozen plans remain intact and cannot be mistaken for the new exact protocol.

## Reliability by construction

Structural features use exact rational and canonical radical arithmetic, including irregular graphs. Tuple multisets retain exact multiplicities, and moment sums are order-independent exact sums. The integer fast path is allowed only after a proven overflow bound. All ordered pairs/triples, including repetitions, are represented. Orbit compression uses integer weights that sum to the full domain size; orbit IDs never enter features.

Each scheduled relabeling also compares the original graph's complete multiset and moment aggregates with its relabeled copy. A successful cross-graph decision alone does not satisfy that requirement. Exact distinctions carry explicit witnesses; incomplete computations never become negative results.

Neural input features are exactly equivariant before float32 conversion. Neural graph outputs remain floating-point quantities: all scheduled outputs within each isomorphism group must satisfy the fixed symmetric tolerance `abs(a-b) <= 1e-6 + 1e-5*max(abs(a),abs(b))`. A mismatch leaves the neural result unresolved. Exact bitwise neural invariance is not claimed.

Workers execute the frozen copy of `reproduce.py` and its BREC support modules. Dataset, source, dependency, configuration, and per-job identities are checked. Output writes are atomic; records and artifacts have checksums. Resume verifies completed artifacts. Interruption handling terminates and reaps owned workers. Every stage preserves failed, interrupted, unresolved, and unexecuted cases.

## Outputs

- `status.json`: active workers, reservations, memory observations, and progress.
- `records/`: exact witnesses, controls, neural decisions, and explicit failure statuses.
- `reports/exact_summary.csv`: per-setting counts and coverage.
- `reports/issues.json`: reference-bound or consistency flags.
- `reports/numerical_discrepancies.csv`: diagnostic disagreements.
- `reports/neural_summary.csv`: per-seed neural results and input comparisons.
- `artifacts/`: neural models and embeddings for rescoring.
- `source/`, `manifest.json`: frozen implementation and experiment identity.

No tests or experiments were executed during this repository integration. Numerical diagnostics and neural runs remain separate from exact structural certificates. The existing `prepare`, `objects`, and historical `ppgn` commands keep their original entry points.
