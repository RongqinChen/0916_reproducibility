# BREC v3 float32 experiments

The encoding and sorted-channel readout use `float32`. A pair is distinguished when the maximum absolute gap in a sorted channel is greater than \(10^{-8}\). All orders start from the symmetrically normalized adjacency matrix and reuse the same Walk features.

| Encoding | Pairs tested | Exact-reference positives | Last verified coverage | Official isomorphic controls |
| --- | ---: | ---: | ---: | ---: |
| 2-FWL | 400 | 270 | 270/270 | 12,800 |
| 3-FWL | 130 left equal by 2-FWL | 110 | 110/110 | 4,160 |
| 4-FWL | 18 first distinguished by 4-FWL | 18 | 18/18 | Outside this run |

The three target sets are `0–109, 160–319`, `110–159, 320–339, 360–399`, and `340–344, 346–355, 357–359`. Pairs `345` and `356` reach stable equality under the exact 4-FWL reference. These labels are frozen for the official `brec_v3.npy` checksum so the experiment can run from a fresh checkout. The 2- and 3-FWL runners also measure agreement on the reference-equal pairs in their tested cohorts.

## Run

```bash
source ~/miniforge3/bin/activate gnn270
python reproduce.py brec --brec-file /path/to/brec_v3.npy \
  --out outputs/brec_float32 --workers 14 --devices 0 1
```

Install `requirements/brec_suite.txt` when setting up a fresh environment. CUDA 12 runs also need `'cupy-cuda12x[ctk]'`. The two CUDA devices compute one graph each. For CPU execution omit `--devices` and set `--threads` to the available OpenMP threads for 4-FWL. That backend requires a C++17 compiler with OpenMP. The host limit for 4-FWL defaults to 58 GiB and can be set with `--memory-gib`.

Use `--order 2`, `--order 3`, or `--order 4` to rerun one order. For a 4-FWL subset, add `--ids 343 358`, for example. Each order writes to its own `fwl2`, `fwl3`, or `fwl4` subdirectory. Repeating the same command reuses completed records. The all-orders command also validates the completed lower-order records before running the next order.

The 3-FWL comparison tests static joint channels before recurrence. The 4-FWL runner directly computes every ordered quadruple and stops each comparison at the first distinguishing round. `progress-main.json`, `progress-official.json` where applicable, and `summary.json` record coverage and results.

## Code layout

The BREC experiment code lives in `brec/float32/`.

- `protocol.py` defines the reference cohorts and fixed experiment settings.
- `numerics.py` contains the Walk bank, Chebyshev transforms, and sorted readouts shared across orders.
- `pair.py`, `triple.py`, and `quadruple.py` implement the order-specific evaluations and summaries.
- `quadruple_backend.py` calls the C++ and CUDA implementations under `kernels/`.
- `runtime.py` manages workers, progress, and resumable records. `__main__.py` runs an individual order for `reproduce.py`.
