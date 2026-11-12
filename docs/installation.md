# Installation

Use Python 3.12 on Linux. CPU scheduling uses POSIX locks and Linux memory
accounting; the PPGN scheduler requires NVIDIA CUDA devices. Start in the release root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Fixed BREC and its archived NumPy/NetworkX versions:
python -m pip install -r requirements.txt -c requirements/brec-constraints.txt

# PPGN, dataset preparation, and optional GNNPlus:
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/datasets.txt -c requirements/brec-constraints.txt
python -m pip install --no-index torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

The PyTorch build and extension index follow the official
[PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
and [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
For CPU-only dataset preparation, select the PyTorch `cpu` index and matching
`torch-2.7.0+cpu.html` extension index. GNNPlus's package imports dataset and
GraphGym dependencies eagerly, so install the full dataset requirements for its adapters.
Already-exported topology NPZs require only `requirements.txt` for object evaluation.

## Recorded environments

| Campaign | Python | NumPy | PyTorch | PyG | pynauty |
| --- | --- | --- | --- | --- | --- |
| ZINC subset / QM9 objects | 3.12.12 | 2.4.2 | 2.7.0+cu128 | 2.6.1 | 2.8.8.1 |
| TU / molhiv objects | 3.12.13 | 2.4.6 | 2.7.0+cu128 | 2.6.1 | 2.8.8.1 |
| Neural BREC | 3.12.13 | 2.4.3 | 2.7.0+cu128 | not required | not required |

For object campaigns replace `brec-constraints.txt` in the installation command
with `zinc-qm9-constraints.txt` or `tu-molhiv-constraints.txt`, preferably in
separate environments. These files pin the versions recorded in the supplied
artifacts. They are **partial environment records**, not a fully captured lockfile:
BLAS builds, all transitive dependencies, and the original GPU model are not
established by those records. Floating-point decisions need not reproduce bit for bit
on another numerical stack. Save `python -m pip freeze` with each new campaign.

## BREC data

Obtain `BREC_data_all.zip` from the [official BREC repository](https://github.com/GraphPKU/BREC),
extract `brec_v3.npy`, and put it in `data/brec_v3.npy`. The official repository
documents this as the graph6 input file used by most models.
The archived file has SHA256:

```text
6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562
```

```bash
sha256sum data/brec_v3.npy
```

The official NumPy object array needs the explicit `--trust-pickle` flag.
Use this flag only with a trusted dataset file. Input checksums are stored in
the experiment manifest. The code expects the 51,200-entry layout, including
the supplied isomorphic reliability block for neural evaluation.
