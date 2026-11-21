# Additional experiment entry points

Use the [unified run guide](rebuttal_experiments.md) for supplementary rebuttal experiments. The commands below cover the retained workflows for broader datasets and models. Prepare the corresponding environment using [installation.md](installation.md) before running. The repository does not bundle historical results; all reports are generated from the specified new output directories.

## Object discrimination

Datasets and object domains are configured in `configs/object_datasets.json`:

```bash
python reproduce.py list
python reproduce.py prepare --dataset qm9 --data-root data/QM9 --topology topology/qm9.npz
python reproduce.py objects --dataset qm9 --topology topology/qm9.npz \
  --out outputs/objects/qm9 --workers 12 --memory-gib 100
python tools/summarize_object_results.py outputs/objects/qm9 \
  --tag rerun --out outputs/qm9_summary.json
```

Adjust `--workers` and `--memory-gib` to the server's resources. Use `--resume` to continue with the same code, inputs, and configuration; use a new output directory after changing the scientific settings. Loading a dataset does not replace checking the coverage of the inputs.

## BREC encodings

See the [BREC float32 run guide](brec_suite.md) for the graph-pair experiments.
