#!/usr/bin/env bash
# Run explicitly on the server. No implicit "all" or automatic training.
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
REBUTTAL_PYTHON="${REBUTTAL_PYTHON:-python}"
REBUTTAL_OUT="${REBUTTAL_OUT:-outputs/rebuttal}"
PROTEIN_DATASET="${PROTEIN_DATASET:-ENZYMES}"
PROTEIN_DEVICE="${PROTEIN_DEVICE:-cpu}"
TU_DATA_ROOT="${TU_DATA_ROOT:-data/tu}"

case "${1:-}" in
  mechanisms)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments mechanisms --out "$REBUTTAL_OUT/mechanisms"
    ;;
  brec)
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    "$REBUTTAL_PYTHON" -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec"
    ;;
  brec-reproduce)
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    "$REBUTTAL_PYTHON" -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec_run1"
    "$REBUTTAL_PYTHON" -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec_run2"
    "$REBUTTAL_PYTHON" -m brec.reproducibility --first "$REBUTTAL_OUT/brec_run1" --second "$REBUTTAL_OUT/brec_run2" --out "$REBUTTAL_OUT/brec_reproducibility.json"
    ;;
  objects)
    : "${MUTAG_TOPOLOGY:?Set MUTAG_TOPOLOGY to the 188-graph topology.npz on the server}"
    "$REBUTTAL_PYTHON" -m rebuttal_experiments objects --topology "$MUTAG_TOPOLOGY" --out "$REBUTTAL_OUT/objects"
    ;;
  protein-prepare)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein prepare --dataset "$PROTEIN_DATASET" \
      --data-root "$TU_DATA_ROOT" --out "$REBUTTAL_OUT/protein_inputs"
    ;;
  protein-train)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein train --prepared "$REBUTTAL_OUT/protein_inputs" \
      --device "$PROTEIN_DEVICE" --out "$REBUTTAL_OUT/protein_training"
    ;;
  protein-predict)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein predict --prepared "$REBUTTAL_OUT/protein_inputs" \
      --run "$REBUTTAL_OUT/protein_training" --device "$PROTEIN_DEVICE"
    ;;
  protein-report)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein report --prepared "$REBUTTAL_OUT/protein_inputs" \
      --runs "$REBUTTAL_OUT/protein_training" --out "$REBUTTAL_OUT/protein_report.json"
    ;;
  *)
    echo 'Usage: bash tools/run_rebuttal.sh {mechanisms|brec|brec-reproduce|objects|protein-prepare|protein-train|protein-predict|protein-report}' >&2
    exit 2
    ;;
esac
