#!/usr/bin/env bash
# Run explicitly on the server. No implicit "all" or automatic training.
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
case "${1:-}" in
  protein-*) ;; # The native TU script sets torch threads to eight itself.
  *)
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    ;;
esac
REBUTTAL_PYTHON="${REBUTTAL_PYTHON:-python}"
REBUTTAL_OUT="${REBUTTAL_OUT:-outputs/rebuttal}"
PROTEIN_DATASET="${PROTEIN_DATASET:-ENZYMES}"
PROTEIN_CONFIG="${PROTEIN_CONFIG:-configs/rebuttal/protein.json}"
PROTEIN_OUT="${PROTEIN_OUT:-$REBUTTAL_OUT/cosparsify_tu/$PROTEIN_DATASET}"
TU_DATA_ROOT="${TU_DATA_ROOT:-data/TUD}"
run_python() {
  if [[ "${REBUTTAL_PROGRESS:-1}" == 0 ]]; then
    "$REBUTTAL_PYTHON" -u -m tools.rebuttal_progress --no-progress -- "$REBUTTAL_PYTHON" -u "$@"
  else
    "$REBUTTAL_PYTHON" -u -m tools.rebuttal_progress -- "$REBUTTAL_PYTHON" -u "$@"
  fi
}

case "${1:-}" in
  mechanisms)
    run_python -m rebuttal_experiments mechanisms --out "$REBUTTAL_OUT/mechanisms"
    ;;
  brec|fwl2-encoding|fwl3-encoding|fwl4-encoding)
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    order=all
    if [[ "$1" != brec ]]; then order="${1:3:1}"; fi
    "$REBUTTAL_PYTHON" -u reproduce.py brec --brec-file "$BREC_FILE" \
      --out "$REBUTTAL_OUT/brec" --order "$order" --workers "${BREC_WORKERS:-1}" \
      --threads "$OMP_NUM_THREADS" "${@:2}"
    ;;
  objects)
    : "${MUTAG_TOPOLOGY:?Set MUTAG_TOPOLOGY to the 188-graph topology.npz on the server}"
    run_python -m rebuttal_experiments objects --topology "$MUTAG_TOPOLOGY" --out "$REBUTTAL_OUT/objects"
    ;;
  protein-prepare)
    run_python -m rebuttal_experiments protein prepare --dataset "$PROTEIN_DATASET" \
      --data-root "$TU_DATA_ROOT" --out "$REBUTTAL_OUT/protein_inputs"
    ;;
  protein-train)
    run_python -m rebuttal_experiments protein train --prepared "$REBUTTAL_OUT/protein_inputs" \
      --device "$PROTEIN_DEVICE" --out "$REBUTTAL_OUT/protein_training"
    ;;
  protein-predict)
    run_python -m rebuttal_experiments protein predict --prepared "$REBUTTAL_OUT/protein_inputs" \
      --run "$REBUTTAL_OUT/protein_training" --device "$PROTEIN_DEVICE"
    ;;
  protein-report)
    run_python -m rebuttal_experiments protein report --prepared "$REBUTTAL_OUT/protein_inputs" \
      --runs "$REBUTTAL_OUT/protein_training" --out "$REBUTTAL_OUT/protein_report.json"
    ;;
  *)
    echo 'Usage: bash tools/run_rebuttal.sh {mechanisms|brec|fwl2-encoding|fwl3-encoding|fwl4-encoding|objects|protein-prepare|protein-train|protein-predict|protein-report}' >&2
    exit 2
    ;;
esac
