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
# Guard empty array expansion for Bash 3.x with nounset.
BREC_ARGS=()
if [[ -n "${BREC_WORKERS:-}" ]]; then
  BREC_ARGS+=(--workers "$BREC_WORKERS")
fi

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
  brec)
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    run_python -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec" ${BREC_ARGS[@]+"${BREC_ARGS[@]}"} "${@:2}"
    ;;
  fwl2-encoding)
    export PYTHONHASHSEED=0
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    "$REBUTTAL_PYTHON" -u -m brec.fwl2_encoding --brec-file "$BREC_FILE" \
      --out "$REBUTTAL_OUT/fwl2_encoding" ${BREC_ARGS[@]+"${BREC_ARGS[@]}"} "${@:2}"
    ;;
  budget-extend)
    export PYTHONHASHSEED=0
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    "$REBUTTAL_PYTHON" -u -m tools.rebuttal_budget_extension --brec-file "$BREC_FILE" \
      --out "$REBUTTAL_OUT" ${BREC_ARGS[@]+"${BREC_ARGS[@]}"} "${@:2}"
    ;;
  brec-reproduce)
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    run_python -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec_run1" ${BREC_ARGS[@]+"${BREC_ARGS[@]}"} "${@:2}"
    run_python -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec_run2" ${BREC_ARGS[@]+"${BREC_ARGS[@]}"} "${@:2}"
    run_python -m brec.reproducibility --first "$REBUTTAL_OUT/brec_run1" --second "$REBUTTAL_OUT/brec_run2" --out "$REBUTTAL_OUT/brec_reproducibility.json"
    ;;
  triple-encodings)
    export PYTHONHASHSEED=0
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    run_python -m rebuttal_experiments triple-encodings --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/triple_encodings" ${BREC_ARGS[@]+"${BREC_ARGS[@]}"} "${@:2}"
    ;;
  triple-ppgn)
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    run_python -m rebuttal_experiments triple-ppgn --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/triple_ppgn" "${@:2}"
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
    echo 'Usage: bash tools/run_rebuttal.sh {mechanisms|brec|fwl2-encoding|budget-extend|brec-reproduce|triple-encodings|triple-ppgn|objects|protein-prepare|protein-train|protein-predict|protein-report}' >&2
    exit 2
    ;;
esac
