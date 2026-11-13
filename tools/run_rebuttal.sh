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
BREC_ARGS=()
if [[ -n "${BREC_WORKERS:-}" ]]; then
  BREC_ARGS+=(--workers "$BREC_WORKERS")
fi

case "${1:-}" in
  mechanisms)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments mechanisms --out "$REBUTTAL_OUT/mechanisms"
    ;;
  brec)
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    "$REBUTTAL_PYTHON" -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec" "${BREC_ARGS[@]}" "${@:2}"
    ;;
  brec-reproduce)
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    : "${BREC_FILE:?Set BREC_FILE to the official brec_v3.npy on the server}"
    "$REBUTTAL_PYTHON" -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec_run1" "${BREC_ARGS[@]}" "${@:2}"
    "$REBUTTAL_PYTHON" -m rebuttal_experiments brec --brec-file "$BREC_FILE" --out "$REBUTTAL_OUT/brec_run2" "${BREC_ARGS[@]}" "${@:2}"
    "$REBUTTAL_PYTHON" -m brec.reproducibility --first "$REBUTTAL_OUT/brec_run1" --second "$REBUTTAL_OUT/brec_run2" --out "$REBUTTAL_OUT/brec_reproducibility.json"
    ;;
  protein-run)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein run --config "$PROTEIN_CONFIG" \
      --dataset "$PROTEIN_DATASET" --data-root "$TU_DATA_ROOT" --out "$PROTEIN_OUT" --resume "${@:2}"
    ;;
  protein-report)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein report --out "$PROTEIN_OUT" "${@:2}"
    ;;
  protein-both)
    for dataset in PROTEINS_full ENZYMES; do
      PROTEIN_DATASET="$dataset" PROTEIN_OUT="$REBUTTAL_OUT/cosparsify_tu/$dataset" \
        bash tools/run_rebuttal.sh protein-run "${@:2}"
      PROTEIN_DATASET="$dataset" PROTEIN_OUT="$REBUTTAL_OUT/cosparsify_tu/$dataset" \
        bash tools/run_rebuttal.sh protein-report
    done
    ;;
  *)
    echo 'Usage: bash tools/run_rebuttal.sh {mechanisms|brec|brec-reproduce|protein-run|protein-report|protein-both}' >&2
    exit 2
    ;;
esac
