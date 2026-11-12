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
PROTEIN_CONFIG="${PROTEIN_CONFIG:-configs/rebuttal/protein.json}"
PROTEIN_OUT="${PROTEIN_OUT:-$REBUTTAL_OUT/protein_encodings/$PROTEIN_DATASET}"
TU_DATA_ROOT="${TU_DATA_ROOT:-data/tu}"
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
  protein-prepare)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein prepare --config "$PROTEIN_CONFIG" --dataset "$PROTEIN_DATASET" \
      --data-root "$TU_DATA_ROOT" --out "$PROTEIN_OUT/protein_inputs" --resume "${@:2}"
    ;;
  protein-train)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein train --config "$PROTEIN_CONFIG" --prepared "$PROTEIN_OUT/protein_inputs" \
      --device "$PROTEIN_DEVICE" --out "$PROTEIN_OUT/protein_training" --resume "${@:2}"
    ;;
  protein-predict)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein predict --prepared "$PROTEIN_OUT/protein_inputs" \
      --run "$PROTEIN_OUT/protein_training" --device "$PROTEIN_DEVICE" "${@:2}"
    ;;
  protein-report)
    "$REBUTTAL_PYTHON" -m rebuttal_experiments protein report --prepared "$PROTEIN_OUT/protein_inputs" \
      --runs "$PROTEIN_OUT/protein_training" --out "$PROTEIN_OUT/protein_report.json" "${@:2}"
    ;;
  protein-both)
    # Run sequentially on one GPU; optional arguments apply to preparation only.
    for dataset in PROTEINS ENZYMES; do
      for stage in prepare train predict report; do
        prepare_args=()
        if [[ "$stage" == prepare ]]; then
          prepare_args=("${@:2}")
        fi
        PROTEIN_DATASET="$dataset" PROTEIN_OUT="$REBUTTAL_OUT/protein_encodings/$dataset" \
          bash tools/run_rebuttal.sh "protein-$stage" "${prepare_args[@]}"
      done
    done
    ;;
  *)
    echo 'Usage: bash tools/run_rebuttal.sh {mechanisms|brec|brec-reproduce|protein-prepare|protein-train|protein-predict|protein-report|protein-both}' >&2
    exit 2
    ;;
esac
