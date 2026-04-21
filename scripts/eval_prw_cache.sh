#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
FUSION_MODE="${FUSION_MODE:-avg}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-1024}"
SCORE_THRESH="${SCORE_THRESH:-0.5}"
FULL_SCORE_MATRIX="${FULL_SCORE_MATRIX:-1}"
GALLERY_FILE="${GALLERY_FILE:-}"
QUERY_FILE="${QUERY_FILE:-}"

if [[ -z "${GALLERY_FILE}" || -z "${QUERY_FILE}" ]]; then
  echo "GALLERY_FILE and QUERY_FILE must be set." >&2
  exit 1
fi

ARGS=(
  --dataset prw
  --gallery-file "${GALLERY_FILE}"
  --query-file "${QUERY_FILE}"
  --score-thresh "${SCORE_THRESH}"
  --fusion-mode "${FUSION_MODE}"
  --device "${DEVICE}"
  --query-batch-size "${QUERY_BATCH_SIZE}"
)

if [[ "${FULL_SCORE_MATRIX}" == "1" ]]; then
  ARGS+=(--full-score-matrix)
fi

"${PYTHON_BIN}" tools/eval_search_from_cache.py "${ARGS[@]}"
