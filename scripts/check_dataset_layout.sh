#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
TEXT_ROOT="${TEXT_ROOT:-${PSD2_TEXT_DATA_ROOT:-${PROJECT_ROOT}/data_text}}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT or PSD2_DATA_ROOT must be set." >&2
  exit 1
fi

declare -a RAW_REQUIRED_PATHS=(
  "${DATA_ROOT}/PRW/query_info.txt"
  "${DATA_ROOT}/PRW/frame_train.mat"
  "${DATA_ROOT}/PRW/frame_test.mat"
  "${DATA_ROOT}/PRW/frames"
  "${DATA_ROOT}/PRW/annotations"
  "${DATA_ROOT}/cuhk_sysu/annotation/Images.mat"
  "${DATA_ROOT}/cuhk_sysu/annotation/pool.mat"
  "${DATA_ROOT}/cuhk_sysu/annotation/test/train_test/Train.mat"
  "${DATA_ROOT}/cuhk_sysu/annotation/test/train_test/TestG100.mat"
  "${DATA_ROOT}/cuhk_sysu/Image/SSM"
)

declare -a TEXT_REQUIRED_PATHS=(
  "${TEXT_ROOT}/PRW/generated_schema.json"
  "${TEXT_ROOT}/PRW/prw_Final_Complete_cleaned_train.json"
  "${TEXT_ROOT}/PRW/prw_Final_Complete_cleaned_test.json"
  "${TEXT_ROOT}/cuhk_sysu/generated_schema.json"
  "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_train_cleaned.json"
  "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_test_cleaned.json"
)

echo "Checking raw dataset layout under: ${DATA_ROOT}"

for path in "${RAW_REQUIRED_PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "[OK] ${path}"
  else
    echo "[MISSING] ${path}" >&2
    exit 1
  fi
done

echo "Checking bundled text assets under: ${TEXT_ROOT}"

for path in "${TEXT_REQUIRED_PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "[OK] ${path}"
  else
    echo "[MISSING] ${path}" >&2
    exit 1
  fi
done

echo "Dataset layout check passed."
