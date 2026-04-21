#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
TEXT_ROOT="${TEXT_ROOT:-${PSD2_TEXT_DATA_ROOT:-${PROJECT_ROOT}/data_text}}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/viper/PRW/clipviper_prw_stage1_baseline}"
CONFIG_FILE="${CONFIG_FILE:-configs/open_source/clipviper_prw_stage1_baseline.yaml}"

if [[ -n "${DATA_ROOT}" ]]; then
  export PSD2_DATA_ROOT="${DATA_ROOT}"
fi
export PSD2_TEXT_DATA_ROOT="${TEXT_ROOT}"

PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" tools/train_ps_net.py \
    --config-file "${CONFIG_FILE}" \
    --num-gpus="${NUM_GPUS}" \
    --dist-url auto \
    OUTPUT_DIR "${OUTPUT_DIR}"
